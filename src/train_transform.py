from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import random
from contextlib import nullcontext

import numpy as np
import torch

from .circuit import QuantumCircuit
from .equiv_rewrite import load_equivalence_rules
from .env_transform import TransformEnv, TransformState
from .models import TransformHierarchicalAgent
from .ppo import gae_advantages, ppo_clip_objective


@dataclass
class TransformTrainConfig:
    episodes: int = 250
    max_steps: int = 80
    ppo_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_final_ratio: float = 0.3
    lr_final_ratio: float = 0.2
    rollout_batch_episodes: int = 2
    rollout_sample_temp_start: float = 1.15
    rollout_sample_temp_end: float = 0.9
    minibatch_size: int = 64
    target_kl: float = 0.03
    greedy_eval_episodes: int = 2
    hidden_dim: int = 64
    value_aggregate_mode: str = "max"
    next_value_mode: str = "max"
    enable_equiv_rewrite: bool = False
    equiv_lib_path: str = ""
    max_equiv_matches: int = 16
    max_dag_nodes: int = 1024
    dag_context: int = 24
    noop_penalty_start: float = 0.14
    noop_penalty_end: float = 0.05
    novelty_bonus_weight: float = 0.03
    equal_cost_change_bonus: float = 0.01
    oracle_rel_reward_weight: float = 1.0
    oracle_abs_reward_weight: float = 0.0
    cx_reduction_reward_weight: float = 0.20
    gate_reduction_reward_weight: float = 0.05
    new_best_bonus_weight: float = 0.10
    worse_cost_penalty_weight: float = 0.02
    best_metric: str = "oracle"
    candidate_sample_temperatures: Tuple[float, ...] = (0.75, 0.9, 1.0, 1.15, 1.3)
    candidate_refine_rounds: int = 1
    candidate_refine_topk: int = 3

    early_stop_enable: bool = True
    early_stop_warmup_ratio: float = 0.25
    early_stop_patience: int = 32
    early_stop_rel_delta: float = 5e-4
    early_stop_abs_delta: float = 1.0
    early_stop_eval_interval: int = 2
    use_amp: bool = True
    amp_dtype: str = "bf16"
    compile_model: bool = False
    seed: int = 0
    device: str = "cpu"


@dataclass
class TransformStep:
    state: TransformState
    group_idx: int
    action_idx: int
    logp_group: float
    logp_action: float
    reward: float
    value: float
    done: bool


@dataclass
class TransformTransition:
    tensors: Dict[str, object]
    group_idx: int
    action_idx: int
    old_logp_group: float
    old_logp_action: float
    old_value: float
    advantage: float
    ret: float


def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _use_cuda_amp(cfg: TransformTrainConfig) -> bool:
    return bool(cfg.use_amp) and str(cfg.device).startswith("cuda")


def _amp_context(cfg: TransformTrainConfig):
    if not _use_cuda_amp(cfg):
        return nullcontext()
    dtype = (
        torch.bfloat16
        if str(cfg.amp_dtype).lower() in {"bf16", "bfloat16"}
        else torch.float16
    )
    return torch.autocast(device_type="cuda", dtype=dtype)


def _transform_selection_score(
    circuit: QuantumCircuit, cost: float, cfg: TransformTrainConfig
) -> float:
    if str(cfg.best_metric) == "structure":
        base_cx = max(1.0, float(circuit.cx_count()))
        base_gates = max(1.0, float(circuit.gate_count()))
        return float(base_cx + 0.01 * base_gates)
    return float(cost)


def _state_to_tensors(state: TransformState, device: str) -> Dict[str, object]:
    return {
        "dag_node_features": _to_tensor(state.dag_node_features, device),
        "dag_adj": _to_tensor(state.dag_adj, device),
        "group_features": _to_tensor(state.group_features, device),
        "group_node_indices": [
            torch.tensor(x, dtype=torch.long, device=device)
            for x in state.group_node_indices
        ],
        "action_features": [_to_tensor(x, device) for x in state.action_features],
        "global_features": _to_tensor(state.global_features, device),
    }


def _sample_from_logits(
    logits: torch.Tensor, temperature: float = 1.0
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    safe_logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    t = max(1e-3, float(temperature))
    probs = torch.softmax(safe_logits / t, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action = int(dist.sample().item())
    return (
        action,
        dist.log_prob(torch.tensor(action, device=logits.device)),
        dist.entropy(),
    )


def _aggregate_state_value(
    values: torch.Tensor, group_logits: torch.Tensor, mode: str = "max"
) -> torch.Tensor:
    if mode == "expected":
        group_probs = torch.softmax(group_logits, dim=-1)
        return (group_probs * values).sum()

    return torch.max(values)


def _run_episode(
    agent: TransformHierarchicalAgent,
    env: TransformEnv,
    cfg: TransformTrainConfig,
    greedy: bool = False,
    sample_temperature: float = 1.0,
) -> Tuple[List[TransformStep], float, QuantumCircuit, float]:
    traj: List[TransformStep] = []
    s = env.reset()
    best_circuit = env.current.copy()
    best_cost = float(env.current_cost)
    best_score = _transform_selection_score(best_circuit, best_cost, cfg)

    while True:
        if s.group_features.shape[0] == 0:
            break

        t = _state_to_tensors(s, cfg.device)

        with torch.inference_mode():
            with _amp_context(cfg):
                values, group_logits, group_embed = agent.group_values_and_logits(
                    t["dag_node_features"],
                    t["dag_adj"],
                    t["group_node_indices"],
                    t["group_features"],
                    t["global_features"],
                )
            values = values.float()
            group_logits = group_logits.float()
            state_value = _aggregate_state_value(
                values, group_logits, mode=cfg.value_aggregate_mode
            )

        if greedy:
            g_idx = int(torch.argmax(group_logits).item())
            g_logp_all = torch.log_softmax(group_logits, dim=-1)
            logp_g = g_logp_all[g_idx]
            entropy_g = torch.tensor(0.0, device=cfg.device)
        else:
            g_idx, logp_g, entropy_g = _sample_from_logits(
                group_logits, temperature=sample_temperature
            )

        af = t["action_features"][g_idx]
        if af.shape[0] == 0:
            break
        with torch.inference_mode():
            with _amp_context(cfg):
                a_logits = agent.action_logits(
                    group_embed[g_idx], af, t["global_features"]
                )
            a_logits = a_logits.float()
        if greedy:
            a_idx = int(torch.argmax(a_logits).item())
            a_logp_all = torch.log_softmax(a_logits, dim=-1)
            logp_a = a_logp_all[a_idx]
            entropy_a = torch.tensor(0.0, device=cfg.device)
        else:
            a_idx, logp_a, entropy_a = _sample_from_logits(
                a_logits, temperature=sample_temperature
            )

        s_next, reward, done, _ = env.step(g_idx, a_idx, state=s)

        traj.append(
            TransformStep(
                state=s,
                group_idx=g_idx,
                action_idx=a_idx,
                logp_group=float(logp_g.item()),
                logp_action=float(logp_a.item()),
                reward=float(reward),
                value=float(state_value.item()),
                done=done,
            )
        )

        s = s_next
        cur_cost = float(env.current_cost)
        cur_score = _transform_selection_score(env.current, cur_cost, cfg)
        if cur_score < best_score:
            best_score = float(cur_score)
            best_cost = float(cur_cost)
            best_circuit = env.current.copy()
        if done:
            break

    total_reward = float(sum(t.reward for t in traj))
    return traj, total_reward, best_circuit, best_cost


def _build_transform_minibatch(
    transitions: List[TransformTransition],
    mb: np.ndarray,
    device: str,
) -> Dict[str, torch.Tensor]:
    batch_items = [transitions[int(i)] for i in mb]
    bsz = len(batch_items)
    if bsz <= 0:
        raise ValueError("Empty minibatch for transform PPO.")

    max_nodes = max(
        1, max(int(it.tensors["dag_node_features"].shape[0]) for it in batch_items)
    )
    max_groups = max(
        1, max(int(it.tensors["group_features"].shape[0]) for it in batch_items)
    )
    max_group_nodes = 1
    max_actions = 1

    for it in batch_items:
        t = it.tensors
        g = int(t["group_features"].shape[0])
        if g <= 0:
            continue
        for gi in range(g):
            k = int(t["group_node_indices"][gi].numel())
            if k > max_group_nodes:
                max_group_nodes = k
        sg = int(it.group_idx)
        if sg < 0 or sg >= g:
            raise RuntimeError(f"Invalid selected group index {sg} for group size {g}.")
        a = int(t["action_features"][sg].shape[0])
        if a <= 0:
            raise RuntimeError("Selected transform group has no available actions.")
        if int(it.action_idx) < 0 or int(it.action_idx) >= a:
            raise RuntimeError(
                f"Invalid selected action index {int(it.action_idx)} for action size {a}."
            )
        if a > max_actions:
            max_actions = a

    node_feat_dim = int(batch_items[0].tensors["dag_node_features"].shape[1])
    group_feat_dim = (
        int(batch_items[0].tensors["group_features"].shape[1])
        if batch_items[0].tensors["group_features"].ndim == 2
        else 14
    )
    global_dim = int(batch_items[0].tensors["global_features"].shape[0])
    action_feat_dim = 6
    for it in batch_items:
        sg = int(it.group_idx)
        af = it.tensors["action_features"][sg]
        if af.ndim == 2 and af.shape[1] > 0:
            action_feat_dim = int(af.shape[1])
            break

    node_features = torch.zeros(
        (bsz, max_nodes, node_feat_dim), dtype=torch.float32, device=device
    )
    dag_adj = torch.zeros(
        (bsz, max_nodes, max_nodes), dtype=torch.float32, device=device
    )
    group_features = torch.zeros(
        (bsz, max_groups, group_feat_dim), dtype=torch.float32, device=device
    )
    group_valid_mask = torch.zeros((bsz, max_groups), dtype=torch.bool, device=device)
    group_node_indices = torch.zeros(
        (bsz, max_groups, max_group_nodes), dtype=torch.long, device=device
    )
    group_node_mask = torch.zeros(
        (bsz, max_groups, max_group_nodes), dtype=torch.bool, device=device
    )
    global_features = torch.zeros((bsz, global_dim), dtype=torch.float32, device=device)
    selected_group_idx = torch.zeros((bsz,), dtype=torch.long, device=device)
    action_features = torch.zeros(
        (bsz, max_actions, action_feat_dim), dtype=torch.float32, device=device
    )
    action_valid_mask = torch.zeros((bsz, max_actions), dtype=torch.bool, device=device)
    selected_action_idx = torch.zeros((bsz,), dtype=torch.long, device=device)

    for b, it in enumerate(batch_items):
        t = it.tensors
        n = int(t["dag_node_features"].shape[0])
        if n > 0:
            node_features[b, :n] = t["dag_node_features"]
            dag_adj[b, :n, :n] = t["dag_adj"]

        g = int(t["group_features"].shape[0])
        if g > 0:
            group_features[b, :g] = t["group_features"]
            group_valid_mask[b, :g] = True
            for gi in range(g):
                idx = t["group_node_indices"][gi]
                k = int(idx.numel())
                if k > 0:
                    kk = min(k, max_group_nodes)
                    group_node_indices[b, gi, :kk] = idx[:kk]
                    group_node_mask[b, gi, :kk] = True

        global_features[b] = t["global_features"]
        sg = int(it.group_idx)
        selected_group_idx[b] = sg

        af = t["action_features"][sg]
        a = int(af.shape[0])
        action_features[b, :a] = af
        action_valid_mask[b, :a] = True
        selected_action_idx[b] = int(it.action_idx)

    return {
        "node_features": node_features,
        "dag_adj": dag_adj,
        "group_features": group_features,
        "group_valid_mask": group_valid_mask,
        "group_node_indices": group_node_indices,
        "group_node_mask": group_node_mask,
        "global_features": global_features,
        "selected_group_idx": selected_group_idx,
        "action_features": action_features,
        "action_valid_mask": action_valid_mask,
        "selected_action_idx": selected_action_idx,
    }


def train_transform_agent(
    circuit: QuantumCircuit,
    mapping_oracle: Callable[[QuantumCircuit], float],
    cfg: TransformTrainConfig,
) -> Tuple[TransformHierarchicalAgent, Dict[str, List[float]], QuantumCircuit, float]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if str(cfg.device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    equiv_rules = None
    if cfg.enable_equiv_rewrite and cfg.equiv_lib_path.strip():
        p = Path(cfg.equiv_lib_path)
        if p.exists():
            try:
                equiv_rules = load_equivalence_rules(str(p.resolve()))
            except Exception:
                equiv_rules = None

    env = TransformEnv(
        circuit=circuit,
        mapping_oracle=mapping_oracle,
        max_steps=cfg.max_steps,
        equivalence_rules=equiv_rules,
        max_equiv_matches=cfg.max_equiv_matches,
        max_dag_nodes=cfg.max_dag_nodes,
        dag_context=cfg.dag_context,
        noop_penalty_start=cfg.noop_penalty_start,
        noop_penalty_end=cfg.noop_penalty_end,
        novelty_bonus_weight=cfg.novelty_bonus_weight,
        equal_cost_change_bonus=cfg.equal_cost_change_bonus,
        oracle_rel_reward_weight=cfg.oracle_rel_reward_weight,
        oracle_abs_reward_weight=cfg.oracle_abs_reward_weight,
        cx_reduction_reward_weight=cfg.cx_reduction_reward_weight,
        gate_reduction_reward_weight=cfg.gate_reduction_reward_weight,
        new_best_bonus_weight=cfg.new_best_bonus_weight,
        worse_cost_penalty_weight=cfg.worse_cost_penalty_weight,
    )
    init_state = env.state()
    node_feat_dim = (
        int(init_state.dag_node_features.shape[1])
        if init_state.dag_node_features.ndim == 2
        else 12
    )
    group_feat_dim = (
        int(init_state.group_features.shape[1])
        if init_state.group_features.ndim == 2
        else 14
    )
    action_feat_dim = 6
    if (
        init_state.action_features
        and init_state.action_features[0].ndim == 2
        and init_state.action_features[0].shape[1] > 0
    ):
        action_feat_dim = int(init_state.action_features[0].shape[1])
    agent = TransformHierarchicalAgent(
        node_feat_dim=node_feat_dim,
        group_feat_dim=group_feat_dim,
        action_feat_dim=action_feat_dim,
        global_dim=4,
        hidden_dim=max(32, int(cfg.hidden_dim)),
    ).to(cfg.device)
    if (
        str(cfg.device).startswith("cuda")
        and bool(cfg.compile_model)
        and hasattr(torch, "compile")
    ):
        try:
            agent = torch.compile(agent, mode="reduce-overhead")
        except Exception:
            pass
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)

    hist = {"reward": [], "best_cost": []}
    best_global_circuit = circuit.copy()
    best_global_cost = env.base_cost
    best_global_score = _transform_selection_score(
        best_global_circuit, best_global_cost, cfg
    )
    plateau_enable = bool(cfg.early_stop_enable)
    plateau_warmup = int(
        max(
            0,
            round(
                float(cfg.early_stop_warmup_ratio) * float(max(1, int(cfg.episodes)))
            ),
        )
    )
    plateau_patience = max(1, int(cfg.early_stop_patience))
    plateau_rel_delta = max(0.0, float(cfg.early_stop_rel_delta))
    plateau_abs_delta = max(0.0, float(cfg.early_stop_abs_delta))
    plateau_eval_interval = max(1, int(cfg.early_stop_eval_interval))
    plateau_best_cost = float("inf")
    plateau_best_state = None
    plateau_no_improve = 0
    plateau_window_best = float("inf")
    stop_training = False

    ep_cursor = 0
    while ep_cursor < cfg.episodes:
        if cfg.episodes > 1:
            p = float(ep_cursor) / float(cfg.episodes - 1)
        else:
            p = 1.0
        lr_now = cfg.lr * (cfg.lr_final_ratio + (1.0 - cfg.lr_final_ratio) * (1.0 - p))
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        entropy_coef = cfg.entropy_coef * (
            cfg.entropy_final_ratio + (1.0 - cfg.entropy_final_ratio) * (1.0 - p)
        )

        transitions: List[TransformTransition] = []
        n_rollouts = min(
            max(1, int(cfg.rollout_batch_episodes)), cfg.episodes - ep_cursor
        )
        for _ in range(n_rollouts):
            sample_temp = float(cfg.rollout_sample_temp_end) + (
                float(cfg.rollout_sample_temp_start)
                - float(cfg.rollout_sample_temp_end)
            ) * (1.0 - p)
            traj, ep_reward, ep_best_circuit, ep_best_cost = _run_episode(
                agent,
                env,
                cfg,
                greedy=False,
                sample_temperature=sample_temp,
            )
            if not traj:
                hist["reward"].append(0.0)
                hist["best_cost"].append(float(ep_best_cost))
                ep_cursor += 1
                if plateau_enable:
                    cur_cost = float(ep_best_cost)
                    if cur_cost < plateau_window_best:
                        plateau_window_best = cur_cost
                    if ep_cursor >= plateau_warmup and (
                        ep_cursor % plateau_eval_interval == 0
                        or ep_cursor >= int(cfg.episodes)
                    ):
                        cand = (
                            plateau_window_best
                            if np.isfinite(plateau_window_best)
                            else cur_cost
                        )
                        if not np.isfinite(plateau_best_cost):
                            plateau_best_cost = float(cand)
                            plateau_best_state = {
                                k: v.detach().cpu().clone()
                                for k, v in agent.state_dict().items()
                            }
                            plateau_no_improve = 0
                        else:
                            delta = max(
                                float(plateau_abs_delta),
                                float(
                                    abs(float(plateau_best_cost))
                                    * float(plateau_rel_delta)
                                ),
                            )
                            if cand <= float(plateau_best_cost) - float(delta):
                                plateau_best_cost = float(cand)
                                plateau_best_state = {
                                    k: v.detach().cpu().clone()
                                    for k, v in agent.state_dict().items()
                                }
                                plateau_no_improve = 0
                            else:
                                plateau_no_improve += 1
                        plateau_window_best = float("inf")
                        if plateau_no_improve >= plateau_patience:
                            stop_training = True
                if stop_training:
                    break
                continue

            rewards = [t.reward for t in traj]
            values = [t.value for t in traj]
            dones = [t.done for t in traj]
            adv, returns = gae_advantages(
                rewards, values, dones, gamma=cfg.gamma, lam=cfg.lam
            )

            for i, t in enumerate(traj):
                transitions.append(
                    TransformTransition(
                        tensors=_state_to_tensors(t.state, cfg.device),
                        group_idx=t.group_idx,
                        action_idx=t.action_idx,
                        old_logp_group=t.logp_group,
                        old_logp_action=t.logp_action,
                        old_value=t.value,
                        advantage=float(adv[i].item()),
                        ret=float(returns[i].item()),
                    )
                )

            ep_best_score = _transform_selection_score(
                ep_best_circuit, ep_best_cost, cfg
            )
            if ep_best_score < best_global_score:
                best_global_score = float(ep_best_score)
                best_global_cost = ep_best_cost
                best_global_circuit = ep_best_circuit.copy()

            hist["reward"].append(float(ep_reward))
            hist["best_cost"].append(float(ep_best_cost))
            ep_cursor += 1
            if plateau_enable:
                cur_cost = float(ep_best_cost)
                if cur_cost < plateau_window_best:
                    plateau_window_best = cur_cost
                if ep_cursor >= plateau_warmup and (
                    ep_cursor % plateau_eval_interval == 0
                    or ep_cursor >= int(cfg.episodes)
                ):
                    cand = (
                        plateau_window_best
                        if np.isfinite(plateau_window_best)
                        else cur_cost
                    )
                    if not np.isfinite(plateau_best_cost):
                        plateau_best_cost = float(cand)
                        plateau_best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in agent.state_dict().items()
                        }
                        plateau_no_improve = 0
                    else:
                        delta = max(
                            float(plateau_abs_delta),
                            float(
                                abs(float(plateau_best_cost)) * float(plateau_rel_delta)
                            ),
                        )
                        if cand <= float(plateau_best_cost) - float(delta):
                            plateau_best_cost = float(cand)
                            plateau_best_state = {
                                k: v.detach().cpu().clone()
                                for k, v in agent.state_dict().items()
                            }
                            plateau_no_improve = 0
                        else:
                            plateau_no_improve += 1
                    plateau_window_best = float("inf")
                    if plateau_no_improve >= plateau_patience:
                        stop_training = True
            if stop_training:
                break

        if not transitions:
            if stop_training:
                break
            continue

        old_logp_g_all = torch.tensor(
            [x.old_logp_group for x in transitions],
            dtype=torch.float32,
            device=cfg.device,
        )
        old_logp_a_all = torch.tensor(
            [x.old_logp_action for x in transitions],
            dtype=torch.float32,
            device=cfg.device,
        )
        old_value_all = torch.tensor(
            [x.old_value for x in transitions], dtype=torch.float32, device=cfg.device
        )
        adv_all = torch.tensor(
            [x.advantage for x in transitions], dtype=torch.float32, device=cfg.device
        )
        ret_all = torch.tensor(
            [x.ret for x in transitions], dtype=torch.float32, device=cfg.device
        )
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std(unbiased=False) + 1e-8)

        idx_all = np.arange(len(transitions), dtype=np.int64)
        mb_size = max(1, min(int(cfg.minibatch_size), len(transitions)))

        early_stop = False
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, len(idx_all), mb_size):
                mb = idx_all[start : start + mb_size]
                mb_t = torch.tensor(mb, dtype=torch.long, device=cfg.device)
                mb_pack = _build_transform_minibatch(transitions, mb, cfg.device)
                with _amp_context(cfg):
                    new_logp_g_t, new_logp_a_t, val_t, entropy_all_t = (
                        agent.selected_logps_and_values_batched(
                            node_features=mb_pack["node_features"],
                            dag_adj=mb_pack["dag_adj"],
                            group_features=mb_pack["group_features"],
                            group_valid_mask=mb_pack["group_valid_mask"],
                            group_node_indices=mb_pack["group_node_indices"],
                            group_node_mask=mb_pack["group_node_mask"],
                            global_features=mb_pack["global_features"],
                            selected_group_idx=mb_pack["selected_group_idx"],
                            action_features=mb_pack["action_features"],
                            action_valid_mask=mb_pack["action_valid_mask"],
                            selected_action_idx=mb_pack["selected_action_idx"],
                            value_aggregate_mode=cfg.value_aggregate_mode,
                        )
                    )
                new_logp_g_t = new_logp_g_t.float()
                new_logp_a_t = new_logp_a_t.float()
                val_t = val_t.float()
                entropy_t = entropy_all_t.float().mean()

                ppo_g = ppo_clip_objective(
                    new_logp_g_t,
                    old_logp_g_all[mb_t],
                    adv_all[mb_t],
                    clip_eps=cfg.clip_eps,
                )
                ppo_a = ppo_clip_objective(
                    new_logp_a_t,
                    old_logp_a_all[mb_t],
                    adv_all[mb_t],
                    clip_eps=cfg.clip_eps,
                )
                actor_loss = -(ppo_g + ppo_a) * 0.5
                value_old_t = old_value_all[mb_t]
                value_clipped = value_old_t + (val_t - value_old_t).clamp(
                    -cfg.value_clip_eps, cfg.value_clip_eps
                )
                critic_unclipped = (val_t - ret_all[mb_t]).pow(2)
                critic_clipped = (value_clipped - ret_all[mb_t]).pow(2)
                critic_loss = 0.5 * torch.max(critic_unclipped, critic_clipped).mean()
                loss = (
                    actor_loss + cfg.value_coef * critic_loss - entropy_coef * entropy_t
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

                if cfg.target_kl > 0:
                    approx_kl = float(
                        0.5
                        * (
                            (old_logp_g_all[mb_t] - new_logp_g_t).mean().item()
                            + (old_logp_a_all[mb_t] - new_logp_a_t).mean().item()
                        )
                    )
                    if approx_kl > 1.5 * float(cfg.target_kl):
                        early_stop = True
                        break
            if early_stop:
                break
        if stop_training:
            break

    for _ in range(max(0, int(cfg.greedy_eval_episodes))):
        _, _, eval_best_circuit, eval_best_cost = _run_episode(
            agent, env, cfg, greedy=True
        )
        eval_best_score = _transform_selection_score(
            eval_best_circuit, eval_best_cost, cfg
        )
        if eval_best_score < best_global_score:
            best_global_score = float(eval_best_score)
            best_global_cost = eval_best_cost
            best_global_circuit = eval_best_circuit.copy()

    if (
        plateau_enable
        and plateau_best_state is not None
        and float(plateau_best_cost) <= float(best_global_cost)
    ):
        agent.load_state_dict(
            {k: v.to(cfg.device) for k, v in plateau_best_state.items()}
        )
        _, _, eval_best_circuit, eval_best_cost = _run_episode(
            agent, env, cfg, greedy=True
        )
        eval_best_score = _transform_selection_score(
            eval_best_circuit, eval_best_cost, cfg
        )
        if eval_best_score < best_global_score:
            best_global_score = float(eval_best_score)
            best_global_cost = float(eval_best_cost)
            best_global_circuit = eval_best_circuit.copy()

    return agent, hist, best_global_circuit, float(best_global_cost)


def collect_transform_candidates(
    agent: TransformHierarchicalAgent,
    circuit: QuantumCircuit,
    mapping_oracle: Callable[[QuantumCircuit], float],
    cfg: TransformTrainConfig,
    greedy_episodes: int = 2,
    sample_episodes: int = 4,
    max_candidates: int = 8,
) -> List[QuantumCircuit]:
    equiv_rules = None
    if cfg.enable_equiv_rewrite and cfg.equiv_lib_path.strip():
        p = Path(cfg.equiv_lib_path)
        if p.exists():
            try:
                equiv_rules = load_equivalence_rules(str(p.resolve()))
            except Exception:
                equiv_rules = None

    pool: Dict[Tuple, Tuple[float, QuantumCircuit]] = {}

    def _sig(c: QuantumCircuit) -> Tuple:
        return tuple(
            (g.op, g.q0, g.q1, None if g.param is None else round(float(g.param), 6))
            for g in c.gates
        )

    def _push(c: QuantumCircuit, cost: float) -> None:
        sig = _sig(c)
        old = pool.get(sig)
        if old is None or float(cost) < float(old[0]):
            pool[sig] = (float(cost), c.copy())

    def _build_env(seed_circuit: QuantumCircuit) -> TransformEnv:
        return TransformEnv(
            circuit=seed_circuit,
            mapping_oracle=mapping_oracle,
            max_steps=cfg.max_steps,
            equivalence_rules=equiv_rules,
            max_equiv_matches=cfg.max_equiv_matches,
            max_dag_nodes=cfg.max_dag_nodes,
            dag_context=cfg.dag_context,
            noop_penalty_start=cfg.noop_penalty_start,
            noop_penalty_end=cfg.noop_penalty_end,
            novelty_bonus_weight=cfg.novelty_bonus_weight,
            equal_cost_change_bonus=cfg.equal_cost_change_bonus,
            oracle_rel_reward_weight=cfg.oracle_rel_reward_weight,
            oracle_abs_reward_weight=cfg.oracle_abs_reward_weight,
            cx_reduction_reward_weight=cfg.cx_reduction_reward_weight,
            gate_reduction_reward_weight=cfg.gate_reduction_reward_weight,
            new_best_bonus_weight=cfg.new_best_bonus_weight,
            worse_cost_penalty_weight=cfg.worse_cost_penalty_weight,
        )

    temps = tuple(float(x) for x in cfg.candidate_sample_temperatures if float(x) > 0)
    if not temps:
        temps = (1.0,)
    large_circuit = bool(circuit.gate_count() >= 800 or circuit.cx_count() >= 300)
    if large_circuit and int(cfg.episodes) >= 120:
        greedy_episodes = max(int(greedy_episodes), 4)
        sample_episodes = max(int(sample_episodes), 8)
        max_candidates = max(int(max_candidates), 16)

    def _rollout_from_seed(
        seed_c: QuantumCircuit,
        greedy_n: int,
        sample_n: int,
        temp_seq: Sequence[float],
    ) -> None:
        env = _build_env(seed_c)
        for _ in range(max(0, int(greedy_n))):
            _, _, best_c, best_cost = _run_episode(agent, env, cfg, greedy=True)
            _push(best_c, float(best_cost))

            _push(env.current, float(env.current_cost))
        for sidx in range(max(0, int(sample_n))):
            temp = float(temp_seq[sidx % len(temp_seq)])
            _, _, best_c, best_cost = _run_episode(
                agent,
                env,
                cfg,
                greedy=False,
                sample_temperature=temp,
            )
            _push(best_c, float(best_cost))
            _push(env.current, float(env.current_cost))

    _push(circuit.copy(), float(mapping_oracle(circuit)))
    _rollout_from_seed(
        circuit,
        greedy_n=greedy_episodes,
        sample_n=sample_episodes,
        temp_seq=temps,
    )

    for ridx in range(max(0, int(cfg.candidate_refine_rounds))):
        ranked_now = sorted(pool.values(), key=lambda x: float(x[0]))
        topk = max(1, int(cfg.candidate_refine_topk))
        seeds = [c.copy() for _, c in ranked_now[:topk]]
        for sidx, seed_c in enumerate(seeds):

            g_n = max(1, int(greedy_episodes // 2))
            s_n = max(1, int(sample_episodes // 2))

            rot = (ridx + sidx) % len(temps)
            rotated_temps = tuple(list(temps)[rot:] + list(temps)[:rot])
            _rollout_from_seed(
                seed_c,
                greedy_n=g_n,
                sample_n=s_n,
                temp_seq=rotated_temps,
            )

    orig_sig = _sig(circuit)
    nonorig_now = sum(1 for sig in pool.keys() if sig != orig_sig)
    if nonorig_now == 0 or large_circuit:
        try:
            env0 = _build_env(circuit)
            s0 = env0.reset()
            one_hop: List[Tuple[float, QuantumCircuit]] = []
            seen_hop = set()
            for opp in s0.opportunities:
                for aidx in range(len(opp.replacements)):
                    cand = circuit.apply(opp, aidx)
                    sig = _sig(cand)
                    if sig == orig_sig or sig in seen_hop:
                        continue
                    seen_hop.add(sig)
                    cst = float(mapping_oracle(cand))
                    one_hop.append((cst, cand))

            one_hop.sort(
                key=lambda x: (
                    float(x[0]),
                    int(x[1].cx_count()),
                    int(x[1].gate_count()),
                )
            )
            keep = max(6, int(max_candidates) * (3 if large_circuit else 2))
            for cst, cand in one_hop[:keep]:
                _push(cand, float(cst))
        except Exception:

            pass

    ranked = sorted(pool.values(), key=lambda x: float(x[0]))
    out_all = [c for _, c in ranked]
    limit = max(1, int(max_candidates))
    if len(out_all) <= limit:
        out = list(out_all)
    else:

        out: List[QuantumCircuit] = []
        seen_bins = set()
        for c in out_all:
            b = (int(c.cx_count()) // 8, int(c.gate_count()) // 16)
            if b in seen_bins and len(out) >= max(4, int(0.5 * limit)):
                continue
            seen_bins.add(b)
            out.append(c)
            if len(out) >= limit:
                break
        if len(out) < limit:
            seen_sig = {
                tuple(
                    (
                        g.op,
                        g.q0,
                        g.q1,
                        None if g.param is None else round(float(g.param), 6),
                    )
                    for g in c.gates
                )
                for c in out
            }
            for c in out_all:
                sig = tuple(
                    (
                        g.op,
                        g.q0,
                        g.q1,
                        None if g.param is None else round(float(g.param), 6),
                    )
                    for g in c.gates
                )
                if sig in seen_sig:
                    continue
                out.append(c)
                seen_sig.add(sig)
                if len(out) >= limit:
                    break

    return out

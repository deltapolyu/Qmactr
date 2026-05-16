from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch

from .cost import total_entanglement_cost
from .env_mapping import MappingEnv, MappingState
from .models import MappingActorCritic
from .ppo import (
    gae_advantages,
    masked_log_softmax,
    ppo_clip_objective,
)
from .topology import DQCTopology


@dataclass
class MappingTrainConfig:
    episodes: int = 400
    ppo_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_final_ratio: float = 0.25
    lr_final_ratio: float = 0.2
    rollout_batch_episodes: int = 2
    minibatch_size: int = 64
    hidden_dim: int = 96
    heads: int = 4
    select_best_checkpoint: bool = False
    select_interval_episodes: int = 16
    select_num_samples: int = 1
    target_kl: float = 0.03
    rollout_temp_start: float = 1.25
    rollout_temp_end: float = 0.95
    rollout_temp_jitter: float = 0.08
    normalize_value_targets: bool = True
    hier_qubit_coef: float = 0.5
    hier_cond_coef: float = 0.5
    hier_qubit_entropy_ratio: float = 0.6
    sil_enable: bool = True
    sil_buffer_size: int = 50000
    sil_batch_size: int = 256
    sil_updates_per_iter: int = 1
    sil_coef: float = 0.05
    sil_value_coef: float = 0.5
    sil_min_adv: float = 0.0
    sil_adv_clip: float = 5.0

    early_stop_enable: bool = True
    early_stop_warmup_ratio: float = 0.2
    early_stop_patience: int = 48
    early_stop_rel_delta: float = 1e-3
    early_stop_abs_delta: float = 1.0
    early_stop_eval_interval: int = 4
    qubit_adj_mode: str = "raw"
    edge_aware: bool = True
    seed: int = 0
    device: str = "cpu"


@dataclass
class MappingStep:
    state: MappingState
    action: int
    logp: float
    logp_qubit: float
    logp_cond: float
    qubit_idx: int
    qpu_idx: int
    reward: float
    done: bool
    value: float


@dataclass
class MappingTransition:
    tensors: Dict[str, torch.Tensor]
    action: int
    old_logp: float
    old_logp_qubit: float
    old_logp_cond: float
    qubit_idx: int
    qpu_idx: int
    old_value: float
    advantage: float
    ret: float


@dataclass
class MappingSILSample:
    tensors_cpu: Dict[str, torch.Tensor]
    qubit_idx: int
    qpu_idx: int
    ret: float
    adv_pos: float


def _state_to_tensors(state: MappingState, device: str) -> Dict[str, torch.Tensor]:
    return {
        "qpu_features": torch.as_tensor(
            state.qpu_features, dtype=torch.float32, device=device
        ),
        "qpu_adj": torch.as_tensor(state.qpu_adj, dtype=torch.float32, device=device),
        "qubit_features": torch.as_tensor(
            state.qubit_features, dtype=torch.float32, device=device
        ),
        "qubit_adj": torch.as_tensor(
            state.qubit_adj, dtype=torch.float32, device=device
        ),
        "global_features": torch.as_tensor(
            state.global_features, dtype=torch.float32, device=device
        ),
        "action_mask": torch.as_tensor(
            state.action_mask, dtype=torch.float32, device=device
        ),
        "pair_prior": torch.as_tensor(
            state.pair_prior, dtype=torch.float32, device=device
        ),
    }


def _states_to_batched_tensors(
    states: List[MappingState], device: str
) -> Dict[str, torch.Tensor]:
    return {
        "qpu_features": torch.as_tensor(
            np.stack([s.qpu_features for s in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "qpu_adj": torch.as_tensor(
            np.stack([s.qpu_adj for s in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "qubit_features": torch.as_tensor(
            np.stack([s.qubit_features for s in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "qubit_adj": torch.as_tensor(
            np.stack([s.qubit_adj for s in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "global_features": torch.as_tensor(
            np.stack([s.global_features for s in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "action_mask": torch.as_tensor(
            np.stack([s.action_mask for s in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "pair_prior": torch.as_tensor(
            np.stack([s.pair_prior for s in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
    }


def _amp_context(device: str, use_amp: bool):
    if bool(use_amp) and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _select_hierarchical_action(
    qubit_logits: torch.Tensor,
    cond_logits: torch.Tensor,
    action_mask: torch.Tensor,
    greedy: bool,
    temperature: float,
    generator: Optional[torch.Generator],
) -> Tuple[int, int, int, float, float, float]:
    num_qpus, num_qubits = int(cond_logits.shape[0]), int(cond_logits.shape[1])
    mask = action_mask.view(num_qpus, num_qubits) > 0
    valid_qubits = mask.any(dim=0)
    if not bool(valid_qubits.any().item()):
        raise RuntimeError("No valid hierarchical action available.")

    temp = max(1e-4, float(temperature))
    qubit_mask_f = valid_qubits.to(dtype=qubit_logits.dtype)

    if greedy:
        cond_masked = cond_logits.masked_fill(~mask, -1e9)
        qubit_scores = qubit_logits + torch.logsumexp(cond_masked, dim=0)
        qubit_scores = qubit_scores.masked_fill(~valid_qubits, -1e9)
        qubit_idx = int(torch.argmax(qubit_scores).item())
    else:
        q_logits_t = (
            torch.nan_to_num(qubit_logits, nan=0.0, posinf=1e4, neginf=-1e4) / temp
        )
        q_logp_all_t = masked_log_softmax(q_logits_t, qubit_mask_f)
        q_probs_t = torch.exp(q_logp_all_t) * qubit_mask_f
        q_probs_t = q_probs_t / q_probs_t.sum().clamp_min(1e-12)
        qubit_idx = int(
            torch.multinomial(q_probs_t, num_samples=1, generator=generator).item()
        )

    valid_qpu = mask[:, qubit_idx]
    qpu_mask_f = valid_qpu.to(dtype=cond_logits.dtype)
    if not bool(valid_qpu.any().item()):
        raise RuntimeError("Chosen qubit has no valid QPU action.")

    if greedy:
        q_logits_cond = cond_logits[:, qubit_idx].masked_fill(~valid_qpu, -1e9)
        qpu_idx = int(torch.argmax(q_logits_cond).item())
    else:
        cond_t = (
            torch.nan_to_num(
                cond_logits[:, qubit_idx], nan=0.0, posinf=1e4, neginf=-1e4
            )
            / temp
        )
        cond_logp_all_t = masked_log_softmax(cond_t, qpu_mask_f)
        cond_probs_t = torch.exp(cond_logp_all_t) * qpu_mask_f
        cond_probs_t = cond_probs_t / cond_probs_t.sum().clamp_min(1e-12)
        qpu_idx = int(
            torch.multinomial(cond_probs_t, num_samples=1, generator=generator).item()
        )

    q_logits_t = torch.nan_to_num(qubit_logits, nan=0.0, posinf=1e4, neginf=-1e4) / temp
    q_logp_all_t = masked_log_softmax(q_logits_t, qubit_mask_f)
    logp_qubit = float(q_logp_all_t[qubit_idx].item())

    cond_t = (
        torch.nan_to_num(cond_logits[:, qubit_idx], nan=0.0, posinf=1e4, neginf=-1e4)
        / temp
    )
    cond_logp_all_t = masked_log_softmax(cond_t, qpu_mask_f)
    logp_cond = float(cond_logp_all_t[qpu_idx].item())

    action = int(qpu_idx * num_qubits + qubit_idx)
    return (
        action,
        qpu_idx,
        qubit_idx,
        logp_qubit,
        logp_cond,
        float(logp_qubit + logp_cond),
    )


def _hierarchical_logps_and_entropy_batched(
    qubit_logits: torch.Tensor,
    cond_logits: torch.Tensor,
    action_mask_flat: torch.Tensor,
    qubit_idx: torch.Tensor,
    qpu_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, num_qpus, num_qubits = cond_logits.shape
    mask = action_mask_flat.view(bsz, num_qpus, num_qubits) > 0

    valid_qubits = mask.any(dim=1)
    qubit_logits_masked = qubit_logits.masked_fill(~valid_qubits, -1e9)
    q_logp_all = torch.log_softmax(qubit_logits_masked, dim=-1)
    q_probs = torch.exp(q_logp_all) * valid_qubits.to(dtype=qubit_logits.dtype)
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    new_logp_qubit = q_logp_all.gather(1, qubit_idx.unsqueeze(1)).squeeze(1)

    gather_idx = qubit_idx.view(bsz, 1, 1).expand(bsz, num_qpus, 1)
    cond_selected = cond_logits.gather(2, gather_idx).squeeze(2)
    valid_qpu = mask.gather(2, gather_idx).squeeze(2)

    cond_masked = cond_selected.masked_fill(~valid_qpu, -1e9)
    cond_logp_all = torch.log_softmax(cond_masked, dim=-1)
    cond_probs = torch.exp(cond_logp_all) * valid_qpu.to(dtype=cond_logits.dtype)
    cond_probs = cond_probs / cond_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    new_logp_cond = cond_logp_all.gather(1, qpu_idx.unsqueeze(1)).squeeze(1)
    new_logp_total = new_logp_qubit + new_logp_cond

    entropy_qubit = -torch.nan_to_num(
        q_probs * q_logp_all, nan=0.0, posinf=0.0, neginf=0.0
    ).sum(dim=-1)
    entropy_cond = -torch.nan_to_num(
        cond_probs * cond_logp_all, nan=0.0, posinf=0.0, neginf=0.0
    ).sum(dim=-1)
    return new_logp_qubit, new_logp_cond, new_logp_total, entropy_qubit, entropy_cond


def _run_episode(
    model: MappingActorCritic,
    env: MappingEnv,
    device: str,
    greedy: bool = False,
    sample_temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    use_amp: bool = False,
) -> Tuple[List[MappingStep], Dict[int, int], float]:
    traj: List[MappingStep] = []
    s = env.reset()

    while True:
        t = _state_to_tensors(s, device)

        with torch.inference_mode():
            with _amp_context(device, use_amp=use_amp):
                _, qubit_logits, cond_logits, value = model.forward_hier(
                    t["qpu_features"],
                    t["qpu_adj"],
                    t["qubit_features"],
                    t["qubit_adj"],
                    t["global_features"],
                    t["pair_prior"],
                )
            qubit_logits = qubit_logits.float()
            cond_logits = cond_logits.float()
            value = value.float()

        action, qpu_idx, qubit_idx, logp_q, logp_cond, logp = (
            _select_hierarchical_action(
                qubit_logits=qubit_logits,
                cond_logits=cond_logits,
                action_mask=t["action_mask"],
                greedy=bool(greedy),
                temperature=float(sample_temperature),
                generator=generator,
            )
        )

        s_next, r, done, _ = env.step(action)
        traj.append(
            MappingStep(
                state=s,
                action=action,
                logp=logp,
                logp_qubit=logp_q,
                logp_cond=logp_cond,
                qubit_idx=int(qubit_idx),
                qpu_idx=int(qpu_idx),
                reward=float(r),
                done=done,
                value=float(value.item()),
            )
        )
        s = s_next
        if done:
            break

    final_cost = total_entanglement_cost(env.circuit, env.mapping, env.topology)
    return traj, dict(env.mapping), float(final_cost)


def _run_training_episodes_batch(
    model: MappingActorCritic,
    circuit,
    topology: DQCTopology,
    device: str,
    sample_temperatures: Sequence[float],
    generators: Sequence[Optional[torch.Generator]],
    qubit_adj_mode: str = "raw",
    use_amp: bool = True,
) -> List[Tuple[List[MappingStep], Dict[int, int], float]]:
    batch_size = len(sample_temperatures)
    if batch_size <= 0:
        return []

    envs = [
        MappingEnv(circuit, topology, qubit_adj_mode=qubit_adj_mode)
        for _ in range(batch_size)
    ]
    states = [e.reset() for e in envs]
    done = [False] * batch_size
    trajs: List[List[MappingStep]] = [[] for _ in range(batch_size)]

    while True:
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break
        active_states = [states[i] for i in active]
        t = _states_to_batched_tensors(active_states, device)
        with torch.inference_mode():
            with _amp_context(device, use_amp=use_amp):
                _, qubit_logits_b, cond_logits_b, value_b = model.forward_hier(
                    t["qpu_features"],
                    t["qpu_adj"],
                    t["qubit_features"],
                    t["qubit_adj"],
                    t["global_features"],
                    t["pair_prior"],
                )
            qubit_logits_b = qubit_logits_b.float()
            cond_logits_b = cond_logits_b.float()
            value_b = value_b.float()

        for local_idx, env_idx in enumerate(active):
            cur_state = states[env_idx]
            qubit_logits = qubit_logits_b[local_idx]
            cond_logits = cond_logits_b[local_idx]
            mask = t["action_mask"][local_idx]
            action, qpu_idx, qubit_idx, logp_q, logp_cond, logp = (
                _select_hierarchical_action(
                    qubit_logits=qubit_logits,
                    cond_logits=cond_logits,
                    action_mask=mask,
                    greedy=False,
                    temperature=float(sample_temperatures[env_idx]),
                    generator=generators[env_idx],
                )
            )
            s_next, r, cur_done, _ = envs[env_idx].step(action)
            trajs[env_idx].append(
                MappingStep(
                    state=cur_state,
                    action=action,
                    logp=logp,
                    logp_qubit=logp_q,
                    logp_cond=logp_cond,
                    qubit_idx=int(qubit_idx),
                    qpu_idx=int(qpu_idx),
                    reward=float(r),
                    done=bool(cur_done),
                    value=float(value_b[local_idx].item()),
                )
            )
            states[env_idx] = s_next
            done[env_idx] = bool(cur_done)

    out: List[Tuple[List[MappingStep], Dict[int, int], float]] = []
    for eidx, env in enumerate(envs):
        cost = float(total_entanglement_cost(env.circuit, env.mapping, env.topology))
        out.append((trajs[eidx], dict(env.mapping), cost))
    return out


def _run_sample_episodes_batch(
    model: MappingActorCritic,
    circuit,
    topology: DQCTopology,
    device: str,
    sample_temperatures: Sequence[float],
    generators: Sequence[Optional[torch.Generator]],
    qubit_adj_mode: str = "raw",
    use_amp: bool = True,
) -> List[Tuple[Dict[int, int], float]]:
    batch_size = len(sample_temperatures)
    if batch_size <= 0:
        return []

    envs = [
        MappingEnv(circuit, topology, qubit_adj_mode=qubit_adj_mode)
        for _ in range(batch_size)
    ]
    states = [e.reset() for e in envs]
    done = [False] * batch_size

    while True:
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break
        active_states = [states[i] for i in active]
        t = _states_to_batched_tensors(active_states, device)
        with torch.inference_mode():
            with _amp_context(device, use_amp=use_amp):
                _, qubit_logits_b, cond_logits_b, _ = model.forward_hier(
                    t["qpu_features"],
                    t["qpu_adj"],
                    t["qubit_features"],
                    t["qubit_adj"],
                    t["global_features"],
                    t["pair_prior"],
                )
            qubit_logits_b = qubit_logits_b.float()
            cond_logits_b = cond_logits_b.float()

        for local_idx, env_idx in enumerate(active):
            qubit_logits = qubit_logits_b[local_idx]
            cond_logits = cond_logits_b[local_idx]
            mask = t["action_mask"][local_idx]
            action, _, _, _, _, _ = _select_hierarchical_action(
                qubit_logits=qubit_logits,
                cond_logits=cond_logits,
                action_mask=mask,
                greedy=False,
                temperature=float(sample_temperatures[env_idx]),
                generator=generators[env_idx],
            )
            s_next, _, cur_done, _ = envs[env_idx].step(action)
            states[env_idx] = s_next
            done[env_idx] = bool(cur_done)

    out: List[Tuple[Dict[int, int], float]] = []
    for env in envs:
        cost = float(total_entanglement_cost(env.circuit, env.mapping, env.topology))
        out.append((dict(env.mapping), cost))
    return out


def _sil_samples_to_device_batch(
    samples: Sequence[MappingSILSample],
    device: str,
) -> Tuple[
    Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    batched = {
        "qpu_features": torch.stack(
            [s.tensors_cpu["qpu_features"] for s in samples], dim=0
        ).to(device=device),
        "qpu_adj": torch.stack([s.tensors_cpu["qpu_adj"] for s in samples], dim=0).to(
            device=device
        ),
        "qubit_features": torch.stack(
            [s.tensors_cpu["qubit_features"] for s in samples], dim=0
        ).to(device=device),
        "qubit_adj": torch.stack(
            [s.tensors_cpu["qubit_adj"] for s in samples], dim=0
        ).to(device=device),
        "global_features": torch.stack(
            [s.tensors_cpu["global_features"] for s in samples], dim=0
        ).to(device=device),
        "action_mask": torch.stack(
            [s.tensors_cpu["action_mask"] for s in samples], dim=0
        ).to(device=device),
        "pair_prior": torch.stack(
            [s.tensors_cpu["pair_prior"] for s in samples], dim=0
        ).to(device=device),
    }
    qubit_idx = torch.tensor(
        [s.qubit_idx for s in samples], dtype=torch.long, device=device
    )
    qpu_idx = torch.tensor(
        [s.qpu_idx for s in samples], dtype=torch.long, device=device
    )
    ret = torch.tensor([s.ret for s in samples], dtype=torch.float32, device=device)
    adv_pos = torch.tensor(
        [s.adv_pos for s in samples], dtype=torch.float32, device=device
    )
    return batched, qubit_idx, qpu_idx, ret, adv_pos


def train_mapping_agent(
    circuit,
    topology: DQCTopology,
    config: MappingTrainConfig,
) -> Tuple[MappingActorCritic, Dict[str, List[float]]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    use_cuda = str(config.device).startswith("cuda")
    if str(config.device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    env = MappingEnv(circuit, topology, qubit_adj_mode=config.qubit_adj_mode)
    init_state = env.state()
    model = MappingActorCritic(
        qpu_feat_dim=int(init_state.qpu_features.shape[1]),
        qubit_feat_dim=int(init_state.qubit_features.shape[1]),
        global_dim=int(init_state.global_features.shape[0]),
        hidden_dim=max(32, int(config.hidden_dim)),
        heads=max(1, int(config.heads)),
        edge_aware=bool(config.edge_aware),
    ).to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    hist = {"reward": [], "cost": []}

    best_state = None
    best_cost = float("inf")
    sil_buffer: deque[MappingSILSample] = deque(
        maxlen=max(1, int(config.sil_buffer_size))
    )
    plateau_best_state = None
    plateau_best_cost = float("inf")
    plateau_no_improve = 0
    plateau_window_best = float("inf")
    plateau_enable = bool(config.early_stop_enable)
    plateau_warmup = int(
        max(
            0,
            round(
                float(config.early_stop_warmup_ratio)
                * float(max(1, int(config.episodes)))
            ),
        )
    )
    plateau_patience = max(1, int(config.early_stop_patience))
    plateau_rel_delta = max(0.0, float(config.early_stop_rel_delta))
    plateau_abs_delta = max(0.0, float(config.early_stop_abs_delta))
    plateau_eval_interval = max(1, int(config.early_stop_eval_interval))
    stop_training = False

    def _snapshot_if_better() -> None:
        nonlocal best_state, best_cost
        _, _, g_cost = _run_episode(model, env, config.device, greedy=True)
        cand_cost = float(g_cost)
        for _ in range(max(0, int(config.select_num_samples))):
            _, _, s_cost = _run_episode(
                model, env, config.device, greedy=False, sample_temperature=1.0
            )
            if s_cost < cand_cost:
                cand_cost = float(s_cost)
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if config.select_best_checkpoint:
        _snapshot_if_better()

    ep_cursor = 0
    while ep_cursor < config.episodes:
        if config.episodes > 1:
            p = float(ep_cursor) / float(config.episodes - 1)
        else:
            p = 1.0
        lr_now = config.lr * (
            config.lr_final_ratio + (1.0 - config.lr_final_ratio) * (1.0 - p)
        )
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        entropy_coef = config.entropy_coef * (
            config.entropy_final_ratio + (1.0 - config.entropy_final_ratio) * (1.0 - p)
        )

        transitions: List[MappingTransition] = []
        n_rollouts = min(
            max(1, int(config.rollout_batch_episodes)), config.episodes - ep_cursor
        )
        rollout_temps: List[float] = []
        rollout_gens: List[Optional[torch.Generator]] = []
        temp_base = (
            float(config.rollout_temp_start)
            + (float(config.rollout_temp_end) - float(config.rollout_temp_start)) * p
        )
        for ridx in range(n_rollouts):
            if float(config.rollout_temp_jitter) > 0:
                temp_scale = float(
                    np.exp(np.random.normal(0.0, float(config.rollout_temp_jitter)))
                )
            else:
                temp_scale = 1.0
            rollout_temps.append(max(0.5, min(2.5, temp_base * temp_scale)))

            g = torch.Generator(device="cuda" if use_cuda else "cpu")
            g.manual_seed(int(config.seed) + 1000003 * int(ep_cursor + ridx + 1))
            rollout_gens.append(g)

        if n_rollouts > 1:
            rollout_out = _run_training_episodes_batch(
                model,
                circuit,
                topology,
                device=config.device,
                sample_temperatures=rollout_temps,
                generators=rollout_gens,
                qubit_adj_mode=config.qubit_adj_mode,
                use_amp=use_cuda,
            )
        else:
            traj, _, cost = _run_episode(
                model,
                env,
                config.device,
                greedy=False,
                sample_temperature=float(rollout_temps[0]),
                generator=rollout_gens[0],
                use_amp=use_cuda,
            )
            rollout_out = [(traj, {}, cost)]

        for traj, _, cost in rollout_out:
            rewards = [s.reward for s in traj]
            values = [s.value for s in traj]
            dones = [s.done for s in traj]
            adv, returns = gae_advantages(
                rewards,
                values,
                dones,
                gamma=config.gamma,
                lam=config.lam,
                normalize=False,
            )

            for i, step in enumerate(traj):
                transitions.append(
                    MappingTransition(
                        tensors=_state_to_tensors(step.state, config.device),
                        action=step.action,
                        old_logp=step.logp,
                        old_logp_qubit=step.logp_qubit,
                        old_logp_cond=step.logp_cond,
                        qubit_idx=int(step.qubit_idx),
                        qpu_idx=int(step.qpu_idx),
                        old_value=step.value,
                        advantage=float(adv[i].item()),
                        ret=float(returns[i].item()),
                    )
                )
                if bool(config.sil_enable):
                    adv_pos = max(0.0, float(returns[i].item()) - float(step.value))
                    if adv_pos >= float(config.sil_min_adv):
                        sil_buffer.append(
                            MappingSILSample(
                                tensors_cpu={
                                    k: v.detach().to("cpu")
                                    for k, v in _state_to_tensors(
                                        step.state, "cpu"
                                    ).items()
                                },
                                qubit_idx=int(step.qubit_idx),
                                qpu_idx=int(step.qpu_idx),
                                ret=float(returns[i].item()),
                                adv_pos=float(adv_pos),
                            )
                        )

            hist["reward"].append(float(sum(rewards)))
            hist["cost"].append(float(cost))
            ep_cursor += 1
            if plateau_enable:
                cur_cost = float(cost)
                if cur_cost < plateau_window_best:
                    plateau_window_best = cur_cost

                if ep_cursor >= plateau_warmup and (
                    ep_cursor % plateau_eval_interval == 0
                    or ep_cursor >= int(config.episodes)
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
                            for k, v in model.state_dict().items()
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
                                for k, v in model.state_dict().items()
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

        old_logp_all = torch.tensor(
            [x.old_logp for x in transitions], dtype=torch.float32, device=config.device
        )
        old_logp_qubit_all = torch.tensor(
            [x.old_logp_qubit for x in transitions],
            dtype=torch.float32,
            device=config.device,
        )
        old_logp_cond_all = torch.tensor(
            [x.old_logp_cond for x in transitions],
            dtype=torch.float32,
            device=config.device,
        )
        old_value_all = torch.tensor(
            [x.old_value for x in transitions],
            dtype=torch.float32,
            device=config.device,
        )
        qubit_idx_all = torch.tensor(
            [x.qubit_idx for x in transitions], dtype=torch.long, device=config.device
        )
        qpu_idx_all = torch.tensor(
            [x.qpu_idx for x in transitions], dtype=torch.long, device=config.device
        )
        adv_all = torch.tensor(
            [x.advantage for x in transitions],
            dtype=torch.float32,
            device=config.device,
        )
        ret_all = torch.tensor(
            [x.ret for x in transitions], dtype=torch.float32, device=config.device
        )
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std(unbiased=False) + 1e-8)
        if bool(config.normalize_value_targets):
            value_mean = ret_all.mean()
            value_std = ret_all.std(unbiased=False).clamp_min(1e-6)
        else:
            value_mean = torch.tensor(0.0, device=config.device)
            value_std = torch.tensor(1.0, device=config.device)

        batched_states = {
            "qpu_features": torch.stack(
                [x.tensors["qpu_features"] for x in transitions], dim=0
            ),
            "qpu_adj": torch.stack([x.tensors["qpu_adj"] for x in transitions], dim=0),
            "qubit_features": torch.stack(
                [x.tensors["qubit_features"] for x in transitions], dim=0
            ),
            "qubit_adj": torch.stack(
                [x.tensors["qubit_adj"] for x in transitions], dim=0
            ),
            "global_features": torch.stack(
                [x.tensors["global_features"] for x in transitions], dim=0
            ),
            "action_mask": torch.stack(
                [x.tensors["action_mask"] for x in transitions], dim=0
            ),
            "pair_prior": torch.stack(
                [x.tensors["pair_prior"] for x in transitions], dim=0
            ),
        }

        idx_all = np.arange(len(transitions), dtype=np.int64)
        mb_size = max(1, min(int(config.minibatch_size), len(transitions)))

        early_stop = False
        for _ in range(config.ppo_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, len(idx_all), mb_size):
                mb = idx_all[start : start + mb_size]
                mb_t = torch.tensor(mb, dtype=torch.long, device=config.device)
                with _amp_context(config.device, use_amp=use_cuda):
                    _, qubit_logits_b, cond_logits_b, value_t = model.forward_hier(
                        batched_states["qpu_features"].index_select(0, mb_t),
                        batched_states["qpu_adj"].index_select(0, mb_t),
                        batched_states["qubit_features"].index_select(0, mb_t),
                        batched_states["qubit_adj"].index_select(0, mb_t),
                        batched_states["global_features"].index_select(0, mb_t),
                        batched_states["pair_prior"].index_select(0, mb_t),
                    )
                qubit_logits_b = qubit_logits_b.float()
                cond_logits_b = cond_logits_b.float()
                value_t = value_t.float()
                mask_b = batched_states["action_mask"].index_select(0, mb_t)
                (
                    new_logp_q_t,
                    new_logp_cond_t,
                    new_logp_t,
                    entropy_q_t,
                    entropy_cond_t,
                ) = _hierarchical_logps_and_entropy_batched(
                    qubit_logits=qubit_logits_b,
                    cond_logits=cond_logits_b,
                    action_mask_flat=mask_b,
                    qubit_idx=qubit_idx_all[mb_t],
                    qpu_idx=qpu_idx_all[mb_t],
                )
                ppo_obj_q = ppo_clip_objective(
                    new_logp_q_t,
                    old_logp_qubit_all[mb_t],
                    adv_all[mb_t],
                    clip_eps=config.clip_eps,
                )
                ppo_obj_cond = ppo_clip_objective(
                    new_logp_cond_t,
                    old_logp_cond_all[mb_t],
                    adv_all[mb_t],
                    clip_eps=config.clip_eps,
                )
                actor_loss = -(
                    float(config.hier_qubit_coef) * ppo_obj_q
                    + float(config.hier_cond_coef) * ppo_obj_cond
                )
                entropy_t = (
                    float(config.hier_qubit_entropy_ratio) * entropy_q_t
                    + (1.0 - float(config.hier_qubit_entropy_ratio)) * entropy_cond_t
                )
                ret_mb = ret_all[mb_t]
                value_t_n = (value_t - value_mean) / value_std
                value_old_t = (old_value_all[mb_t] - value_mean) / value_std
                ret_mb_n = (ret_mb - value_mean) / value_std
                value_clipped = value_old_t + (value_t_n - value_old_t).clamp(
                    -config.value_clip_eps, config.value_clip_eps
                )
                critic_unclipped = (value_t_n - ret_mb_n).pow(2)
                critic_clipped = (value_clipped - ret_mb_n).pow(2)
                critic_loss = 0.5 * torch.max(critic_unclipped, critic_clipped).mean()
                entropy_loss = -entropy_t.mean()
                loss = (
                    actor_loss
                    + config.value_coef * critic_loss
                    + entropy_coef * entropy_loss
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                if config.target_kl > 0:
                    approx_kl = float((old_logp_all[mb_t] - new_logp_t).mean().item())
                    if approx_kl > 1.5 * float(config.target_kl):
                        early_stop = True
                        break
            if early_stop:
                break

        if bool(config.sil_enable) and len(sil_buffer) >= max(
            8, int(config.sil_batch_size // 2)
        ):
            sil_updates = max(1, int(config.sil_updates_per_iter))
            sil_batch_size = max(8, int(config.sil_batch_size))
            sil_pool = list(sil_buffer)
            for _ in range(sil_updates):
                k = min(sil_batch_size, len(sil_pool))
                chosen = random.sample(sil_pool, k=k)
                sil_states, sil_qubit_idx, sil_qpu_idx, sil_ret, sil_adv_pos = (
                    _sil_samples_to_device_batch(
                        chosen,
                        device=config.device,
                    )
                )
                with _amp_context(config.device, use_amp=use_cuda):
                    _, sil_qubit_logits, sil_cond_logits, sil_value = (
                        model.forward_hier(
                            sil_states["qpu_features"],
                            sil_states["qpu_adj"],
                            sil_states["qubit_features"],
                            sil_states["qubit_adj"],
                            sil_states["global_features"],
                            sil_states["pair_prior"],
                        )
                    )
                sil_qubit_logits = sil_qubit_logits.float()
                sil_cond_logits = sil_cond_logits.float()
                sil_value = sil_value.float()
                sil_logp_q, sil_logp_cond, _, _, _ = (
                    _hierarchical_logps_and_entropy_batched(
                        qubit_logits=sil_qubit_logits,
                        cond_logits=sil_cond_logits,
                        action_mask_flat=sil_states["action_mask"],
                        qubit_idx=sil_qubit_idx,
                        qpu_idx=sil_qpu_idx,
                    )
                )
                sil_logp = (
                    float(config.hier_qubit_coef) * sil_logp_q
                    + float(config.hier_cond_coef) * sil_logp_cond
                )
                sil_w = torch.clamp(
                    sil_adv_pos, min=0.0, max=float(config.sil_adv_clip)
                )
                sil_policy_loss = -(sil_w.detach() * sil_logp).mean()
                sil_value_loss = 0.5 * (sil_value - sil_ret).pow(2).mean()
                sil_loss = float(config.sil_coef) * (
                    sil_policy_loss + float(config.sil_value_coef) * sil_value_loss
                )
                opt.zero_grad()
                sil_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

        if config.select_best_checkpoint and (
            ep_cursor % max(1, int(config.select_interval_episodes)) == 0
            or ep_cursor >= config.episodes
        ):
            _snapshot_if_better()
        if stop_training:
            break

    restore_state = None
    if config.select_best_checkpoint and best_state is not None:
        restore_state = best_state
    if plateau_enable and plateau_best_state is not None:
        if restore_state is None:
            restore_state = plateau_best_state
        elif float(plateau_best_cost) < float(best_cost):
            restore_state = plateau_best_state
    if restore_state is not None:
        model.load_state_dict(
            {k: v.to(config.device) for k, v in restore_state.items()}
        )

    return model, hist


def infer_mapping(
    model: MappingActorCritic,
    circuit,
    topology: DQCTopology,
    device: str = "cpu",
    num_samples: int = 4,
    sample_temperatures: Optional[Sequence[float]] = None,
    sample_seed: Optional[int] = None,
    rollout_batch_size: int = 1,
    qubit_adj_mode: str = "raw",
    use_amp: bool = True,
) -> Tuple[Dict[int, int], float]:
    env = MappingEnv(circuit, topology, qubit_adj_mode=qubit_adj_mode)
    _, best_mapping, best_cost = _run_episode(
        model,
        env,
        device=device,
        greedy=True,
        use_amp=use_amp,
    )

    temps = list(sample_temperatures) if sample_temperatures else [1.0]
    temps = [max(1e-4, float(t)) for t in temps]
    total_samples = max(0, int(num_samples))
    bsz = max(1, int(rollout_batch_size))
    if total_samples > 0 and bsz > 1:
        gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
        sample_gens: List[Optional[torch.Generator]] = [None] * total_samples
        if sample_seed is not None:
            for i in range(total_samples):
                g = torch.Generator(device=gen_device)
                g.manual_seed(int(sample_seed) + 1009 * i)
                sample_gens[i] = g
        start = 0
        while start < total_samples:
            end = min(total_samples, start + bsz)
            cur_temps = [temps[i % len(temps)] for i in range(start, end)]
            cur_gens = sample_gens[start:end]
            batch_out = _run_sample_episodes_batch(
                model,
                circuit,
                topology,
                device=device,
                sample_temperatures=cur_temps,
                generators=cur_gens,
                qubit_adj_mode=qubit_adj_mode,
                use_amp=use_amp,
            )
            for cand_mapping, cand_cost in batch_out:
                if cand_cost < best_cost:
                    best_mapping, best_cost = cand_mapping, cand_cost
            start = end
    else:
        generator: Optional[torch.Generator] = None
        if sample_seed is not None:
            gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
            generator = torch.Generator(device=gen_device)
            generator.manual_seed(int(sample_seed))
        for _ in range(total_samples):
            temp = temps[_ % len(temps)]
            _, cand_mapping, cand_cost = _run_episode(
                model,
                env,
                device=device,
                greedy=False,
                sample_temperature=temp,
                generator=generator,
                use_amp=use_amp,
            )
            if cand_cost < best_cost:
                best_mapping, best_cost = cand_mapping, cand_cost

    return best_mapping, float(best_cost)

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import json
import math
import re

import numpy as np
import torch

from .circuit import QuantumCircuit
from .cost import logical_reuse_aware_cx_counts
from .equiv_rewrite import load_equivalence_rules
from .io_qasm import dump_qasm_circuit, load_qasm_circuit
from .topology import load_topology_file
from .train_mapping import MappingTrainConfig, infer_mapping, train_mapping_agent
from .train_transform import (
    TransformTrainConfig,
    collect_transform_candidates,
    train_transform_agent,
)
from .unitary_check import check_circuit_equivalence


@dataclass
class EvalResult:
    benchmark: str
    topology: str
    qmactr: float
    equivalence_ok: bool
    equivalence_mode: str
    equivalence_detail: str
    qmactr_qasm_path: str
    qmactr_off_same_model: float = float("nan")
    transform_gain_vs_off_same_model_pct: float = 0.0


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _circuit_signature(c: QuantumCircuit) -> Tuple:
    return tuple(
        (g.op, g.q0, g.q1, None if g.param is None else round(float(g.param), 8))
        for g in c.gates
    )


def _seed_offset_from_signature(sig: Tuple, mod: int = 1000003) -> int:
    s = json.dumps(sig, separators=(",", ":"), ensure_ascii=True)
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False) % max(1, int(mod))


def evaluate_one(
    benchmark_name: str,
    topology_path: str,
    out_dir: Path,
    seed: int = 0,
    mapping_episodes: int = 220,
    transform_episodes: int = 160,
    strict_oracle_samples: int = 0,
    strict_final_samples: int = 0,
    device: str = "auto",
    transform_restarts: int = 1,
    transform_max_steps: int | None = None,
    map_restarts_override: int | None = None,
    mapping_early_stop_enable: bool = True,
    mapping_early_stop_warmup_ratio: float = 0.2,
    mapping_early_stop_patience: int = 48,
    mapping_early_stop_rel_delta: float = 1e-3,
    mapping_early_stop_abs_delta: float = 1.0,
    mapping_early_stop_eval_interval: int = 4,
    transform_early_stop_enable: bool = True,
    transform_early_stop_warmup_ratio: float = 0.25,
    transform_early_stop_patience: int = 32,
    transform_early_stop_rel_delta: float = 5e-4,
    transform_early_stop_abs_delta: float = 1.0,
    transform_early_stop_eval_interval: int = 2,
    transform_enable: bool = True,
    transform_equiv_enable: bool = False,
    transform_equiv_lib: str = "",
    transform_equiv_max_matches: int = 16,
    transform_max_dag_nodes_override: int | None = None,
    transform_dag_context_override: int | None = None,
    transform_large_opportunity_boost: bool = True,
    transform_dense_reward_boost: bool = True,
    transform_reward_mode: str = "oracle",
    mapping_edge_aware: bool = True,
    mapping_qubit_adj_mode: str = "raw",
) -> Tuple[EvalResult, Dict[str, List[float]], Dict[str, List[float]]]:
    cuda_available = torch.cuda.is_available()
    if device == "auto":
        runtime_device = "cuda" if cuda_available else "cpu"
    else:
        runtime_device = device

    if str(runtime_device).startswith("cuda") and not cuda_available:
        print(f"[QMACTR][device] requested={device} resolved=cpu (cuda unavailable)")
        runtime_device = "cpu"
    elif str(runtime_device).startswith("cuda"):
        try:
            tdev = torch.device(runtime_device)
            didx = 0 if tdev.index is None else int(tdev.index)
            dname = torch.cuda.get_device_name(didx)
            print(
                f"[QMACTR][device] requested={device} resolved={runtime_device} ({dname})"
            )
        except Exception as ex:
            print(
                f"[QMACTR][device] requested={device} resolved=cpu (cuda init failed: {ex})"
            )
            runtime_device = "cpu"
    else:
        print(f"[QMACTR][device] requested={device} resolved=cpu")

    circuit_path = Path(benchmark_name).expanduser()
    print(f"Loading circuit from {circuit_path}")
    circuit = load_qasm_circuit(circuit_path)
    circuit_ops = {g.op for g in circuit.gates}

    if not transform_equiv_lib:
        ibm_paths = [
            Path("ecc/ibm_325_ecc.json"),
            Path("ecc/ibm_ecc.json"),
            Path("data/ecc/ibm_325_ecc.json"),
            Path("data/ecc/ibm_ecc.json"),
            Path("ibm_325_ecc.json"),
            Path("ibm_ecc.json"),
            Path("/workspace/ecc/ibm_325_ecc.json"),
            Path("/workspace/ecc/ibm_ecc.json"),
            Path("/workspace/data/ecc/ibm_325_ecc.json"),
            Path("/workspace/data/ecc/ibm_ecc.json"),
            Path("/workspace/ibm_325_ecc.json"),
            Path("/workspace/ibm_ecc.json"),
        ]
        nam_paths = [
            Path("ecc/nam_ecc.json"),
            Path("data/ecc/nam_ecc.json"),
            Path("nam_ecc.json"),
            Path("/workspace/ecc/nam_ecc.json"),
            Path("/workspace/data/ecc/nam_ecc.json"),
            Path("/workspace/nam_ecc.json"),
        ]
        search_paths = (
            ibm_paths + nam_paths if "sx" in circuit_ops else nam_paths + ibm_paths
        )
        for p in search_paths:
            if p.exists():
                transform_equiv_lib = str(p)
                break
    if transform_equiv_lib and not transform_equiv_enable:
        transform_equiv_enable = True

    benchmark_label = (
        Path(benchmark_name).name if Path(benchmark_name).exists() else benchmark_name
    )
    topology = load_topology_file(topology_path)
    topology_label = topology.name or Path(topology_path).stem
    capacity_total = sum(int(v) for v in topology.capacities.values())
    if capacity_total < int(circuit.num_qubits):
        raise ValueError(
            f"Topology '{topology_label}' has total capacity {capacity_total}, "
            f"but circuit '{benchmark_label}' needs {circuit.num_qubits} qubits."
        )
    pair_cost_mat = topology.pair_cost_matrix().astype(np.float32)
    if pair_cost_mat.shape[0] > 1:
        _mask = ~np.eye(pair_cost_mat.shape[0], dtype=bool)
        _pc_vals = pair_cost_mat[_mask]
        topo_avg_pair_cost = float(np.mean(_pc_vals))
        topo_p90_pair_cost = float(np.quantile(_pc_vals, 0.90))
    else:
        topo_avg_pair_cost = 0.0
        topo_p90_pair_cost = 0.0

    mapping_tight_topology = bool(
        topo_avg_pair_cost >= 28.0 or topo_p90_pair_cost >= 40.0
    )
    raw_dir, effective_dir = logical_reuse_aware_cx_counts(circuit)
    raw_total = float(raw_dir.sum())
    effective_total = float(effective_dir.sum())
    compression = effective_total / max(1e-12, raw_total)
    print(
        f"[QMACTR][mapping_adj] mode={mapping_qubit_adj_mode} "
        f"raw_dir_total={raw_total:.1f} effective_dir_total={effective_total:.1f} "
        f"compression={compression:.6f}"
    )

    complexity = max(1.0, float(circuit.num_qubits) / 24.0)

    cap_vals = np.array(
        [float(topology.capacities[i]) for i in range(topology.num_qpus)],
        dtype=np.float32,
    )
    cap_mean = float(cap_vals.mean()) if cap_vals.size > 0 else 0.0
    cap_cv = float(cap_vals.std() / max(1e-6, cap_mean)) if cap_mean > 0 else 0.0
    cx_count = int(circuit.cx_count())
    cx_deg = np.zeros((circuit.num_qubits,), dtype=np.float32)
    for g in circuit.gates:
        if g.op == "cx" and g.q1 is not None:
            cx_deg[g.q0] += 1.0
            cx_deg[g.q1] += 1.0
    cx_mean = float(cx_deg.mean()) if cx_deg.size > 0 else 0.0
    cx_cv = float(cx_deg.std() / max(1e-6, cx_mean)) if cx_mean > 0 else 0.0
    mapping_heavy_regime = bool(circuit.num_qubits >= 24 and cx_count >= 300)
    mapping_stress_regime = bool(cap_cv >= 0.40 and cx_cv >= 0.30 and cx_count >= 200)

    mapping_episodes_eff = int(mapping_episodes)
    transform_episodes_eff = int(transform_episodes)

    if str(transform_reward_mode) not in {"oracle", "structural"}:
        raise ValueError(f"Unsupported transform_reward_mode: {transform_reward_mode}")

    map_hidden_dim = 128
    map_heads = 4
    transform_hidden_dim = 96 if circuit.num_qubits >= 30 else 72

    oracle_topo_mul, final_topo_mul = 1.4, 2.4
    sample_temps = [0.5, 0.6, 0.7, 0.8, 0.95, 1.0, 1.15, 1.3, 1.4, 1.5, 1.8, 2.2]
    if circuit.num_qubits >= 32:
        sample_temps = sorted(set(sample_temps + [0.75, 1.35]))

    eval_seed_ensemble = 2 if circuit.num_qubits >= 30 else 1
    if mapping_heavy_regime:
        eval_seed_ensemble = max(eval_seed_ensemble, 2)
    if mapping_stress_regime:
        eval_seed_ensemble = max(eval_seed_ensemble, 2)
    if not bool(transform_enable) or int(transform_episodes_eff) <= 0:
        eval_seed_ensemble = max(eval_seed_ensemble, 2)
    if str(runtime_device).startswith("cuda"):
        if circuit.num_qubits >= 40:
            mapping_infer_batch = 2
        elif circuit.num_qubits >= 30:
            mapping_infer_batch = 4
        elif circuit.num_qubits >= 20:
            mapping_infer_batch = 6
        else:
            mapping_infer_batch = 8
    else:
        mapping_infer_batch = 1

    def _infer_cost_ensemble(
        c: QuantumCircuit,
        num_samples: int,
        temps: List[float],
        base_seed: int,
        ensemble: int,
    ) -> float:
        e = max(1, int(ensemble))
        per_samples = max(1, int(math.ceil(float(max(1, int(num_samples))) / float(e))))
        best = float("inf")
        for sidx in range(e):
            _, cand_cost = infer_mapping(
                mapping_model,
                c,
                topology,
                device=runtime_device,
                num_samples=per_samples,
                sample_temperatures=temps,
                sample_seed=base_seed + 1009 * sidx,
                rollout_batch_size=mapping_infer_batch,
                qubit_adj_mode=mapping_qubit_adj_mode,
                use_amp=str(runtime_device).startswith("cuda"),
            )
            if cand_cost < best:
                best = float(cand_cost)
        return float(best)

    sample_complexity_mul = min(1.5, complexity**0.5)

    oracle_samples_base = (
        int(strict_oracle_samples) if int(strict_oracle_samples) > 0 else 1
    )
    final_samples_base = (
        int(strict_final_samples) if int(strict_final_samples) > 0 else 8
    )
    oracle_samples_eff = int(
        max(
            oracle_samples_base,
            round(oracle_samples_base * oracle_topo_mul * sample_complexity_mul),
        )
    )
    final_samples_eff = int(
        max(
            final_samples_base,
            round(final_samples_base * final_topo_mul * sample_complexity_mul),
        )
    )
    if circuit.num_qubits >= 32:
        final_samples_eff = max(final_samples_eff, 16)
        oracle_samples_eff = max(oracle_samples_eff, 2)
    if not bool(transform_enable) or int(transform_episodes_eff) <= 0:
        final_samples_eff = max(final_samples_eff, 192)
        oracle_samples_eff = max(oracle_samples_eff, 4)
    if mapping_stress_regime:
        final_samples_eff = max(final_samples_eff, 160)
        oracle_samples_eff = max(oracle_samples_eff, 8)
    elif mapping_heavy_regime:
        final_samples_eff = max(final_samples_eff, 128)
        oracle_samples_eff = max(oracle_samples_eff, 6)

    if mapping_tight_topology:
        map_ppo_epochs, map_rollout_batch, map_entropy = 6, 4, 0.014
        map_temp_start, map_temp_end, map_temp_jitter = 1.35, 0.90, 0.10
        map_ckpt_interval = 6
        map_ckpt_samples = 4
    else:
        map_ppo_epochs, map_rollout_batch, map_entropy = 8, 6, 0.010
        map_temp_start, map_temp_end, map_temp_jitter = 1.20, 0.85, 0.06
        map_ckpt_interval = 4
        map_ckpt_samples = 6
    map_target_kl = 0.015
    if circuit.num_qubits >= 32:
        map_ppo_epochs = max(map_ppo_epochs, 6)
        map_rollout_batch = max(map_rollout_batch, 4)

    num_map_restarts = 4
    if circuit.num_qubits >= 32:
        num_map_restarts = max(num_map_restarts, 4)
    if complexity >= 1.3:
        num_map_restarts = max(num_map_restarts, 4)
    if mapping_episodes_eff >= 1200:
        num_map_restarts = max(num_map_restarts, 4)
    elif mapping_heavy_regime:
        num_map_restarts = max(num_map_restarts, 4)
    if mapping_stress_regime:
        num_map_restarts = max(num_map_restarts, 4)
    if map_restarts_override is not None:
        num_map_restarts = max(1, int(map_restarts_override))
    if mapping_heavy_regime:
        map_ckpt_samples = max(map_ckpt_samples, 4 if mapping_tight_topology else 6)
    if mapping_stress_regime:
        map_ckpt_interval = min(
            map_ckpt_interval, 4 if not mapping_tight_topology else 6
        )
        map_ckpt_samples = max(map_ckpt_samples, 4 if mapping_tight_topology else 6)

    mapping_model = None
    map_hist: Dict[str, List[float]] = {"reward": [], "cost": []}
    best_map_cost = float("inf")
    for midx in range(num_map_restarts):
        map_cfg = MappingTrainConfig(
            episodes=mapping_episodes_eff,
            ppo_epochs=map_ppo_epochs,
            rollout_batch_episodes=map_rollout_batch,
            entropy_coef=map_entropy,
            entropy_final_ratio=0.25,
            minibatch_size=64,
            target_kl=map_target_kl,
            rollout_temp_start=map_temp_start,
            rollout_temp_end=map_temp_end,
            rollout_temp_jitter=map_temp_jitter,
            seed=seed + 313 * midx,
            device=runtime_device,
            hidden_dim=map_hidden_dim,
            heads=map_heads,
            edge_aware=bool(mapping_edge_aware),
            select_best_checkpoint=True,
            select_interval_episodes=map_ckpt_interval,
            select_num_samples=map_ckpt_samples,
            qubit_adj_mode=mapping_qubit_adj_mode,
            early_stop_enable=bool(mapping_early_stop_enable),
            early_stop_warmup_ratio=float(mapping_early_stop_warmup_ratio),
            early_stop_patience=int(mapping_early_stop_patience),
            early_stop_rel_delta=float(mapping_early_stop_rel_delta),
            early_stop_abs_delta=float(mapping_early_stop_abs_delta),
            early_stop_eval_interval=int(mapping_early_stop_eval_interval),
        )
        cand_model, cand_hist = train_mapping_agent(circuit, topology, map_cfg)
        map_select_ensemble = 3
        if mapping_heavy_regime:
            map_select_ensemble = max(map_select_ensemble, 3)
        if mapping_stress_regime:
            map_select_ensemble = max(map_select_ensemble, 3)
        e = max(1, int(map_select_ensemble))
        if not bool(transform_enable) or int(transform_episodes_eff) <= 0:
            map_select_total_samples = max(16, int(final_samples_eff))
        else:
            map_select_total_samples = max(
                8, int(round(0.75 * float(final_samples_eff)))
            )
        if mapping_heavy_regime:
            map_select_total_samples = max(map_select_total_samples, 72)
        if mapping_stress_regime:
            map_select_total_samples = max(map_select_total_samples, 96)
        per_samples = max(1, int(math.ceil(float(map_select_total_samples) / float(e))))
        cand_cost = float("inf")
        for sidx in range(e):
            _, cur_cost = infer_mapping(
                cand_model,
                circuit,
                topology,
                device=runtime_device,
                num_samples=per_samples,
                sample_temperatures=sample_temps,
                sample_seed=seed + 10007 * (midx + 1) + 313 * sidx,
                rollout_batch_size=mapping_infer_batch,
                qubit_adj_mode=mapping_qubit_adj_mode,
                use_amp=str(runtime_device).startswith("cuda"),
            )
            if cur_cost < cand_cost:
                cand_cost = float(cur_cost)
        if cand_cost < best_map_cost or mapping_model is None:
            best_map_cost = float(cand_cost)
            mapping_model = cand_model
            map_hist = cand_hist

    assert mapping_model is not None

    if circuit.gate_count() >= 800 or cx_count >= 300:
        oracle_train_samples = 2
    else:
        oracle_train_samples = 1
    oracle_train_ensemble = 1
    if mapping_stress_regime:
        oracle_train_samples = max(oracle_train_samples, 3)
        oracle_train_ensemble = max(oracle_train_ensemble, 3)
    elif mapping_heavy_regime:
        oracle_train_samples = max(oracle_train_samples, 2)
        oracle_train_ensemble = max(oracle_train_ensemble, 2)
    if circuit.gate_count() >= 800 or cx_count >= 300:
        oracle_train_ensemble = max(oracle_train_ensemble, 2)
    if circuit.gate_count() >= 800 or cx_count >= 300:
        oracle_train_temps = [t for t in sample_temps if 0.75 <= float(t) <= 1.35]
        if not oracle_train_temps:
            oracle_train_temps = [0.85, 1.0, 1.15]
    else:
        oracle_train_temps = (
            list(sample_temps[: min(5, len(sample_temps))]) if sample_temps else [1.0]
        )
    if 1.0 not in oracle_train_temps:
        oracle_train_temps.append(1.0)
    oracle_train_temps = sorted(set(float(t) for t in oracle_train_temps))
    oracle_train_calls = 0
    oracle_eval_calls = 0

    oracle_train_cache: Dict[Tuple, float] = {}
    oracle_eval_cache: Dict[Tuple, float] = {}
    oracle_final_cache: Dict[Tuple, float] = {}

    def _robust_train_oracle_score(costs: List[float]) -> float:
        arr = np.array([float(x) for x in costs], dtype=np.float64)
        if arr.size == 0:
            return float("inf")
        if arr.size >= 3:
            s = np.sort(arr)
            core = s[1:-1]
        else:
            core = arr
        mean_core = float(np.mean(core))
        std_all = float(np.std(arr))

        return float(mean_core + 0.20 * std_all)

    def qmactr_oracle_train(c: QuantumCircuit) -> float:
        nonlocal oracle_train_calls
        sig = _circuit_signature(c)
        cached = oracle_train_cache.get(sig)
        if cached is not None:
            return float(cached)
        oracle_train_calls += 1
        s_off = _seed_offset_from_signature(sig)
        costs: List[float] = []
        for eidx in range(max(1, int(oracle_train_ensemble))):
            _, cost = infer_mapping(
                mapping_model,
                c,
                topology,
                device=runtime_device,
                num_samples=max(0, oracle_train_samples),
                sample_temperatures=oracle_train_temps,
                sample_seed=seed + 200003 + int(s_off) + 1091 * eidx,
                rollout_batch_size=mapping_infer_batch,
                qubit_adj_mode=mapping_qubit_adj_mode,
                use_amp=str(runtime_device).startswith("cuda"),
            )
            costs.append(float(cost))
        out = _robust_train_oracle_score(costs)
        oracle_train_cache[sig] = out
        return out

    def qmactr_oracle_eval(c: QuantumCircuit) -> float:
        nonlocal oracle_eval_calls
        sig = _circuit_signature(c)
        cached = oracle_eval_cache.get(sig)
        if cached is not None:
            return float(cached)
        oracle_eval_calls += 1
        s_off = _seed_offset_from_signature(sig)
        cost = _infer_cost_ensemble(
            c,
            num_samples=max(1, int(oracle_samples_eff)),
            temps=sample_temps,
            base_seed=seed + 400009 + int(s_off),
            ensemble=eval_seed_ensemble,
        )
        out = float(cost)
        oracle_eval_cache[sig] = out
        return out

    def qmactr_oracle_final(c: QuantumCircuit) -> float:
        sig = _circuit_signature(c)
        cached = oracle_final_cache.get(sig)
        if cached is not None:
            return float(cached)
        s_off = _seed_offset_from_signature(sig)
        cost = _infer_cost_ensemble(
            c,
            num_samples=max(1, int(final_samples_eff)),
            temps=sample_temps,
            base_seed=seed + 700001 + int(s_off),
            ensemble=eval_seed_ensemble,
        )
        out = float(cost)
        oracle_final_cache[sig] = out
        return out

    qmactr_off_same_model = qmactr_oracle_final(circuit)

    tr_hist: Dict[str, List[float]] = {"reward": [], "best_cost": []}
    transform_candidates: List[QuantumCircuit] = [circuit.copy()]
    best_circuit = circuit.copy()
    best_proxy_cost = qmactr_oracle_eval(circuit)
    if bool(transform_enable) and int(transform_episodes_eff) > 0:

        if transform_max_steps is None:
            adaptive_max_steps = min(180, max(64, circuit.gate_count() // 12))
        else:
            adaptive_max_steps = int(transform_max_steps)
        if circuit.gate_count() >= 2000:
            adaptive_max_steps = min(adaptive_max_steps, 48)

        num_restarts = max(1, int(transform_restarts))
        if int(transform_restarts) <= 1:

            num_restarts = 1
            if transform_episodes_eff >= 120 and (
                mapping_heavy_regime or circuit.num_qubits >= 32
            ):
                num_restarts = max(num_restarts, 2)
            if transform_episodes_eff >= 600 and circuit.num_qubits >= 32:
                num_restarts = max(num_restarts, 3)
        tr_ppo_epochs = 5
        tr_rollout_batch = 3
        tr_greedy_eval = 4
        tr_entropy = 0.012
        large_transform_case = bool(circuit.gate_count() >= 800 or cx_count >= 300)
        if cx_count >= 200:
            tr_rollout_batch = max(tr_rollout_batch, 3)
            tr_greedy_eval = max(tr_greedy_eval, 4)
        if circuit.num_qubits >= 32:
            tr_ppo_epochs = max(tr_ppo_epochs, 5)
            tr_rollout_batch = max(tr_rollout_batch, 3)

        if large_transform_case:
            tr_max_dag_nodes, tr_dag_context = 1536, 40
            if circuit.gate_count() >= 1600 or cx_count >= 550:
                tr_max_dag_nodes, tr_dag_context = 2048, 48
        else:
            tr_max_dag_nodes, tr_dag_context = 1400, 28
        if transform_max_dag_nodes_override is not None:
            tr_max_dag_nodes = max(64, int(transform_max_dag_nodes_override))
        if transform_dag_context_override is not None:
            tr_dag_context = max(0, int(transform_dag_context_override))
        tr_max_equiv_matches = int(max(1, transform_equiv_max_matches))
        if transform_episodes_eff >= 120 and circuit.gate_count() < 2000:
            tr_max_equiv_matches = max(tr_max_equiv_matches, 6)
        if (
            bool(transform_large_opportunity_boost)
            and large_transform_case
            and transform_episodes_eff >= 120
        ):
            tr_max_equiv_matches = max(tr_max_equiv_matches, 32)
            if circuit.gate_count() >= 1500 or cx_count >= 500:
                tr_max_equiv_matches = max(tr_max_equiv_matches, 48)
        elif large_transform_case:
            tr_max_equiv_matches = max(tr_max_equiv_matches, 12)
        tr_max_equiv_matches = min(tr_max_equiv_matches, 64)

        small_transform_case = bool(circuit.gate_count() <= 900 and cx_count <= 260)
        tr_candidate_refine_rounds = 2 if transform_episodes_eff >= 120 else 1
        tr_candidate_refine_topk = 4 if cx_count >= 200 else 3
        if large_transform_case and transform_episodes_eff >= 120:
            tr_candidate_refine_rounds = max(tr_candidate_refine_rounds, 3)
            tr_candidate_refine_topk = max(tr_candidate_refine_topk, 6)
            if bool(transform_large_opportunity_boost):
                tr_candidate_refine_rounds = max(tr_candidate_refine_rounds, 4)
                tr_candidate_refine_topk = max(tr_candidate_refine_topk, 8)
        elif large_transform_case:
            tr_candidate_refine_topk = max(tr_candidate_refine_topk, 4)
        if small_transform_case and transform_episodes_eff >= 120:
            tr_candidate_refine_rounds = max(tr_candidate_refine_rounds, 3)
            tr_candidate_refine_topk = max(tr_candidate_refine_topk, 5)

        tr_noop_penalty_start = 0.14
        tr_noop_penalty_end = 0.05
        tr_novelty_bonus_weight = 0.03
        tr_equal_cost_change_bonus = 0.01
        tr_oracle_abs_reward_weight = 0.0
        tr_cx_reduction_reward_weight = 0.20
        tr_gate_reduction_reward_weight = 0.05
        tr_new_best_bonus_weight = 0.10
        tr_worse_cost_penalty_weight = 0.02
        if small_transform_case:
            tr_noop_penalty_start = 0.16
            tr_noop_penalty_end = 0.06
            tr_novelty_bonus_weight = 0.035
            tr_equal_cost_change_bonus = 0.015
        elif large_transform_case:
            tr_noop_penalty_start = 0.14
            tr_noop_penalty_end = 0.05
            tr_novelty_bonus_weight = 0.035
            tr_equal_cost_change_bonus = 0.016
            tr_cx_reduction_reward_weight = 0.24
            if bool(transform_dense_reward_boost):
                tr_oracle_abs_reward_weight = 0.10
                tr_gate_reduction_reward_weight = 0.06
                tr_new_best_bonus_weight = 0.12
                tr_worse_cost_penalty_weight = 0.018
        tr_oracle_rel_reward_weight = 1.0
        tr_best_metric = "oracle"
        if str(transform_reward_mode) == "structural":
            tr_oracle_rel_reward_weight = 0.0
            tr_oracle_abs_reward_weight = 0.0
            tr_cx_reduction_reward_weight = max(
                float(tr_cx_reduction_reward_weight), 1.0
            )
            tr_gate_reduction_reward_weight = max(
                float(tr_gate_reduction_reward_weight), 0.20
            )
            tr_new_best_bonus_weight = 0.0
            tr_worse_cost_penalty_weight = 0.0
            tr_best_metric = "structure"
        for ridx in range(num_restarts):
            tr_cfg = TransformTrainConfig(
                episodes=transform_episodes_eff,
                max_steps=adaptive_max_steps,
                ppo_epochs=tr_ppo_epochs,
                rollout_batch_episodes=tr_rollout_batch,
                minibatch_size=64,
                entropy_coef=tr_entropy,
                entropy_final_ratio=0.30,
                target_kl=0.02,
                hidden_dim=transform_hidden_dim,
                value_aggregate_mode="max",
                next_value_mode="max",
                greedy_eval_episodes=tr_greedy_eval,
                candidate_refine_rounds=tr_candidate_refine_rounds,
                candidate_refine_topk=tr_candidate_refine_topk,
                enable_equiv_rewrite=bool(transform_equiv_enable),
                equiv_lib_path=str(transform_equiv_lib or ""),
                max_equiv_matches=tr_max_equiv_matches,
                max_dag_nodes=tr_max_dag_nodes,
                dag_context=tr_dag_context,
                noop_penalty_start=tr_noop_penalty_start,
                noop_penalty_end=tr_noop_penalty_end,
                novelty_bonus_weight=tr_novelty_bonus_weight,
                equal_cost_change_bonus=tr_equal_cost_change_bonus,
                oracle_rel_reward_weight=tr_oracle_rel_reward_weight,
                oracle_abs_reward_weight=tr_oracle_abs_reward_weight,
                cx_reduction_reward_weight=tr_cx_reduction_reward_weight,
                gate_reduction_reward_weight=tr_gate_reduction_reward_weight,
                new_best_bonus_weight=tr_new_best_bonus_weight,
                worse_cost_penalty_weight=tr_worse_cost_penalty_weight,
                best_metric=tr_best_metric,
                use_amp=str(runtime_device).startswith("cuda"),
                amp_dtype="bf16",
                compile_model=bool(
                    str(runtime_device).startswith("cuda")
                    and transform_episodes_eff >= 120
                ),
                early_stop_enable=bool(transform_early_stop_enable),
                early_stop_warmup_ratio=float(transform_early_stop_warmup_ratio),
                early_stop_patience=int(transform_early_stop_patience),
                early_stop_rel_delta=float(transform_early_stop_rel_delta),
                early_stop_abs_delta=float(transform_early_stop_abs_delta),
                early_stop_eval_interval=int(transform_early_stop_eval_interval),
                seed=seed + 1009 * ridx,
                device=runtime_device,
            )
            tr_agent, cur_hist, cur_best_circuit, _ = train_transform_agent(
                circuit, qmactr_oracle_train, tr_cfg
            )
            transform_candidates.append(cur_best_circuit.copy())

            extra_sample_eps = 1 if circuit.gate_count() >= 2000 else 3
            if large_transform_case:
                extra_sample_eps = max(extra_sample_eps, 5)
            extra_max_candidates = 6
            if large_transform_case:
                extra_max_candidates = 10
                if bool(transform_large_opportunity_boost):
                    extra_sample_eps = max(extra_sample_eps, 8)
                    extra_max_candidates = max(extra_max_candidates, 16)
            if small_transform_case:
                extra_sample_eps = max(extra_sample_eps, 6)
                extra_max_candidates = max(extra_max_candidates, 12)
            extra_candidates = collect_transform_candidates(
                tr_agent,
                circuit,
                qmactr_oracle_eval,
                tr_cfg,
                greedy_episodes=max(2, int(tr_greedy_eval)),
                sample_episodes=extra_sample_eps,
                max_candidates=extra_max_candidates,
            )
            transform_candidates.extend([c.copy() for c in extra_candidates])
            cur_proxy_rl = qmactr_oracle_eval(cur_best_circuit)
            if ridx == 0:
                tr_hist = cur_hist

            if cur_proxy_rl < best_proxy_cost:
                best_proxy_cost = cur_proxy_rl
                best_circuit = cur_best_circuit.copy()
                tr_hist = cur_hist

    final_candidates = transform_candidates + [best_circuit, circuit]
    original_sig = _circuit_signature(circuit)
    unique_candidates: List[QuantumCircuit] = []
    seen = set()
    for cand in final_candidates:
        sig = _circuit_signature(cand)
        if sig not in seen:
            seen.add(sig)
            unique_candidates.append(cand)

    pre_ranked: List[Tuple[float, Tuple, QuantumCircuit]] = []
    for cand in unique_candidates:
        sig = _circuit_signature(cand)
        pre_ranked.append((float(qmactr_oracle_train(cand)), sig, cand))
    pre_ranked.sort(key=lambda x: float(x[0]))

    large_transform_case = bool(circuit.gate_count() >= 800 or cx_count >= 300)
    expanded: List[QuantumCircuit] = []
    if (
        bool(transform_enable)
        and int(transform_episodes_eff) > 0
        and large_transform_case
    ):
        expand_rules = None
        if bool(transform_equiv_enable) and str(transform_equiv_lib).strip():
            p = Path(str(transform_equiv_lib))
            if p.exists():
                try:
                    expand_rules = load_equivalence_rules(str(p.resolve()))
                except Exception:
                    expand_rules = None

        seen_sig = {_circuit_signature(c) for c in unique_candidates}
        seed_k = min(
            len(pre_ranked), 5 if bool(transform_large_opportunity_boost) else 3
        )
        seed_candidates = [c for _, _, c in pre_ranked[:seed_k]]
        first_hop_matches = max(6, int(max(1, transform_equiv_max_matches)))
        if bool(transform_large_opportunity_boost):
            first_hop_matches = max(first_hop_matches, 24)
            first_hop_matches = min(first_hop_matches, 48)
        else:
            first_hop_matches = min(first_hop_matches, 18)
        second_hop_matches = max(8, first_hop_matches // 2)
        first_hop_keep = 2 if bool(transform_large_opportunity_boost) else 1
        second_hop_keep = 2 if bool(transform_large_opportunity_boost) else 1
        max_new_expand = 12 if bool(transform_large_opportunity_boost) else 6
        expand_quick_cache: Dict[Tuple, float] = {}

        def _quick_expand_oracle(cand: QuantumCircuit) -> float:
            sig_c = _circuit_signature(cand)
            cached_q = expand_quick_cache.get(sig_c)
            if cached_q is not None:
                return float(cached_q)
            s_off_q = _seed_offset_from_signature(sig_c)
            _, quick_cost = infer_mapping(
                mapping_model,
                cand,
                topology,
                device=runtime_device,
                num_samples=1,
                sample_temperatures=(
                    oracle_train_temps[: min(3, len(oracle_train_temps))]
                    if oracle_train_temps
                    else [1.0]
                ),
                sample_seed=seed + 260003 + int(s_off_q),
                rollout_batch_size=max(1, min(int(mapping_infer_batch), 4)),
                qubit_adj_mode=mapping_qubit_adj_mode,
                use_amp=str(runtime_device).startswith("cuda"),
            )
            out_q = float(quick_cost)
            expand_quick_cache[sig_c] = out_q
            return out_q

        stop_expand = False
        for seed_c in seed_candidates:
            if stop_expand:
                break
            first_pool: List[Tuple[float, Tuple, QuantumCircuit]] = []
            try:
                opps1 = seed_c.opportunities(
                    equivalence_rules=expand_rules, max_equiv_matches=first_hop_matches
                )
            except Exception:
                opps1 = []
            for opp in opps1:
                for aidx in range(len(opp.replacements)):
                    cand1 = seed_c.apply(opp, aidx)
                    sig1 = _circuit_signature(cand1)
                    if sig1 in seen_sig:
                        continue
                    first_pool.append((float(_quick_expand_oracle(cand1)), sig1, cand1))
            first_pool.sort(key=lambda x: float(x[0]))

            for _, sig1, cand1 in first_pool[:first_hop_keep]:
                if sig1 not in seen_sig:
                    expanded.append(cand1.copy())
                    seen_sig.add(sig1)
                    if len(expanded) >= max_new_expand:
                        stop_expand = True
                        break
                second_pool: List[Tuple[float, Tuple, QuantumCircuit]] = []
                try:
                    opps2 = cand1.opportunities(
                        equivalence_rules=expand_rules,
                        max_equiv_matches=second_hop_matches,
                    )
                except Exception:
                    opps2 = []
                for opp2 in opps2:
                    for aidx2 in range(len(opp2.replacements)):
                        cand2 = cand1.apply(opp2, aidx2)
                        sig2 = _circuit_signature(cand2)
                        if sig2 in seen_sig:
                            continue
                        second_pool.append(
                            (float(_quick_expand_oracle(cand2)), sig2, cand2)
                        )
                second_pool.sort(key=lambda x: float(x[0]))
                for _, sig2, cand2 in second_pool[:second_hop_keep]:
                    if sig2 in seen_sig:
                        continue
                    expanded.append(cand2.copy())
                    seen_sig.add(sig2)
                    if len(expanded) >= max_new_expand:
                        stop_expand = True
                        break
                if stop_expand:
                    break

    if expanded:
        unique_candidates.extend(expanded)
        pre_ranked = []
        for cand in unique_candidates:
            sig = _circuit_signature(cand)
            pre_ranked.append((float(qmactr_oracle_train(cand)), sig, cand))
        pre_ranked.sort(key=lambda x: float(x[0]))

    pre_rank_eval_scores: Dict[Tuple, float] = {}
    if len(pre_ranked) > 8:
        small_transform_case = bool(circuit.gate_count() <= 900 and cx_count <= 260)
        if small_transform_case:
            rerank_window = min(
                len(pre_ranked), max(12, int(math.ceil(0.90 * len(pre_ranked))))
            )
            alpha = 0.55
        else:
            rerank_window = min(
                len(pre_ranked), max(10, int(math.ceil(0.70 * len(pre_ranked))))
            )
            alpha = 0.70
        mixed_ranked: List[Tuple[float, float, Tuple, QuantumCircuit]] = []
        for idx, (tr_score, sig, cand) in enumerate(pre_ranked):
            mix_score = float(tr_score)
            if idx < rerank_window:
                ev_score = float(qmactr_oracle_eval(cand))
                pre_rank_eval_scores[sig] = ev_score
                mix_score = float(alpha) * float(tr_score) + (
                    1.0 - float(alpha)
                ) * float(ev_score)
            mixed_ranked.append((float(mix_score), float(tr_score), sig, cand))
        mixed_ranked.sort(key=lambda x: (float(x[0]), float(x[1])))
        pre_ranked = [(float(tr), sig, cand) for _, tr, sig, cand in mixed_ranked]
    pre_rank_scores: Dict[Tuple, float] = {
        sig: float(tr_score) for tr_score, sig, _ in pre_ranked
    }

    if len(unique_candidates) <= 10:
        prefilter_k = len(unique_candidates)
    else:
        prefilter_ratio = 0.65
        prefilter_min = 10
        small_transform_case = bool(circuit.gate_count() <= 900 and cx_count <= 260)
        if small_transform_case:
            prefilter_ratio = 0.85
            prefilter_min = 12
        elif large_transform_case:
            prefilter_ratio = 0.92
            prefilter_min = 18
            if bool(transform_large_opportunity_boost):
                prefilter_ratio = 0.96
                prefilter_min = 24
        prefilter_k = min(
            len(unique_candidates),
            max(
                prefilter_min, int(math.ceil(prefilter_ratio * len(unique_candidates)))
            ),
        )
    strict_candidates: List[QuantumCircuit] = [
        c.copy() for _, _, c in pre_ranked[:prefilter_k]
    ]

    strict_candidates.extend([circuit.copy(), best_circuit.copy()])

    small_transform_case = bool(circuit.gate_count() <= 900 and cx_count <= 260)
    min_nonorig = 6 if small_transform_case else 4
    if large_transform_case:
        min_nonorig = max(min_nonorig, 8)
        if bool(transform_large_opportunity_boost):
            min_nonorig = max(min_nonorig, 12)
    strict_sig = {_circuit_signature(c) for c in strict_candidates}
    cur_nonorig = sum(
        1 for c in strict_candidates if _circuit_signature(c) != original_sig
    )
    if cur_nonorig < min_nonorig:
        for _, sig, cand in pre_ranked:
            if sig == original_sig or sig in strict_sig:
                continue
            strict_candidates.append(cand.copy())
            strict_sig.add(sig)
            cur_nonorig += 1
            if cur_nonorig >= min_nonorig:
                break

    low_cx_reserve_target = 0
    if large_transform_case:
        low_cx_reserve_target = 8
        if bool(transform_large_opportunity_boost):
            low_cx_reserve_target = max(low_cx_reserve_target, 12)
    elif len(unique_candidates) > 10:
        low_cx_reserve_target = 3
    if small_transform_case:
        low_cx_reserve_target = max(low_cx_reserve_target, 4)
    low_cx_reserve_added = 0
    if low_cx_reserve_target > 0:
        low_cx_pool: List[Tuple[int, int, float, Tuple, QuantumCircuit]] = []
        for cand in unique_candidates:
            sig = _circuit_signature(cand)
            if sig == original_sig or sig in strict_sig:
                continue
            low_cx_pool.append(
                (
                    int(cand.cx_count()),
                    int(cand.gate_count()),
                    float(pre_rank_scores.get(sig, float("inf"))),
                    sig,
                    cand,
                )
            )
        low_cx_pool.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, _, _, sig, cand in low_cx_pool:
            strict_candidates.append(cand.copy())
            strict_sig.add(sig)
            low_cx_reserve_added += 1
            if low_cx_reserve_added >= low_cx_reserve_target:
                break

    diverse_reserve_target = 0
    if large_transform_case:
        diverse_reserve_target = 6
        if bool(transform_large_opportunity_boost):
            diverse_reserve_target = max(diverse_reserve_target, 10)
    elif len(unique_candidates) > 12:
        diverse_reserve_target = 2
    diverse_reserve_added = 0
    if diverse_reserve_target > 0:

        cx_bin_w = 8 if (circuit.gate_count() >= 800 or cx_count >= 300) else 4
        g_bin_w = 16 if (circuit.gate_count() >= 800 or cx_count >= 300) else 8
        taken_bins = {
            (int(c.cx_count()) // cx_bin_w, int(c.gate_count()) // g_bin_w)
            for c in strict_candidates
        }
        diverse_pool: List[Tuple[float, int, int, Tuple, QuantumCircuit]] = []
        for _, sig, cand in pre_ranked:
            if sig == original_sig or sig in strict_sig:
                continue
            cx_bin = int(cand.cx_count()) // cx_bin_w
            g_bin = int(cand.gate_count()) // g_bin_w
            diverse_pool.append(
                (
                    float(pre_rank_scores.get(sig, float("inf"))),
                    cx_bin,
                    g_bin,
                    sig,
                    cand,
                )
            )
        diverse_pool.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, cx_bin, g_bin, sig, cand in diverse_pool:
            bkey = (int(cx_bin), int(g_bin))
            if bkey in taken_bins:
                continue
            strict_candidates.append(cand.copy())
            strict_sig.add(sig)
            taken_bins.add(bkey)
            diverse_reserve_added += 1
            if diverse_reserve_added >= diverse_reserve_target:
                break
    strict_unique: List[QuantumCircuit] = []
    strict_seen = set()
    for cand in strict_candidates:
        sig = _circuit_signature(cand)
        if sig in strict_seen:
            continue
        strict_seen.add(sig)
        strict_unique.append(cand)

    def _tie_key(c: QuantumCircuit) -> Tuple[int, int, int]:

        is_original = 1 if _circuit_signature(c) == original_sig else 0
        return (int(c.cx_count()), int(c.gate_count()), int(is_original))

    remap_refine_enable = (
        bool(transform_enable)
        and int(transform_episodes_eff) >= 120
        and (circuit.gate_count() >= 800 or cx_count >= 300)
    )
    remap_refine_topk = 0
    if remap_refine_enable:
        remap_refine_topk = 3 if (mapping_heavy_regime or mapping_stress_regime) else 2
    remap_refined_cost_cache: Dict[Tuple, float] = {}
    remap_refined_base_cost_cache: Dict[Tuple, float] = {}
    remap_refined_candidates: List[Dict[str, float | int | bool]] = []

    def _remap_refined_cost(cand: QuantumCircuit) -> float:
        sig = _circuit_signature(cand)
        cached = remap_refined_cost_cache.get(sig)
        if cached is not None:
            return float(cached)
        s_off = _seed_offset_from_signature(sig)
        remap_episodes = max(
            24, min(96, int(round(max(1, int(mapping_episodes)) * 0.20)))
        )
        if mapping_heavy_regime:
            remap_episodes = max(remap_episodes, 64)
        if mapping_stress_regime:
            remap_episodes = max(remap_episodes, 96)
        remap_cfg = MappingTrainConfig(
            episodes=int(remap_episodes),
            ppo_epochs=max(4, int(map_ppo_epochs)),
            entropy_coef=max(0.004, 0.70 * float(map_entropy)),
            rollout_batch_episodes=max(2, min(4, int(map_rollout_batch))),
            hidden_dim=map_hidden_dim,
            heads=map_heads,
            edge_aware=bool(mapping_edge_aware),
            target_kl=max(0.012, float(map_target_kl)),
            rollout_temp_start=max(1.0, float(map_temp_start)),
            rollout_temp_end=max(0.80, float(map_temp_end)),
            rollout_temp_jitter=min(0.08, float(map_temp_jitter)),
            select_best_checkpoint=True,
            select_interval_episodes=8,
            select_num_samples=2,
            qubit_adj_mode=mapping_qubit_adj_mode,
            early_stop_enable=True,
            early_stop_warmup_ratio=0.20,
            early_stop_patience=24,
            early_stop_rel_delta=1e-3,
            early_stop_abs_delta=1.0,
            early_stop_eval_interval=2,
            seed=seed + 910003 + int(s_off),
            device=runtime_device,
        )
        remap_model, _ = train_mapping_agent(cand, topology, remap_cfg)
        remap_samples = max(8, int(round(0.50 * float(final_samples_eff))))
        if mapping_heavy_regime:
            remap_samples = max(remap_samples, 24)
        if mapping_stress_regime:
            remap_samples = max(remap_samples, 32)
        _, remap_cost = infer_mapping(
            remap_model,
            cand,
            topology,
            device=runtime_device,
            num_samples=max(1, int(remap_samples)),
            sample_temperatures=sample_temps,
            sample_seed=seed + 930011 + int(s_off),
            rollout_batch_size=max(1, min(int(mapping_infer_batch), 4)),
            qubit_adj_mode=mapping_qubit_adj_mode,
            use_amp=str(runtime_device).startswith("cuda"),
        )
        remap_out = float(remap_cost)
        remap_refined_cost_cache[sig] = remap_out
        return remap_out

    if remap_refine_topk > 0 and strict_unique:
        ranked_for_refine: List[
            Tuple[float, float, int, int, Tuple, QuantumCircuit]
        ] = []
        for cand in strict_unique:
            sig = _circuit_signature(cand)
            tr_s = float(pre_rank_scores.get(sig, float("inf")))
            ev_s = float(pre_rank_eval_scores.get(sig, tr_s))
            ranked_for_refine.append(
                (tr_s, ev_s, int(cand.cx_count()), int(cand.gate_count()), sig, cand)
            )
        ranked_for_refine.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        shortlist: List[QuantumCircuit] = []
        shortlist_sig = set()
        for _, _, _, _, sig, cand in ranked_for_refine[:remap_refine_topk]:
            shortlist.append(cand)
            shortlist_sig.add(sig)

        nonorig_low_cx = [
            (int(c.cx_count()), int(c.gate_count()), _circuit_signature(c), c)
            for c in strict_unique
            if _circuit_signature(c) != original_sig
        ]
        nonorig_low_cx.sort(key=lambda x: (x[0], x[1]))
        if nonorig_low_cx:
            _, _, sig_l, cand_l = nonorig_low_cx[0]
            if sig_l not in shortlist_sig:
                shortlist.append(cand_l)
                shortlist_sig.add(sig_l)
        for cand in shortlist:
            sig = _circuit_signature(cand)
            base_cost = float(qmactr_oracle_final(cand))
            refined_cost = float(_remap_refined_cost(cand))
            remap_refined_base_cost_cache[sig] = base_cost
            remap_refined_candidates.append(
                {
                    "is_original": bool(sig == original_sig),
                    "gate_count": int(cand.gate_count()),
                    "cx_count": int(cand.cx_count()),
                    "base_final_oracle": float(base_cost),
                    "remap_refined_oracle": float(refined_cost),
                    "remap_gain_pct": float(
                        100.0 * (base_cost - refined_cost) / max(1e-12, base_cost)
                    ),
                }
            )

    qmactr_cost = float("inf")
    final_qmactr_c = circuit.copy()
    cand_cost_cache: Dict[Tuple, float] = {}
    eps_tie = 1e-12
    for cand in strict_unique:
        sig = _circuit_signature(cand)
        base_cost = float(qmactr_oracle_final(cand))
        refined_cost = remap_refined_cost_cache.get(sig)
        cand_cost = (
            float(min(base_cost, float(refined_cost)))
            if refined_cost is not None
            else base_cost
        )
        cand_cost_cache[sig] = float(cand_cost)
        if cand_cost + eps_tie < qmactr_cost:
            qmactr_cost = float(cand_cost)
            final_qmactr_c = cand.copy()
        elif abs(float(cand_cost) - float(qmactr_cost)) <= eps_tie:
            if _tie_key(cand) < _tie_key(final_qmactr_c):
                final_qmactr_c = cand.copy()

    equiv_ok, equiv_mode, equiv_detail = check_circuit_equivalence(
        circuit,
        final_qmactr_c,
        atol=1e-7,
        exact_max_qubits=10,
    )
    if not equiv_ok:

        final_qmactr_c = circuit.copy()
        orig_sig = _circuit_signature(circuit)
        if orig_sig in cand_cost_cache:
            qmactr_cost = float(cand_cost_cache[orig_sig])
        else:
            qmactr_cost = _infer_cost_ensemble(
                circuit,
                num_samples=max(1, int(final_samples_eff)),
                temps=sample_temps,
                base_seed=seed + 799991,
                ensemble=eval_seed_ensemble,
            )
        equiv_detail = f"{equiv_detail}; fallback_to_original"

    key = f"{_slug(benchmark_label)}_{_slug(topology_label)}"
    qasm_path = out_dir / f"qmactr_{key}.qasm"
    dump_qasm_circuit(final_qmactr_c, qasm_path)

    result = EvalResult(
        benchmark=benchmark_label,
        topology=topology_label,
        qmactr=float(qmactr_cost),
        equivalence_ok=bool(equiv_ok),
        equivalence_mode=str(equiv_mode),
        equivalence_detail=str(equiv_detail),
        qmactr_qasm_path=str(qasm_path),
        qmactr_off_same_model=float(qmactr_off_same_model),
        transform_gain_vs_off_same_model_pct=(
            0.0
            if abs(float(qmactr_off_same_model)) < 1e-12
            else 100.0
            * (float(qmactr_off_same_model) - float(qmactr_cost))
            / float(qmactr_off_same_model)
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    return result, map_hist, tr_hist


def evaluate_suite(
    circuits: List[str],
    topologies: List[str],
    out_dir: Path,
    seed: int = 0,
    mapping_episodes: int = 220,
    transform_episodes: int = 160,
    strict_oracle_samples: int = 0,
    strict_final_samples: int = 0,
    device: str = "auto",
    transform_restarts: int = 1,
    transform_max_steps: int | None = None,
    map_restarts_override: int | None = None,
    mapping_early_stop_enable: bool = True,
    mapping_early_stop_warmup_ratio: float = 0.2,
    mapping_early_stop_patience: int = 48,
    mapping_early_stop_rel_delta: float = 1e-3,
    mapping_early_stop_abs_delta: float = 1.0,
    mapping_early_stop_eval_interval: int = 4,
    transform_early_stop_enable: bool = True,
    transform_early_stop_warmup_ratio: float = 0.25,
    transform_early_stop_patience: int = 32,
    transform_early_stop_rel_delta: float = 5e-4,
    transform_early_stop_abs_delta: float = 1.0,
    transform_early_stop_eval_interval: int = 2,
    transform_enable: bool = True,
    transform_equiv_enable: bool = False,
    transform_equiv_lib: str = "",
    transform_equiv_max_matches: int = 16,
    transform_max_dag_nodes_override: int | None = None,
    transform_dag_context_override: int | None = None,
    transform_large_opportunity_boost: bool = True,
    transform_dense_reward_boost: bool = True,
    transform_reward_mode: str = "oracle",
    mapping_edge_aware: bool = True,
    mapping_qubit_adj_mode: str = "raw",
) -> List[EvalResult]:
    all_results: List[EvalResult] = []
    for b in circuits:
        for t in topologies:
            res, _, _ = evaluate_one(
                benchmark_name=b,
                topology_path=t,
                out_dir=out_dir,
                seed=seed,
                mapping_episodes=mapping_episodes,
                transform_episodes=transform_episodes,
                strict_oracle_samples=strict_oracle_samples,
                strict_final_samples=strict_final_samples,
                device=device,
                transform_restarts=transform_restarts,
                transform_max_steps=transform_max_steps,
                map_restarts_override=map_restarts_override,
                mapping_early_stop_enable=mapping_early_stop_enable,
                mapping_early_stop_warmup_ratio=mapping_early_stop_warmup_ratio,
                mapping_early_stop_patience=mapping_early_stop_patience,
                mapping_early_stop_rel_delta=mapping_early_stop_rel_delta,
                mapping_early_stop_abs_delta=mapping_early_stop_abs_delta,
                mapping_early_stop_eval_interval=mapping_early_stop_eval_interval,
                transform_early_stop_enable=transform_early_stop_enable,
                transform_early_stop_warmup_ratio=transform_early_stop_warmup_ratio,
                transform_early_stop_patience=transform_early_stop_patience,
                transform_early_stop_rel_delta=transform_early_stop_rel_delta,
                transform_early_stop_abs_delta=transform_early_stop_abs_delta,
                transform_early_stop_eval_interval=transform_early_stop_eval_interval,
                transform_enable=transform_enable,
                transform_equiv_enable=transform_equiv_enable,
                transform_equiv_lib=transform_equiv_lib,
                transform_equiv_max_matches=transform_equiv_max_matches,
                transform_max_dag_nodes_override=transform_max_dag_nodes_override,
                transform_dag_context_override=transform_dag_context_override,
                transform_large_opportunity_boost=transform_large_opportunity_boost,
                transform_dense_reward_boost=transform_dense_reward_boost,
                transform_reward_mode=transform_reward_mode,
                mapping_edge_aware=mapping_edge_aware,
                mapping_qubit_adj_mode=mapping_qubit_adj_mode,
            )
            all_results.append(res)

    table = [asdict(r) for r in all_results]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    return all_results

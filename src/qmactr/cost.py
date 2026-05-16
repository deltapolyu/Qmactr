from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .circuit import QuantumCircuit
from .topology import DQCTopology


def _invalidate_for_mutated_controller(
    active: Dict[Tuple[int, int, int], bool],
    mutated_qubit: int,
) -> None:
    dead = [k for k in active if k[0] == mutated_qubit]
    for k in dead:
        del active[k]


def logical_reuse_aware_cx_counts(
    circuit: QuantumCircuit,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(circuit.num_qubits)
    raw_dir = np.zeros((n, n), dtype=np.float32)
    effective_dir = np.zeros((n, n), dtype=np.float32)
    active: Dict[Tuple[int, int], bool] = {}

    for g in circuit.gates:
        if g.op == "cx" and g.q1 is not None:
            c = int(g.q0)
            t = int(g.q1)
            raw_dir[c, t] += 1.0
            key = (c, t)
            if key not in active:
                effective_dir[c, t] += 1.0
                active[key] = True

        if g.op == "cx" and g.q1 is not None:
            mutated = int(g.q1)
        else:
            mutated = int(g.q0)
        dead = [k for k in active if k[0] == mutated]
        for k in dead:
            del active[k]

    return raw_dir, effective_dir


def remote_entanglement_usage(
    circuit: QuantumCircuit,
    mapping: Dict[int, int],
) -> Dict[Tuple[int, int], int]:
    usage: Dict[Tuple[int, int], int] = {}
    active: Dict[Tuple[int, int, int], bool] = {}

    for g in circuit.gates:

        if g.op == "cx" and g.q1 is not None and g.q0 in mapping and g.q1 in mapping:
            from_qpu = mapping[g.q0]
            to_qpu = mapping[g.q1]
            if from_qpu != to_qpu:
                key = (g.q0, from_qpu, to_qpu)
                if key not in active:
                    pair = (
                        (from_qpu, to_qpu) if from_qpu < to_qpu else (to_qpu, from_qpu)
                    )
                    usage[pair] = usage.get(pair, 0) + 1
                    active[key] = True

        if g.op == "cx" and g.q1 is not None:
            _invalidate_for_mutated_controller(active, g.q1)
        else:
            _invalidate_for_mutated_controller(active, g.q0)

    return usage


def total_entanglement_cost(
    circuit: QuantumCircuit,
    mapping: Dict[int, int],
    topology: DQCTopology,
) -> float:
    usage = remote_entanglement_usage(circuit, mapping)
    total = 0.0
    for (a, b), cnt in usage.items():
        total += float(cnt) * topology.edge_cost(a, b)
    return total


def normalize_cost(cost: float, circuit: QuantumCircuit) -> float:
    denom = max(1.0, float(circuit.gate_count()))
    return cost / denom

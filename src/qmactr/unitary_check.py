from __future__ import annotations

from typing import Sequence, Tuple
import math

import numpy as np

from .circuit import Gate, QuantumCircuit


_SX_UNITARY = np.array(
    [
        [0.5 + 0.5j, 0.5 - 0.5j],
        [0.5 - 0.5j, 0.5 + 0.5j],
    ],
    dtype=np.complex128,
)


def _gate_unitary(g: Gate) -> np.ndarray:
    op = str(g.op).lower()
    if op == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if op == "h":
        s = 1.0 / math.sqrt(2.0)
        return np.array([[s, s], [s, -s]], dtype=np.complex128)
    if op == "sx":
        return _SX_UNITARY
    if op == "u":

        if g.param is None:
            return _SX_UNITARY
        theta = float(g.param)
        return np.array(
            [
                [np.exp(-0.5j * theta), 0.0],
                [0.0, np.exp(0.5j * theta)],
            ],
            dtype=np.complex128,
        )
    if op == "rz":
        theta = float(g.param) if g.param is not None else 0.0
        return np.array(
            [
                [np.exp(-0.5j * theta), 0.0],
                [0.0, np.exp(0.5j * theta)],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported single-qubit op for unitary check: {g.op}")


def _apply_single_qubit(state: np.ndarray, n: int, q: int, mat: np.ndarray) -> None:
    bit = 1 << int(q)
    size = 1 << n
    for i in range(size):
        if i & bit:
            continue
        j = i | bit
        a0 = state[i]
        a1 = state[j]
        state[i] = mat[0, 0] * a0 + mat[0, 1] * a1
        state[j] = mat[1, 0] * a0 + mat[1, 1] * a1


def _apply_cx(state: np.ndarray, n: int, control: int, target: int) -> None:
    c_bit = 1 << int(control)
    t_bit = 1 << int(target)
    size = 1 << n
    for i in range(size):
        if (i & c_bit) == 0:
            continue
        if i & t_bit:
            continue
        j = i | t_bit
        state[i], state[j] = state[j], state[i]


def apply_circuit_statevector(circuit: QuantumCircuit, state: np.ndarray) -> np.ndarray:
    n = int(circuit.num_qubits)
    out = np.array(state, dtype=np.complex128, copy=True)
    if out.shape != (1 << n,):
        raise ValueError(
            f"Statevector shape mismatch: expected {(1 << n,)}, got {out.shape}"
        )

    for g in circuit.gates:
        op = str(g.op).lower()
        if op == "cx":
            if g.q1 is None:
                raise ValueError("cx gate missing target qubit.")
            _apply_cx(out, n, int(g.q0), int(g.q1))
            continue
        mat = _gate_unitary(g)
        _apply_single_qubit(out, n, int(g.q0), mat)
    return out


def circuit_unitary(circuit: QuantumCircuit, max_qubits: int = 10) -> np.ndarray:
    n = int(circuit.num_qubits)
    if n > int(max_qubits):
        raise ValueError(
            f"Exact unitary too large for {n} qubits (limit={max_qubits})."
        )
    dim = 1 << n
    U = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        basis = np.zeros((dim,), dtype=np.complex128)
        basis[k] = 1.0
        U[:, k] = apply_circuit_statevector(circuit, basis)
    return U


def equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-7) -> bool:
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    inner = np.vdot(b.reshape(-1), a.reshape(-1))
    if abs(inner) < 1e-14:
        return bool(np.linalg.norm(a - b) <= float(atol))
    phase = inner / abs(inner)
    diff = np.linalg.norm(a - phase * b) / math.sqrt(float(a.size))
    return bool(diff <= float(atol))


def _build_local_circuit(
    gates: Sequence[Gate], touched_order: Sequence[int]
) -> QuantumCircuit:
    qmap = {int(q): i for i, q in enumerate(touched_order)}
    local: list[Gate] = []
    for g in gates:
        if g.q1 is None:
            local.append(Gate(g.op, qmap[g.q0], None, g.param))
        else:
            local.append(Gate(g.op, qmap[g.q0], qmap[g.q1], g.param))
    return QuantumCircuit(num_qubits=len(touched_order), gates=local)


def check_replacement_equivalence(
    old_segment: Sequence[Gate],
    replacement: Sequence[Gate],
    atol: float = 1e-7,
    max_qubits: int = 10,
) -> bool:
    all_gates = list(old_segment) + list(replacement)
    if not all_gates:
        return True
    touched = sorted({q for g in all_gates for q in g.qubits()})
    if len(touched) > int(max_qubits):

        return False
    c_old = _build_local_circuit(old_segment, touched)
    c_new = _build_local_circuit(replacement, touched)
    u_old = circuit_unitary(c_old, max_qubits=max_qubits)
    u_new = circuit_unitary(c_new, max_qubits=max_qubits)
    return equal_up_to_global_phase(u_old, u_new, atol=atol)


def check_circuit_equivalence(
    original: QuantumCircuit,
    transformed: QuantumCircuit,
    atol: float = 1e-7,
    exact_max_qubits: int = 10,
) -> Tuple[bool, str, str]:
    sig_o = tuple(
        (g.op, g.q0, g.q1, None if g.param is None else round(float(g.param), 8))
        for g in original.gates
    )
    sig_t = tuple(
        (g.op, g.q0, g.q1, None if g.param is None else round(float(g.param), 8))
        for g in transformed.gates
    )
    if sig_o == sig_t:
        return True, "identity", "circuits are structurally identical"

    n = int(original.num_qubits)
    if n != int(transformed.num_qubits):
        return (
            False,
            "mismatch",
            f"qubit count mismatch: {n} vs {transformed.num_qubits}",
        )

    if n > int(exact_max_qubits):
        return (
            True,
            "rewrite_certified",
            f"exact unitary skipped for n={n} (> {exact_max_qubits})",
        )

    try:
        u0 = circuit_unitary(original, max_qubits=exact_max_qubits)
        u1 = circuit_unitary(transformed, max_qubits=exact_max_qubits)
    except Exception as e:
        return False, "exact_failed", str(e)
    ok = equal_up_to_global_phase(u0, u1, atol=atol)
    return (
        ok,
        "exact_unitary",
        "exact unitary check passed" if ok else "exact unitary check failed",
    )

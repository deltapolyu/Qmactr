from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np


SINGLE_QUBIT_GATES = {"x", "h", "sx", "rz"}
TWO_QUBIT_GATES = {"cx"}


@dataclass(frozen=True)
class Gate:
    op: str
    q0: int
    q1: Optional[int] = None
    param: Optional[float] = None

    def qubits(self) -> Tuple[int, ...]:
        if self.q1 is None:
            return (self.q0,)
        return (self.q0, self.q1)

    def acts_on(self, qubit: int) -> bool:
        return qubit in self.qubits()

    def mutates(self, qubit: int) -> bool:

        if self.op == "cx":
            return self.q1 == qubit
        return self.q0 == qubit


@dataclass(frozen=True)
class Opportunity:
    kind: str
    index: int
    span: int = 1
    replacements: Tuple[Tuple["Gate", ...], ...] = field(default_factory=tuple)


class QuantumCircuit:
    def __init__(self, num_qubits: int, gates: Sequence[Gate]) -> None:
        self.num_qubits = num_qubits
        self.gates: List[Gate] = list(gates)

    def copy(self) -> "QuantumCircuit":
        return QuantumCircuit(self.num_qubits, list(self.gates))

    def gate_count(self) -> int:
        return len(self.gates)

    def cx_count(self) -> int:
        return sum(1 for g in self.gates if g.op == "cx")

    def op_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for g in self.gates:
            hist[g.op] = hist.get(g.op, 0) + 1
        return hist

    def interaction_matrix(self) -> np.ndarray:
        mat = np.zeros((self.num_qubits, self.num_qubits), dtype=np.float32)
        for g in self.gates:
            if g.op == "cx" and g.q1 is not None:
                mat[g.q0, g.q1] += 1.0
                mat[g.q1, g.q0] += 1.0

        np.fill_diagonal(mat, mat.diagonal() + 1.0)
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return mat / row_sum

    def dag_node_features(self) -> np.ndarray:

        n = len(self.gates)
        feat = np.zeros((n, 12), dtype=np.float32)
        if n == 0:
            return feat

        op_to_idx = {"x": 0, "h": 1, "sx": 2, "rz": 3, "cx": 4}
        q_den = max(1.0, float(self.num_qubits - 1))
        p_den = max(1.0, float(n - 1))

        for i, g in enumerate(self.gates):
            feat[i, op_to_idx.get(g.op, 5)] = 1.0
            feat[i, 6] = float(g.q0) / q_den
            feat[i, 7] = (float(g.q1) / q_den) if g.q1 is not None else -1.0
            feat[i, 8] = 1.0 if g.q1 is not None else 0.0
            if g.op == "rz" and g.param is not None:
                feat[i, 9] = math.sin(float(g.param))
                feat[i, 10] = math.cos(float(g.param))
            feat[i, 11] = float(i) / p_den
        return feat

    def dag_adjacency(self) -> np.ndarray:

        n = len(self.gates)
        adj = np.zeros((n, n), dtype=np.float32)
        if n == 0:
            return adj

        last_on_qubit: Dict[int, int] = {}
        for i, g in enumerate(self.gates):
            preds = set()
            for q in g.qubits():
                if q in last_on_qubit:
                    preds.add(last_on_qubit[q])
            for p in preds:
                adj[p, i] = 1.0
            for q in g.qubits():
                last_on_qubit[q] = i

        adj = adj + adj.T
        np.fill_diagonal(adj, adj.diagonal() + 1.0)
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return adj / row_sum

    def opportunities(
        self,
        equivalence_rules: Optional[object] = None,
        max_equiv_matches: int = 16,
    ) -> List[Opportunity]:

        mm = max(1, int(max_equiv_matches))
        gate_count = int(self.gate_count())
        cx_count = int(self.cx_count())
        large_circuit = bool(gate_count >= 800 or cx_count >= 300)
        eq_ops: List[Opportunity] = []
        find_budget = max(mm * 3, 24)
        if large_circuit:

            find_budget = max(mm * 5, 64)
        if equivalence_rules is not None and hasattr(
            equivalence_rules, "find_opportunities"
        ):
            try:

                found = equivalence_rules.find_opportunities(
                    self, max_matches=find_budget
                )
                eq_ops = list(found) if found else []
            except Exception:

                eq_ops = []

        swap_budget = mm if not eq_ops else max(2, mm // 2)
        if large_circuit:
            swap_budget = max(swap_budget, min(24, max(6, mm)))
        swap_ops = self._commutation_opportunities(max_matches=swap_budget)
        if not eq_ops and swap_ops:
            selected = self._layered_sample_opportunities(swap_ops, mm)
            return selected
        if not swap_ops:
            if not eq_ops:
                return []
            selected = self._layered_sample_opportunities(eq_ops, mm)
            return selected

        out: List[Opportunity] = []
        seen = set()
        for opp in list(eq_ops) + list(swap_ops):
            sig = (
                str(opp.kind),
                int(opp.index),
                int(opp.span),
                tuple(
                    tuple(
                        (
                            g.op,
                            g.q0,
                            g.q1,
                            None if g.param is None else round(float(g.param), 8),
                        )
                        for g in rep
                    )
                    for rep in opp.replacements
                ),
            )
            if sig in seen:
                continue
            seen.add(sig)
            out.append(opp)
        selected = self._layered_sample_opportunities(out, mm)
        return selected

    def _layered_sample_opportunities(
        self, opps: Sequence[Opportunity], limit: int
    ) -> List[Opportunity]:
        target = max(1, int(limit))
        if len(opps) <= target:
            return list(opps)

        n = max(1, int(len(self.gates)))
        num_segments = min(10, max(3, int(round(math.sqrt(float(target)) + 1.0))))
        num_q_regions = min(
            6, max(2, int(round(math.sqrt(float(max(1, self.num_qubits))))))
        )
        buckets: Dict[Tuple[str, int, int, int], List[Opportunity]] = {}

        for opp in opps:
            idx = max(0, min(n - 1, int(opp.index)))
            seg = min(num_segments - 1, int((idx * num_segments) / max(1, n)))
            span_bin = min(3, max(1, int(opp.span)))
            kind_bin = str(opp.kind)
            touched: List[int] = []
            i0 = max(0, int(opp.index))
            i1 = min(n, i0 + max(1, int(opp.span)))
            for g in self.gates[i0:i1]:
                touched.append(int(g.q0))
                if g.q1 is not None:
                    touched.append(int(g.q1))
            if touched:
                q_mid = float(sum(touched)) / float(len(touched))
                q_region = min(
                    num_q_regions - 1,
                    max(0, int((q_mid * num_q_regions) / max(1, self.num_qubits))),
                )
            else:
                q_region = 0
            bkey = (kind_bin, span_bin, seg, q_region)
            buckets.setdefault(bkey, []).append(opp)

        keys = sorted(
            buckets.keys(),
            key=lambda k: (k[2], k[3], 0 if k[0] == "library_rewrite" else 1, k[1]),
        )
        out: List[Opportunity] = []
        ptr: Dict[Tuple[str, int, int, int], int] = {k: 0 for k in keys}
        seen = set()
        while len(out) < target:
            progressed = False
            for k in keys:
                if len(out) >= target:
                    break
                p = ptr[k]
                bucket = buckets[k]
                if p >= len(bucket):
                    continue
                opp = bucket[p]
                ptr[k] = p + 1
                sig = (
                    str(opp.kind),
                    int(opp.index),
                    int(opp.span),
                    tuple(
                        tuple(
                            (
                                g.op,
                                g.q0,
                                g.q1,
                                None if g.param is None else round(float(g.param), 8),
                            )
                            for g in rep
                        )
                        for rep in opp.replacements
                    ),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(opp)
                progressed = True
            if not progressed:
                break

        if len(out) < target:
            for opp in opps:
                if len(out) >= target:
                    break
                sig = (
                    str(opp.kind),
                    int(opp.index),
                    int(opp.span),
                    tuple(
                        tuple(
                            (
                                g.op,
                                g.q0,
                                g.q1,
                                None if g.param is None else round(float(g.param), 8),
                            )
                            for g in rep
                        )
                        for rep in opp.replacements
                    ),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(opp)
        return out[:target]

    @staticmethod
    def _cx_cx_commute(g1: Gate, g2: Gate) -> bool:
        if g1.op != "cx" or g2.op != "cx" or g1.q1 is None or g2.q1 is None:
            return False
        a_c, a_t = int(g1.q0), int(g1.q1)
        b_c, b_t = int(g2.q0), int(g2.q1)

        if len({a_c, a_t, b_c, b_t}) == 4:
            return True
        if a_c == b_c and a_t != b_t:
            return True
        if a_t == b_t and a_c != b_c:
            return True
        return False

    @staticmethod
    def _single_cx_commute(single: Gate, cx: Gate) -> bool:
        if cx.op != "cx" or cx.q1 is None or single.q1 is not None:
            return False
        q = int(single.q0)
        c = int(cx.q0)
        t = int(cx.q1)

        if q != c and q != t:
            return True

        if single.op == "rz" and q == c:
            return True

        if single.op == "x" and q == t:
            return True
        return False

    @classmethod
    def _gates_commute(cls, g1: Gate, g2: Gate) -> bool:
        q1 = set(g1.qubits())
        q2 = set(g2.qubits())
        if q1.isdisjoint(q2):
            return True
        if g1.op == "cx" and g2.op == "cx":
            return cls._cx_cx_commute(g1, g2)
        if g1.q1 is None and g2.op == "cx":
            return cls._single_cx_commute(g1, g2)
        if g2.q1 is None and g1.op == "cx":
            return cls._single_cx_commute(g2, g1)

        if g1.q1 is None and g2.q1 is None and int(g1.q0) == int(g2.q0):
            return g1.op == "rz" and g2.op == "rz"
        return False

    def _commutation_opportunities(self, max_matches: int) -> List[Opportunity]:
        n = len(self.gates)
        if n < 2:
            return []
        candidates: List[Tuple[int, int, Opportunity]] = []
        for i in range(n - 1):
            g1 = self.gates[i]
            g2 = self.gates[i + 1]
            if not self._gates_commute(g1, g2):
                continue
            repl = ((g2, g1),)

            cx_score = int(g1.op == "cx") + int(g2.op == "cx")
            candidates.append(
                (
                    -cx_score,
                    i,
                    Opportunity(
                        kind="commute_swap", index=i, span=2, replacements=repl
                    ),
                )
            )
        if not candidates:
            return []
        candidates.sort(key=lambda x: (x[0], x[1]))
        out = [opp for _, _, opp in candidates[: max(1, int(max_matches))]]
        return out

    def opportunity_features(self, opp: Opportunity) -> np.ndarray:

        feat = np.zeros(14, dtype=np.float32)
        feat[0] = 1.0 if opp.kind == "library_rewrite" else 0.0

        i = opp.index
        span = max(1, int(opp.span))
        local = self.gates[max(0, i - 1) : min(len(self.gates), i + span + 1)]
        local_cx = sum(1 for g in local if g.op == "cx")
        feat[6] = float(local_cx)
        feat[7] = float(len(local))
        feat[8] = float(local_cx) / max(1.0, float(len(local)))

        n = len(self.gates)
        feat[9] = float(i) / max(1.0, float(n - 1))
        feat[10] = float(max(0, n - i - span)) / max(1.0, float(n))

        touched = set()
        if 0 <= i < n and span > 0:
            for j in range(i, min(n, i + span)):
                touched.update(self.gates[j].qubits())
        feat[11] = float(len(touched)) / max(1.0, float(self.num_qubits))

        suffix = self.gates[i : min(n, i + max(8, span + 2))] if 0 <= i < n else []
        suffix_cx = sum(1 for g in suffix if g.op == "cx")
        feat[12] = float(suffix_cx) / max(1.0, float(len(suffix)))
        feat[13] = float(span) / max(1.0, float(n))
        return feat

    def opportunity_node_indices(self, opp: Opportunity) -> np.ndarray:
        n = len(self.gates)
        i = max(0, min(int(opp.index), n))
        end = max(i, min(n, i + max(1, int(opp.span))))
        if end <= i:
            return np.zeros((0,), dtype=np.int64)
        return np.arange(i, end, dtype=np.int64)

    def apply(self, opp: Opportunity, action: int) -> "QuantumCircuit":

        new_c = self.copy()
        i = opp.index
        if i < 0 or i >= len(new_c.gates):
            return new_c

        if action < 0 or action >= len(opp.replacements):
            return new_c
        replacement = opp.replacements[action]

        span = max(1, int(opp.span))
        end = min(len(new_c.gates), i + span)
        if end <= i:
            return new_c
        new_c.gates[i:end] = list(replacement)

        new_span = max(1, int(len(replacement)))
        pad = 8
        left = max(0, i - pad)
        right = min(len(new_c.gates), i + max(span, new_span) + pad)
        new_c.simplify_window(left, right, atol=1e-10)
        return new_c

    def simplify_window(
        self, left: int, right: int, atol: float = 1e-10, max_passes: int = 16
    ) -> None:
        n = len(self.gates)
        if n < 2:
            return
        l = max(0, min(int(left), n))
        r = max(l, min(int(right), n))
        if r - l < 2:
            return

        passes = 0
        changed = True
        while changed and passes < max(1, int(max_passes)):
            changed = False
            passes += 1
            r = min(r, len(self.gates))
            i = l
            while i + 1 < r and i + 1 < len(self.gates):
                g1 = self.gates[i]
                g2 = self.gates[i + 1]

                if (
                    g1.q1 is None
                    and g2.q1 is None
                    and int(g1.q0) == int(g2.q0)
                    and g1.op == g2.op
                    and g1.op in {"h", "x"}
                ):
                    del self.gates[i : i + 2]
                    r -= 2
                    l = max(0, l - 1)
                    i = max(l, i - 1)
                    changed = True
                    continue

                if (
                    g1.op == "rz"
                    and g2.op == "rz"
                    and g1.q1 is None
                    and g2.q1 is None
                    and int(g1.q0) == int(g2.q0)
                ):
                    theta = float(g1.param or 0.0) + float(g2.param or 0.0)
                    if abs(theta) <= float(atol):
                        del self.gates[i : i + 2]
                        r -= 2
                    else:
                        self.gates[i : i + 2] = [Gate("rz", q0=int(g1.q0), param=theta)]
                        r -= 1
                    l = max(0, l - 1)
                    i = max(l, i - 1)
                    changed = True
                    continue

                if (
                    g1.op == "cx"
                    and g2.op == "cx"
                    and g1.q1 is not None
                    and g2.q1 is not None
                    and int(g1.q0) == int(g2.q0)
                    and int(g1.q1) == int(g2.q1)
                ):
                    del self.gates[i : i + 2]
                    r -= 2
                    l = max(0, l - 1)
                    i = max(l, i - 1)
                    changed = True
                    continue

                i += 1

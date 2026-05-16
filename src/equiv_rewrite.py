from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
import math

from .circuit import Gate, Opportunity, QuantumCircuit


_ARITH_OPS = {"add", "mul", "div", "neg"}
_SUPPORTED_OPS = {"x", "h", "sx", "rz", "cx"}


@dataclass(frozen=True)
class PatternGate:
    op: str
    qubits: Tuple[int, ...]
    param_token: Optional[str] = None


@dataclass(frozen=True)
class RewriteRule:
    lhs: Tuple[PatternGate, ...]
    rhs: Tuple[PatternGate, ...]
    lhs_len: int
    lhs_cx: int
    rhs_cx: int
    lhs_edges: Tuple[Tuple[int, int], ...]
    op_hist_sig: Tuple[int, int, int, int, int]


def _to_qubit_idx(wire: str) -> Optional[int]:
    if not isinstance(wire, str):
        return None
    wire_u = wire.upper()
    if not wire_u.startswith("Q"):
        return None
    try:
        return int(wire_u[1:])
    except ValueError:
        return None


def _to_param_token(wire: str) -> Optional[str]:
    if not isinstance(wire, str):
        return None
    wire_u = wire.upper()
    if not wire_u.startswith("P"):
        return None
    return wire_u


def _parse_quantum_gate(gate_info: Sequence) -> Optional[PatternGate]:
    if len(gate_info) < 3:
        return None
    op = str(gate_info[0]).lower()
    if op == "u":
        inputs = gate_info[2]
        qubit_wires = [w for w in inputs if _to_qubit_idx(w) is not None]
        qubits = tuple(_to_qubit_idx(w) for w in qubit_wires)
        if len(qubits) != 1:
            return None

        if any(_to_param_token(w) is not None for w in inputs):
            return None
        return PatternGate(op="sx", qubits=(int(qubits[0]),))
    if op in _ARITH_OPS or op not in _SUPPORTED_OPS:
        return None

    inputs = gate_info[2]
    qubit_wires = [w for w in inputs if _to_qubit_idx(w) is not None]
    qubits = tuple(_to_qubit_idx(w) for w in qubit_wires)

    if op == "cx":
        if len(qubits) != 2:
            return None
        return PatternGate(op="cx", qubits=(int(qubits[0]), int(qubits[1])))

    if op in {"x", "h", "sx"}:
        if len(qubits) != 1:
            return None
        return PatternGate(op=op, qubits=(int(qubits[0]),))

    if op == "rz":
        if len(qubits) != 1:
            return None
        param_tokens = [
            _to_param_token(w) for w in inputs if _to_param_token(w) is not None
        ]
        if len(param_tokens) != 1:
            return None
        return PatternGate(
            op="rz", qubits=(int(qubits[0]),), param_token=param_tokens[0]
        )

    return None


def _pattern_metric(pattern: Tuple[PatternGate, ...]) -> Tuple[int, int]:
    return (len(pattern), sum(1 for g in pattern if g.op == "cx"))


def _pattern_param_tokens(pattern: Tuple[PatternGate, ...]) -> Set[str]:
    out: Set[str] = set()
    for g in pattern:
        if g.param_token is not None:
            out.add(g.param_token)
    return out


def _gate_dependency_edges(
    gates: Sequence[PatternGate | Gate],
) -> Tuple[Tuple[int, int], ...]:
    def _qs(g: PatternGate | Gate) -> Tuple[int, ...]:
        if isinstance(g, PatternGate):
            return g.qubits
        return g.qubits()

    edges: List[Tuple[int, int]] = []
    last_on_qubit: Dict[int, int] = {}
    for i, g in enumerate(gates):
        preds = set()
        for q in _qs(g):
            if q in last_on_qubit:
                preds.add(last_on_qubit[q])
        for p in sorted(preds):
            edges.append((p, i))
        for q in _qs(g):
            last_on_qubit[q] = i
    return tuple(edges)


def _op_hist_sig(ops: Sequence[str]) -> Tuple[int, int, int, int, int]:

    return (
        sum(1 for op in ops if op == "x"),
        sum(1 for op in ops if op == "h"),
        sum(1 for op in ops if op == "sx"),
        sum(1 for op in ops if op == "rz"),
        sum(1 for op in ops if op == "cx"),
    )


class EquivalenceRuleSet:
    def __init__(self, rules: Iterable[RewriteRule]) -> None:
        self.rules: List[RewriteRule] = list(rules)
        self.rules_by_sig: DefaultDict[
            Tuple[int, Tuple[int, int, int, int, int]], List[RewriteRule]
        ] = defaultdict(list)
        self.rules_by_role_sig: DefaultDict[
            Tuple[
                int,
                Tuple[int, int, int, int, int],
                Tuple[Tuple[Tuple[str, int, int, int], int], ...],
            ],
            List[RewriteRule],
        ] = defaultdict(list)
        self.max_lhs_len = 0
        self.rule_lengths: List[int] = []
        self.rule_ids: Dict[RewriteRule, int] = {}
        self._rule_graph_meta: Dict[
            int,
            Tuple[
                Tuple[int, ...],
                Tuple[int, ...],
                Tuple[int, ...],
                Tuple[int, ...],
                Tuple[str, ...],
                Tuple[int, ...],
            ],
        ] = {}
        self._window_edge_cache: Dict[Tuple, Set[Tuple[int, int]]] = {}
        self._window_role_cache: Dict[
            Tuple, Tuple[Tuple[Tuple[str, int, int, int], int], ...]
        ] = {}
        self._rule_match_cache: Dict[
            Tuple[int, Tuple], Tuple[Tuple[Gate, ...], ...]
        ] = {}
        self._cache_cap = 200000
        for r in self.rules:
            rid = len(self.rule_ids)
            self.rule_ids[r] = rid
            self.rules_by_sig[(r.lhs_len, r.op_hist_sig)].append(r)
            p_out_masks, p_in_masks = self._edge_masks_from_edges(
                r.lhs_len, r.lhs_edges
            )
            p_in_deg = tuple(mask.bit_count() for mask in p_in_masks)
            p_out_deg = tuple(mask.bit_count() for mask in p_out_masks)
            p_ops = tuple(pg.op for pg in r.lhs)
            p_arity = tuple(len(pg.qubits) for pg in r.lhs)
            self._rule_graph_meta[rid] = (
                p_out_masks,
                p_in_masks,
                p_in_deg,
                p_out_deg,
                p_ops,
                p_arity,
            )
            role_sig = self._node_role_signature_from_rule(r, p_in_deg, p_out_deg)
            self.rules_by_role_sig[(r.lhs_len, r.op_hist_sig, role_sig)].append(r)
            self.max_lhs_len = max(self.max_lhs_len, r.lhs_len)
        self.rule_lengths = sorted({r.lhs_len for r in self.rules})

    @staticmethod
    def _segment_signature(segment: Sequence[Gate]) -> Tuple:
        return tuple(
            (g.op, g.q0, g.q1, None if g.param is None else round(float(g.param), 8))
            for g in segment
        )

    @staticmethod
    def _edge_masks_from_edges(
        n: int, edges: Sequence[Tuple[int, int]]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        out_masks = [0] * n
        in_masks = [0] * n
        for u, v in edges:
            out_masks[int(u)] |= 1 << int(v)
            in_masks[int(v)] |= 1 << int(u)
        return tuple(out_masks), tuple(in_masks)

    @staticmethod
    def _node_role_signature_from_rule(
        rule: RewriteRule,
        in_deg: Tuple[int, ...],
        out_deg: Tuple[int, ...],
    ) -> Tuple[Tuple[Tuple[str, int, int, int], int], ...]:
        hist: Dict[Tuple[str, int, int, int], int] = {}
        for i, pg in enumerate(rule.lhs):
            key = (pg.op, len(pg.qubits), int(in_deg[i]), int(out_deg[i]))
            hist[key] = hist.get(key, 0) + 1
        return tuple(sorted(hist.items()))

    @staticmethod
    def _node_role_signature_from_window(
        window: Sequence[Gate],
        in_deg: Tuple[int, ...],
        out_deg: Tuple[int, ...],
    ) -> Tuple[Tuple[Tuple[str, int, int, int], int], ...]:
        hist: Dict[Tuple[str, int, int, int], int] = {}
        for i, g in enumerate(window):
            key = (g.op, len(g.qubits()), int(in_deg[i]), int(out_deg[i]))
            hist[key] = hist.get(key, 0) + 1
        return tuple(sorted(hist.items()))

    @staticmethod
    def from_quartz_json(path: str | Path) -> "EquivalenceRuleSet":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], dict):
            raise ValueError(f"Invalid equivalence JSON format: {p}")
        ecc: Dict[str, List] = data[1]

        unique_rules: Dict[
            Tuple[Tuple[PatternGate, ...], Tuple[PatternGate, ...]], RewriteRule
        ] = {}
        for _, circuits_in_class in ecc.items():
            parsed_patterns: List[Tuple[PatternGate, ...]] = []
            for item in circuits_in_class:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                gate_seq = item[1]
                if not isinstance(gate_seq, list):
                    continue

                pattern: List[PatternGate] = []
                valid = True
                for g in gate_seq:
                    if not isinstance(g, list) or len(g) < 3:
                        valid = False
                        break
                    op = str(g[0]).lower()
                    if op in _ARITH_OPS:
                        valid = False
                        break
                    pg = _parse_quantum_gate(g)
                    if pg is None:
                        valid = False
                        break
                    pattern.append(pg)
                if valid and pattern:
                    parsed_patterns.append(tuple(pattern))

            if len(parsed_patterns) < 2:
                continue

            dedup_patterns = list(dict.fromkeys(parsed_patterns))
            metrics = [_pattern_metric(pat) for pat in dedup_patterns]

            for src_pat, src_metric in zip(dedup_patterns, metrics):
                src_tokens = _pattern_param_tokens(src_pat)
                for dst_pat, dst_metric in zip(dedup_patterns, metrics):
                    if src_pat == dst_pat:
                        continue

                    if dst_metric > src_metric:
                        continue

                    dst_tokens = _pattern_param_tokens(dst_pat)
                    if not dst_tokens.issubset(src_tokens):
                        continue

                    lhs_ops = tuple(g.op for g in src_pat)
                    lhs_cx = sum(1 for g in src_pat if g.op == "cx")
                    rhs_cx = sum(1 for g in dst_pat if g.op == "cx")
                    rule = RewriteRule(
                        lhs=src_pat,
                        rhs=dst_pat,
                        lhs_len=len(src_pat),
                        lhs_cx=lhs_cx,
                        rhs_cx=rhs_cx,
                        lhs_edges=_gate_dependency_edges(src_pat),
                        op_hist_sig=_op_hist_sig(lhs_ops),
                    )
                    unique_rules[(rule.lhs, rule.rhs)] = rule

        return EquivalenceRuleSet(rules=unique_rules.values())

    @staticmethod
    def _match_and_build(
        rule: RewriteRule, segment: Sequence[Gate]
    ) -> Optional[Tuple[Gate, ...]]:
        if len(segment) != rule.lhs_len:
            return None

        qmap: Dict[int, int] = {}
        qrev: Dict[int, int] = {}
        pmap: Dict[str, float] = {}

        for pg, g in zip(rule.lhs, segment):
            if pg.op != g.op:
                return None

            if pg.op in {"x", "h", "sx"}:
                if len(pg.qubits) != 1 or g.q1 is not None:
                    return None
                pq = pg.qubits[0]
                cq = g.q0
                if pq in qmap and qmap[pq] != cq:
                    return None
                if cq in qrev and qrev[cq] != pq:
                    return None
                qmap[pq] = cq
                qrev[cq] = pq
                continue

            if pg.op == "rz":
                if len(pg.qubits) != 1 or g.q1 is not None or g.param is None:
                    return None
                pq = pg.qubits[0]
                cq = g.q0
                if pq in qmap and qmap[pq] != cq:
                    return None
                if cq in qrev and qrev[cq] != pq:
                    return None
                qmap[pq] = cq
                qrev[cq] = pq

                tok = pg.param_token
                if tok is None:
                    return None
                val = float(g.param)
                if tok in pmap:
                    if not math.isclose(pmap[tok], val, rel_tol=1e-6, abs_tol=1e-8):
                        return None
                else:
                    pmap[tok] = val
                continue

            if pg.op == "cx":
                if len(pg.qubits) != 2 or g.q1 is None:
                    return None
                pairs = ((pg.qubits[0], g.q0), (pg.qubits[1], int(g.q1)))
                for pq, cq in pairs:
                    if pq in qmap and qmap[pq] != cq:
                        return None
                    if cq in qrev and qrev[cq] != pq:
                        return None
                    qmap[pq] = cq
                    qrev[cq] = pq
                continue

            return None

        replacement: List[Gate] = []
        for pg in rule.rhs:
            if pg.op in {"x", "h", "sx"}:
                if len(pg.qubits) != 1 or pg.qubits[0] not in qmap:
                    return None
                replacement.append(Gate(pg.op, qmap[pg.qubits[0]]))
                continue

            if pg.op == "rz":
                if (
                    len(pg.qubits) != 1
                    or pg.qubits[0] not in qmap
                    or pg.param_token is None
                ):
                    return None
                if pg.param_token not in pmap:
                    return None
                replacement.append(
                    Gate("rz", qmap[pg.qubits[0]], None, pmap[pg.param_token])
                )
                continue

            if pg.op == "cx":
                if len(pg.qubits) != 2:
                    return None
                if pg.qubits[0] not in qmap or pg.qubits[1] not in qmap:
                    return None
                replacement.append(Gate("cx", qmap[pg.qubits[0]], qmap[pg.qubits[1]]))
                continue

            return None

        return tuple(replacement)

    def _match_pattern_to_window(
        self,
        rule: RewriteRule,
        rule_id: int,
        window: Sequence[Gate],
        window_edges: Optional[Set[Tuple[int, int]]] = None,
        max_mappings: int = 4,
    ) -> List[Tuple[int, ...]]:
        n = len(window)
        if n != rule.lhs_len:
            return []

        rule_meta = self._rule_graph_meta.get(rule_id)
        if rule_meta is None:
            return []
        p_out_masks, _p_in_masks, p_in_deg, p_out_deg, p_ops, p_arity = rule_meta
        if window_edges is None:
            w_edges = set(_gate_dependency_edges(window))
        else:
            w_edges = window_edges
        if len(rule.lhs_edges) != len(w_edges):
            return []

        w_out_masks, w_in_masks = self._edge_masks_from_edges(n, tuple(w_edges))
        w_in_deg = tuple(mask.bit_count() for mask in w_in_masks)
        w_out_deg = tuple(mask.bit_count() for mask in w_out_masks)

        cand_by_p: List[List[int]] = [[] for _ in range(n)]
        for p_idx in range(n):
            cands: List[int] = []
            for w_idx, wg in enumerate(window):
                if wg.op != p_ops[p_idx]:
                    continue
                if len(wg.qubits()) != p_arity[p_idx]:
                    continue
                if w_in_deg[w_idx] != p_in_deg[p_idx]:
                    continue
                if w_out_deg[w_idx] != p_out_deg[p_idx]:
                    continue
                cands.append(w_idx)
            if not cands:
                return []
            cand_by_p[p_idx] = cands

        order = sorted(
            range(n),
            key=lambda p: (
                len(cand_by_p[p]),
                -(p_out_deg[p] + p_in_deg[p]),
            ),
        )

        out: List[Tuple[int, ...]] = []
        mapping = [-1] * n
        assigned_p: List[int] = []

        def _dfs(pos: int, used_w_mask: int) -> None:
            if len(out) >= max(1, int(max_mappings)):
                return
            if pos == n:
                out.append(tuple(mapping))
                return

            p = order[pos]
            for w in cand_by_p[p]:
                if ((used_w_mask >> w) & 1) != 0:
                    continue

                ok = True
                for p2 in assigned_p:
                    w2 = int(mapping[p2])
                    if ((p_out_masks[p2] >> p) & 1) != ((w_out_masks[w2] >> w) & 1):
                        ok = False
                        break
                    if ((p_out_masks[p] >> p2) & 1) != ((w_out_masks[w] >> w2) & 1):
                        ok = False
                        break
                if not ok:
                    continue

                mapping[p] = w
                assigned_p.append(p)
                _dfs(pos + 1, used_w_mask | (1 << w))
                assigned_p.pop()
                mapping[p] = -1

        _dfs(0, 0)
        return out

    def find_opportunities(
        self, circuit: QuantumCircuit, max_matches: int = 16
    ) -> List[Opportunity]:
        if self.max_lhs_len < 2 or not self.rules:
            return []

        out: List[
            Tuple[Tuple[int, int, int, int, int], Opportunity, Tuple[int, ...]]
        ] = []
        n = len(circuit.gates)
        if n < 2:
            return []

        mm = max(1, int(max_matches))
        max_keep = max(mm * 12, 96)
        if mm >= 8:
            mapping_budget = 6
        elif mm >= 5:
            mapping_budget = 5
        else:
            mapping_budget = 4
        gates = circuit.gates
        pref_x = [0] * (n + 1)
        pref_h = [0] * (n + 1)
        pref_sx = [0] * (n + 1)
        pref_rz = [0] * (n + 1)
        pref_cx = [0] * (n + 1)
        for i, g in enumerate(gates, start=1):
            pref_x[i] = pref_x[i - 1] + (1 if g.op == "x" else 0)
            pref_h[i] = pref_h[i - 1] + (1 if g.op == "h" else 0)
            pref_sx[i] = pref_sx[i - 1] + (1 if g.op == "sx" else 0)
            pref_rz[i] = pref_rz[i - 1] + (1 if g.op == "rz" else 0)
            pref_cx[i] = pref_cx[i - 1] + (1 if g.op == "cx" else 0)

        def _touch_qubits(seg_gates: Sequence[Gate]) -> Tuple[int, ...]:
            qs: Set[int] = set()
            for gg in seg_gates:
                qs.update(int(q) for q in gg.qubits())
            return tuple(sorted(qs))

        def _cx_pressure(i0: int, j0: int, touched: Tuple[int, ...]) -> Tuple[int, int]:
            if not touched:
                return (0, 0)
            touched_set = set(touched)
            left = max(0, int(i0) - 12)
            right = min(n, int(j0) + 24)
            future_right = min(n, int(j0) + 48)

            local_touch_cx = 0
            future_touch_cx = 0
            for k in range(left, right):
                gk = gates[k]
                if gk.op != "cx":
                    continue
                if any(int(q) in touched_set for q in gk.qubits()):
                    local_touch_cx += 1
            for k in range(j0, future_right):
                gk = gates[k]
                if gk.op != "cx":
                    continue
                if any(int(q) in touched_set for q in gk.qubits()):
                    future_touch_cx += 1
            return (local_touch_cx, future_touch_cx)

        for win_len in self.rule_lengths:
            if win_len < 2:
                continue
            if win_len > n:
                break
            for i in range(0, n - win_len + 1):
                j = i + win_len
                hist_sig = (
                    pref_x[j] - pref_x[i],
                    pref_h[j] - pref_h[i],
                    pref_sx[j] - pref_sx[i],
                    pref_rz[j] - pref_rz[i],
                    pref_cx[j] - pref_cx[i],
                )
                broad_key = (win_len, hist_sig)
                if broad_key not in self.rules_by_sig:
                    continue

                seg = gates[i:j]
                seg_sig = self._segment_signature(seg)
                seg_edges = self._window_edge_cache.get(seg_sig)
                if seg_edges is None:
                    seg_edges = set(_gate_dependency_edges(seg))
                    if len(self._window_edge_cache) >= self._cache_cap:
                        self._window_edge_cache.clear()
                    self._window_edge_cache[seg_sig] = seg_edges

                role_sig = self._window_role_cache.get(seg_sig)
                if role_sig is None:
                    w_out_masks, w_in_masks = self._edge_masks_from_edges(
                        win_len, tuple(seg_edges)
                    )
                    role_sig = self._node_role_signature_from_window(
                        seg,
                        tuple(mask.bit_count() for mask in w_in_masks),
                        tuple(mask.bit_count() for mask in w_out_masks),
                    )
                    if len(self._window_role_cache) >= self._cache_cap:
                        self._window_role_cache.clear()
                    self._window_role_cache[seg_sig] = role_sig

                candidates = self.rules_by_role_sig.get((win_len, hist_sig, role_sig))
                if not candidates:
                    continue

                replacements: List[Tuple[Gate, ...]] = []
                seg_tuple = tuple(seg)
                for rule in candidates:
                    rid = self.rule_ids.get(rule, -1)
                    cache_key = (rid, seg_sig)
                    cached_rep = self._rule_match_cache.get(cache_key)
                    if cached_rep is None:
                        mappings = self._match_pattern_to_window(
                            rule,
                            rid,
                            seg,
                            window_edges=seg_edges,
                            max_mappings=mapping_budget,
                        )
                        built: List[Tuple[Gate, ...]] = []
                        if mappings:
                            for m in mappings:
                                aligned = [
                                    seg[m[p_idx]] for p_idx in range(rule.lhs_len)
                                ]
                                replacement = self._match_and_build(rule, aligned)
                                if replacement is None:
                                    continue
                                if seg_tuple == replacement:
                                    continue
                                built.append(replacement)
                        cached_rep = tuple(built)
                        if len(self._rule_match_cache) >= self._cache_cap:
                            self._rule_match_cache.clear()
                        self._rule_match_cache[cache_key] = cached_rep
                    if cached_rep:
                        replacements.extend(cached_rep)
                if not replacements:
                    continue

                dedup: List[Tuple[Gate, ...]] = []
                seen = set()
                for r in replacements:
                    key = tuple(
                        (
                            g.op,
                            g.q0,
                            g.q1,
                            None if g.param is None else round(float(g.param), 8),
                        )
                        for g in r
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    dedup.append(r)
                dedup.sort(key=lambda r: (len(r), sum(1 for g in r if g.op == "cx")))
                seg_cx = sum(1 for g in seg if g.op == "cx")
                best_len_delta = min(len(r) - win_len for r in dedup)
                best_cx_delta = min(
                    sum(1 for g in r if g.op == "cx") - seg_cx for r in dedup
                )
                touched = _touch_qubits(seg)
                local_touch_cx, future_touch_cx = _cx_pressure(i, j, touched)
                opp = Opportunity(
                    kind="library_rewrite",
                    index=i,
                    span=win_len,
                    replacements=tuple(dedup),
                )

                score = (
                    int(best_cx_delta),
                    int(best_len_delta),
                    -int(future_touch_cx),
                    -int(local_touch_cx),
                    int(i),
                )
                out.append((score, opp, touched))
                if len(out) > max_keep:
                    out.sort(key=lambda x: x[0])
                    out = out[:max_keep]

        if not out:
            return []
        out.sort(key=lambda x: x[0])

        target = max(1, int(max_matches))
        selected: List[Opportunity] = []
        used_exact: Set[Tuple[int, int]] = set()
        used_index: Set[int] = set()
        covered_qubits: Set[int] = set()
        for _, opp, touched in out:
            key = (int(opp.index), int(opp.span))
            if key in used_exact:
                continue
            if int(opp.index) in used_index:
                continue

            if touched and set(touched).issubset(covered_qubits):
                continue
            selected.append(opp)
            used_exact.add(key)
            used_index.add(int(opp.index))
            covered_qubits.update(int(q) for q in touched)
            if len(selected) >= target:
                return selected

        for _, opp, touched in out:
            if len(selected) >= target:
                break
            key = (int(opp.index), int(opp.span))
            if key in used_exact:
                continue
            if int(opp.index) in used_index:
                continue
            selected.append(opp)
            used_exact.add(key)
            used_index.add(int(opp.index))
            covered_qubits.update(int(q) for q in touched)

        for _, opp, _ in out:
            if len(selected) >= target:
                break
            key = (int(opp.index), int(opp.span))
            if key in used_exact:
                continue
            selected.append(opp)
            used_exact.add(key)
        return selected


@lru_cache(maxsize=4)
def load_equivalence_rules(path: str) -> EquivalenceRuleSet:
    return EquivalenceRuleSet.from_quartz_json(path)

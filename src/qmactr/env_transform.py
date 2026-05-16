from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .circuit import Gate, Opportunity, QuantumCircuit
from .unitary_check import check_replacement_equivalence


MappingOracleFn = Callable[[QuantumCircuit], float]


@dataclass
class TransformState:
    dag_node_features: np.ndarray
    dag_adj: np.ndarray
    group_features: np.ndarray
    group_node_indices: List[np.ndarray]
    action_features: List[np.ndarray]
    global_features: np.ndarray
    opportunities: List[Opportunity]


class TransformEnv:
    def __init__(
        self,
        circuit: QuantumCircuit,
        mapping_oracle: MappingOracleFn,
        max_steps: int = 80,
        equivalence_rules: Optional[object] = None,
        max_equiv_matches: int = 16,
        max_dag_nodes: int = 1024,
        dag_context: int = 24,
        verify_replacement_equivalence: bool = True,
        replacement_equiv_atol: float = 1e-7,
        replacement_equiv_max_qubits: int = 10,
        noop_penalty_start: float = 0.14,
        noop_penalty_end: float = 0.05,
        novelty_bonus_weight: float = 0.03,
        equal_cost_change_bonus: float = 0.01,
        oracle_rel_reward_weight: float = 1.0,
        oracle_abs_reward_weight: float = 0.0,
        cx_reduction_reward_weight: float = 0.20,
        gate_reduction_reward_weight: float = 0.05,
        new_best_bonus_weight: float = 0.10,
        worse_cost_penalty_weight: float = 0.02,
    ) -> None:
        self.original = circuit.copy()
        self.mapping_oracle = mapping_oracle
        self.max_steps = max_steps
        self.equivalence_rules = equivalence_rules
        self.max_equiv_matches = max(1, int(max_equiv_matches))
        self.max_dag_nodes = max(64, int(max_dag_nodes))
        self.dag_context = max(0, int(dag_context))
        self.verify_replacement_equivalence = bool(verify_replacement_equivalence)
        self.replacement_equiv_atol = float(replacement_equiv_atol)
        self.replacement_equiv_max_qubits = max(1, int(replacement_equiv_max_qubits))
        self.noop_penalty_start = float(noop_penalty_start)
        self.noop_penalty_end = float(noop_penalty_end)
        self.novelty_bonus_weight = float(novelty_bonus_weight)
        self.equal_cost_change_bonus = float(equal_cost_change_bonus)
        self.oracle_rel_reward_weight = float(oracle_rel_reward_weight)
        self.oracle_abs_reward_weight = float(oracle_abs_reward_weight)
        self.cx_reduction_reward_weight = float(cx_reduction_reward_weight)
        self.gate_reduction_reward_weight = float(gate_reduction_reward_weight)
        self.new_best_bonus_weight = float(new_best_bonus_weight)
        self.worse_cost_penalty_weight = float(worse_cost_penalty_weight)
        self._cache: Dict[Tuple, float] = {}
        self._equiv_cache: Dict[Tuple, bool] = {}

        self._state_struct_cache: Dict[
            Tuple,
            Tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                List[np.ndarray],
                List[np.ndarray],
                List[Opportunity],
            ],
        ] = {}
        self._full_opp_cache: Dict[Tuple, List[Opportunity]] = {}
        self._visit_count: Dict[Tuple, int] = {}
        self.current_sig: Tuple = tuple()
        self.reset()

    def _signature(self, c: QuantumCircuit) -> Tuple:
        return tuple(
            (g.op, g.q0, g.q1, None if g.param is None else round(float(g.param), 6))
            for g in c.gates
        )

    def _cost(self, c: QuantumCircuit) -> float:
        sig = self._signature(c)
        if sig not in self._cache:
            self._cache[sig] = float(self.mapping_oracle(c))
        return self._cache[sig]

    def _segment_signature(self, gates: Sequence[Gate]) -> Tuple:
        return tuple(
            (g.op, g.q0, g.q1, None if g.param is None else round(float(g.param), 6))
            for g in gates
        )

    def _replacement_equivalent(self, opp: Opportunity, action_idx: int) -> bool:
        span = max(1, int(opp.span))
        i = int(opp.index)
        if i < 0 or i >= len(self.current.gates):
            return False
        if action_idx < 0 or action_idx >= len(opp.replacements):
            return False
        end = min(len(self.current.gates), i + span)
        if end <= i:
            return False

        old_seg = self.current.gates[i:end]
        repl = list(opp.replacements[action_idx])
        key = (self._segment_signature(old_seg), self._segment_signature(repl))
        cached = self._equiv_cache.get(key)
        if cached is not None:
            return bool(cached)

        try:
            ok = bool(
                check_replacement_equivalence(
                    old_seg,
                    repl,
                    atol=self.replacement_equiv_atol,
                    max_qubits=self.replacement_equiv_max_qubits,
                )
            )
        except Exception:
            ok = False
        self._equiv_cache[key] = ok
        return ok

    def _global_features(self) -> np.ndarray:
        gate_cnt = self.current.gate_count()
        cx_cnt = self.current.cx_count()
        return np.array(
            [
                float(gate_cnt) / max(1.0, float(self.original.gate_count())),
                float(cx_cnt) / max(1.0, float(self.original.cx_count() or 1)),
                float(self.current_cost) / max(1.0, float(self.base_cost)),
                float(self.step_idx) / max(1.0, float(self.max_steps)),
            ],
            dtype=np.float32,
        )

    def _action_features(self, opp: Opportunity) -> np.ndarray:

        old_span = max(1, int(opp.span))
        if 0 <= opp.index < len(self.current.gates):
            end = min(len(self.current.gates), opp.index + old_span)
            old_seg = self.current.gates[opp.index : end]
        else:
            old_seg = []
        old_cx = sum(1 for g in old_seg if g.op == "cx")
        old_cx_den = max(1.0, float(old_cx))
        kind_code = 1.0 if opp.kind == "library_rewrite" else 0.0

        feats: List[np.ndarray] = []
        for repl in opp.replacements:
            new_len = len(repl)
            new_cx = sum(1 for g in repl if g.op == "cx")
            d_gate = float(new_len - old_span)
            d_cx = float(new_cx - old_cx)

            f = np.zeros(6, dtype=np.float32)
            f[0] = d_gate
            f[1] = d_cx
            f[2] = float(new_len) / max(1.0, float(old_span))
            f[3] = float(new_cx) / old_cx_den
            f[4] = d_gate + 0.5 * d_cx
            f[5] = kind_code
            feats.append(f)

        skip = np.zeros(6, dtype=np.float32)
        skip[2] = 1.0
        skip[3] = 1.0
        skip[5] = -1.0
        feats.append(skip)

        if not feats:
            return np.zeros((0, 6), dtype=np.float32)
        return np.stack(feats, axis=0)

    def _opportunity_signature(self, opp: Opportunity) -> Tuple:
        reps = tuple(
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
        )
        return (str(opp.kind), int(opp.index), int(opp.span), reps)

    def _opportunity_score(self, opp: Opportunity) -> Tuple[int, int, int]:
        i = int(opp.index)
        span = max(1, int(opp.span))
        n = len(self.current.gates)
        if i < 0 or i >= n:
            return (10**9, 10**9, i)
        old_seg = self.current.gates[i : min(n, i + span)]
        old_cx = sum(1 for g in old_seg if g.op == "cx")
        best_cx_delta = 10**6
        best_len_delta = 10**6
        for rep in opp.replacements:
            new_cx = sum(1 for g in rep if g.op == "cx")
            cx_delta = int(new_cx - old_cx)
            len_delta = int(len(rep) - span)
            if (cx_delta, len_delta) < (best_cx_delta, best_len_delta):
                best_cx_delta, best_len_delta = cx_delta, len_delta
        return (best_cx_delta, best_len_delta, i)

    def _rank_and_dedup_opportunities(
        self,
        opps: List[Opportunity],
        topk: int,
    ) -> List[Opportunity]:
        uniq: List[Opportunity] = []
        seen = set()
        n = len(self.current.gates)
        for opp in opps:
            i = int(opp.index)
            span = max(1, int(opp.span))
            if i < 0 or i >= n:
                continue
            if i + span <= i:
                continue
            sig = self._opportunity_signature(opp)
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(opp)
        uniq.sort(key=self._opportunity_score)
        target = max(1, int(topk))
        if len(uniq) <= target:
            return uniq
        return self._layered_topk_opportunities(uniq, target)

    def _layered_topk_opportunities(
        self, sorted_opps: List[Opportunity], topk: int
    ) -> List[Opportunity]:
        target = max(1, int(topk))
        if len(sorted_opps) <= target:
            return list(sorted_opps)

        n = max(1, int(len(self.current.gates)))

        num_segments = min(8, max(3, int(round(np.sqrt(float(target)) + 1.0))))
        buckets: List[List[Opportunity]] = [[] for _ in range(num_segments)]

        for opp in sorted_opps:
            idx = max(0, min(n - 1, int(opp.index)))
            seg = min(num_segments - 1, int((idx * num_segments) / max(1, n)))
            buckets[seg].append(opp)

        out: List[Opportunity] = []
        ptr = [0 for _ in range(num_segments)]
        while len(out) < target:
            progressed = False
            for b in range(num_segments):
                if len(out) >= target:
                    break
                p = ptr[b]
                if p < len(buckets[b]):
                    out.append(buckets[b][p])
                    ptr[b] = p + 1
                    progressed = True
            if not progressed:
                break

        if len(out) < target:
            seen = {self._opportunity_signature(x) for x in out}
            for opp in sorted_opps:
                if len(out) >= target:
                    break
                sig = self._opportunity_signature(opp)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(opp)
        return out[:target]

    def _compute_full_opportunities(self) -> List[Opportunity]:
        try:
            opps = self.current.opportunities(
                equivalence_rules=self.equivalence_rules,
                max_equiv_matches=self.max_equiv_matches,
            )
        except Exception:
            opps = []
        return list(opps) if opps else []

    def _incremental_update_opportunities(
        self,
        prev_opps: List[Opportunity],
        change_index: int,
        old_span: int,
        new_span: int,
        old_n: int,
    ) -> List[Opportunity]:

        if self.equivalence_rules is None:
            opps, _ = self._compute_full_opportunities()
            return opps

        new_n = len(self.current.gates)
        delta = int(new_span) - int(old_span)
        max_rule_len = max(2, int(getattr(self.equivalence_rules, "max_lhs_len", 2)))
        margin = max_rule_len + 2

        left_old = max(0, int(change_index) - margin)
        right_old = min(int(old_n), int(change_index) + max(1, int(old_span)) + margin)

        carried: List[Opportunity] = []
        for opp in prev_opps:
            s = int(opp.index)
            e = s + max(1, int(opp.span))
            if e <= left_old:
                carried.append(opp)
                continue
            if s >= right_old:
                ns = s + delta
                if 0 <= ns < new_n:
                    carried.append(
                        Opportunity(
                            kind=opp.kind,
                            index=ns,
                            span=opp.span,
                            replacements=opp.replacements,
                        )
                    )

        left_new = max(0, int(change_index) - margin - abs(delta))
        right_new = min(
            new_n, int(change_index) + max(1, int(new_span)) + margin + abs(delta)
        )
        rescanned: List[Opportunity] = []
        if right_new > left_new:
            local = QuantumCircuit(
                num_qubits=self.current.num_qubits,
                gates=self.current.gates[left_new:right_new],
            )
            local_budget = max(self.max_equiv_matches * 8, 64)
            try:
                local_opps = local.opportunities(
                    equivalence_rules=self.equivalence_rules,
                    max_equiv_matches=local_budget,
                )
            except Exception:
                local_opps = []
            for opp in local_opps:
                gi = int(opp.index) + left_new
                if 0 <= gi < new_n:
                    rescanned.append(
                        Opportunity(
                            kind=opp.kind,
                            index=gi,
                            span=opp.span,
                            replacements=opp.replacements,
                        )
                    )

        merged = carried + rescanned
        return self._rank_and_dedup_opportunities(merged, topk=self.max_equiv_matches)

    def _select_dag_indices(self, opps: List[Opportunity], n_nodes: int) -> np.ndarray:
        if n_nodes <= 0:
            return np.zeros((0,), dtype=np.int64)
        if n_nodes <= self.max_dag_nodes:
            return np.arange(n_nodes, dtype=np.int64)

        keep = np.zeros((n_nodes,), dtype=bool)
        centers: List[int] = []
        for opp in opps:
            start = max(0, int(opp.index) - self.dag_context)
            end = min(
                n_nodes, int(opp.index) + max(1, int(opp.span)) + self.dag_context
            )
            if end > start:
                keep[start:end] = True
                centers.append(int(opp.index) + max(0, int(opp.span) // 2))

        idx = np.flatnonzero(keep)
        if idx.size == 0:

            start = max(0, (n_nodes - self.max_dag_nodes) // 2)
            end = min(n_nodes, start + self.max_dag_nodes)
            return np.arange(start, end, dtype=np.int64)
        if idx.size <= self.max_dag_nodes:
            return idx.astype(np.int64)

        if not centers:
            return idx[: self.max_dag_nodes].astype(np.int64)
        centers_arr = np.array(centers, dtype=np.int64).reshape(1, -1)
        dist = np.min(np.abs(idx.reshape(-1, 1) - centers_arr), axis=1)
        keep_local = np.argsort(dist)[: self.max_dag_nodes]
        selected = np.sort(idx[keep_local]).astype(np.int64)
        return selected

    def _build_state_struct_from_opps(
        self,
        opps: List[Opportunity],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[np.ndarray],
        List[np.ndarray],
        List[Opportunity],
    ]:
        if not opps:
            return (
                np.zeros((0, 12), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 14), dtype=np.float32),
                [],
                [],
                [],
            )

        full_nodes = self.current.dag_node_features()
        full_adj = self.current.dag_adjacency()
        sel_idx = self._select_dag_indices(opps, int(full_nodes.shape[0]))
        if sel_idx.size > 0:
            dag_node_features = full_nodes[sel_idx]
            dag_adj = full_adj[np.ix_(sel_idx, sel_idx)].astype(np.float32, copy=True)
            row_sum = dag_adj.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            dag_adj = dag_adj / row_sum
            old_to_new = np.full((int(full_nodes.shape[0]),), -1, dtype=np.int64)
            old_to_new[sel_idx] = np.arange(sel_idx.shape[0], dtype=np.int64)
        else:
            dag_node_features = np.zeros(
                (0, full_nodes.shape[1] if full_nodes.ndim == 2 else 12),
                dtype=np.float32,
            )
            dag_adj = np.zeros((0, 0), dtype=np.float32)
            old_to_new = np.zeros((0,), dtype=np.int64)

        kept_opps: List[Opportunity] = []
        kept_group_nodes: List[np.ndarray] = []
        kept_group_feats: List[np.ndarray] = []
        kept_actions: List[np.ndarray] = []
        for opp in opps:
            raw_nodes = self.current.opportunity_node_indices(opp)
            if raw_nodes.size == 0:
                continue
            if old_to_new.size == 0:
                continue
            raw_nodes = raw_nodes[(raw_nodes >= 0) & (raw_nodes < old_to_new.shape[0])]
            if raw_nodes.size == 0:
                continue
            mapped = old_to_new[raw_nodes]
            mapped = mapped[mapped >= 0]
            if mapped.size == 0:
                continue
            kept_opps.append(opp)
            kept_group_nodes.append(np.array(mapped, dtype=np.int64))
            kept_group_feats.append(self.current.opportunity_features(opp))
            kept_actions.append(self._action_features(opp))

        if kept_group_feats:
            group_features = np.stack(kept_group_feats, axis=0)
        else:
            group_features = np.zeros((0, 14), dtype=np.float32)
        return (
            dag_node_features,
            dag_adj,
            group_features,
            kept_group_nodes,
            kept_actions,
            kept_opps,
        )

    def _build_state_struct(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[np.ndarray],
        List[np.ndarray],
        List[Opportunity],
    ]:
        full_opps = self._compute_full_opportunities()
        return self._build_state_struct_from_opps(full_opps)

    def state(self) -> TransformState:
        cache_key = self.current_sig
        cached = self._state_struct_cache.get(cache_key)
        if cached is None:
            full_opps = self._full_opp_cache.get(cache_key)
            if full_opps is None:
                full_opps = self._compute_full_opportunities()
                if len(self._full_opp_cache) >= 4096:
                    self._full_opp_cache.clear()
                self._full_opp_cache[cache_key] = full_opps
            cached = self._build_state_struct_from_opps(full_opps)

            if len(self._state_struct_cache) >= 4096:
                self._state_struct_cache.clear()
            self._state_struct_cache[cache_key] = cached
        (
            dag_node_features,
            dag_adj,
            group_features,
            group_nodes,
            action_features,
            opps,
        ) = cached

        return TransformState(
            dag_node_features=dag_node_features,
            dag_adj=dag_adj,
            group_features=group_features,
            group_node_indices=group_nodes,
            action_features=action_features,
            global_features=self._global_features(),
            opportunities=opps,
        )

    def reset(self) -> TransformState:
        self.current = self.original.copy()
        self.current_sig = self._signature(self.current)
        self.step_idx = 0
        self.base_cost = self._cost(self.current)
        self.current_cost = self.base_cost
        self.best_circuit = self.current.copy()
        self.best_cost = self.current_cost
        self._visit_count.clear()
        self._visit_count[self.current_sig] = 1
        return self.state()

    def step(
        self,
        group_idx: int,
        action_idx: int,
        state: Optional[TransformState] = None,
    ) -> Tuple[TransformState, float, bool, Dict[str, float]]:
        s = state if state is not None else self.state()
        if group_idx < 0 or group_idx >= len(s.opportunities):
            self.step_idx += 1
            next_state = self.state()
            done = (
                self.step_idx >= self.max_steps
                or next_state.group_features.shape[0] == 0
            )
            return (
                next_state,
                -0.2,
                done,
                {
                    "invalid": 1.0,
                    "noop": 0.0,
                    "oracle_relative_improvement_term": 0.0,
                    "oracle_absolute_improvement_term": 0.0,
                    "cx_reduction_shaping_term": 0.0,
                    "gate_reduction_shaping_term": 0.0,
                    "new_best_bonus_term": 0.0,
                    "worse_cost_penalty_term": 0.0,
                    "equal_cost_structural_bonus_term": 0.0,
                    "noop_penalty_term": 0.0,
                    "novelty_bonus_term": 0.0,
                },
            )

        opp = s.opportunities[group_idx]

        if action_idx < 0 or action_idx > len(opp.replacements):
            self.step_idx += 1
            next_state = self.state()
            done = (
                self.step_idx >= self.max_steps
                or next_state.group_features.shape[0] == 0
            )
            return (
                next_state,
                -0.2,
                done,
                {
                    "invalid": 1.0,
                    "noop": 0.0,
                    "oracle_relative_improvement_term": 0.0,
                    "oracle_absolute_improvement_term": 0.0,
                    "cx_reduction_shaping_term": 0.0,
                    "gate_reduction_shaping_term": 0.0,
                    "new_best_bonus_term": 0.0,
                    "worse_cost_penalty_term": 0.0,
                    "equal_cost_structural_bonus_term": 0.0,
                    "noop_penalty_term": 0.0,
                    "novelty_bonus_term": 0.0,
                },
            )

        if action_idx == len(opp.replacements):

            progress = float(self.step_idx) / max(1.0, float(self.max_steps))
            noop_penalty = self.noop_penalty_end + (
                self.noop_penalty_start - self.noop_penalty_end
            ) * (1.0 - progress)
            if len(s.opportunities) >= 8:
                noop_penalty *= 1.15
            noop_penalty = float(np.clip(noop_penalty, 0.02, 0.25))
            self.step_idx += 1
            next_state = self.state()
            done = (
                self.step_idx >= self.max_steps
                or next_state.group_features.shape[0] == 0
            )
            info = {
                "current_cost": float(self.current_cost),
                "best_cost": float(self.best_cost),
                "invalid": 0.0,
                "noop": 1.0,
                "oracle_relative_improvement_term": 0.0,
                "oracle_absolute_improvement_term": 0.0,
                "cx_reduction_shaping_term": 0.0,
                "gate_reduction_shaping_term": 0.0,
                "new_best_bonus_term": 0.0,
                "worse_cost_penalty_term": 0.0,
                "equal_cost_structural_bonus_term": 0.0,
                "noop_penalty_term": -float(noop_penalty),
                "novelty_bonus_term": 0.0,
            }
            return next_state, -float(noop_penalty), done, info

        if self.verify_replacement_equivalence and not self._replacement_equivalent(
            opp, action_idx
        ):
            self.step_idx += 1
            next_state = self.state()
            done = (
                self.step_idx >= self.max_steps
                or next_state.group_features.shape[0] == 0
            )
            info = {
                "current_cost": float(self.current_cost),
                "best_cost": float(self.best_cost),
                "invalid": 1.0,
                "noop": 0.0,
                "equiv_fail": 1.0,
                "oracle_relative_improvement_term": 0.0,
                "oracle_absolute_improvement_term": 0.0,
                "cx_reduction_shaping_term": 0.0,
                "gate_reduction_shaping_term": 0.0,
                "new_best_bonus_term": 0.0,
                "worse_cost_penalty_term": 0.0,
                "equal_cost_structural_bonus_term": 0.0,
                "noop_penalty_term": 0.0,
                "novelty_bonus_term": 0.0,
            }
            return next_state, -0.2, done, info

        new_c = self.current.apply(opp, action_idx)
        new_sig = self._signature(new_c)
        new_cost = self._cost(new_c)
        old_sig = self.current_sig
        old_n = len(self.current.gates)
        old_span = max(1, int(opp.span))
        new_span = (
            len(opp.replacements[action_idx])
            if 0 <= action_idx < len(opp.replacements)
            else old_span
        )

        prev_cost = float(self.current_cost)

        oracle_rel = (prev_cost - float(new_cost)) / max(1e-6, prev_cost)
        oracle_rel_term = float(self.oracle_rel_reward_weight) * float(oracle_rel)

        oracle_abs = (prev_cost - float(new_cost)) / max(1.0, float(self.base_cost))
        oracle_abs_term = float(self.oracle_abs_reward_weight) * float(oracle_abs)
        reward = float(oracle_rel_term + oracle_abs_term)

        prev_g = max(1.0, float(self.current.gate_count()))
        next_g = max(1.0, float(new_c.gate_count()))
        prev_cx = max(1.0, float(self.current.cx_count()))
        next_cx = max(1.0, float(new_c.cx_count()))
        cx_term = float(self.cx_reduction_reward_weight) * float(
            (prev_cx - next_cx) / prev_cx
        )
        gate_term = float(self.gate_reduction_reward_weight) * float(
            (prev_g - next_g) / prev_g
        )
        reward += cx_term
        reward += gate_term

        new_best_bonus_term = 0.0
        worse_cost_penalty_term = 0.0
        equal_cost_structural_bonus_term = 0.0
        if new_cost < self.best_cost:
            new_best_bonus_term = float(self.new_best_bonus_weight) * float(
                (self.best_cost - new_cost) / max(1e-6, self.best_cost)
            )
            reward += new_best_bonus_term
        elif new_cost > self.current_cost:
            worse_cost_penalty_term = -float(self.worse_cost_penalty_weight) * float(
                (new_cost - self.current_cost) / max(1e-6, self.current_cost)
            )
            reward += worse_cost_penalty_term
        elif new_sig != old_sig:

            equal_cost_structural_bonus_term = float(self.equal_cost_change_bonus)
            reward += equal_cost_structural_bonus_term
        reward = float(np.clip(reward, -1.0, 1.0))
        self.current = new_c
        self.current_sig = new_sig
        self.current_cost = new_cost
        self.step_idx += 1

        try:
            prev_full = self._full_opp_cache.get(old_sig)
            if prev_full is None:

                prev_full = list(s.opportunities)
            next_full = self._incremental_update_opportunities(
                prev_opps=prev_full,
                change_index=int(opp.index),
                old_span=old_span,
                new_span=new_span,
                old_n=old_n,
            )
            if len(self._full_opp_cache) >= 4096:
                self._full_opp_cache.clear()
            self._full_opp_cache[new_sig] = next_full
            cached_struct = self._build_state_struct_from_opps(next_full)
            if len(self._state_struct_cache) >= 4096:
                self._state_struct_cache.clear()
            self._state_struct_cache[new_sig] = cached_struct
        except Exception:

            self._state_struct_cache.pop(new_sig, None)
            self._full_opp_cache.pop(new_sig, None)

        if new_cost < self.best_cost:
            self.best_cost = new_cost
            self.best_circuit = new_c.copy()
        v = int(self._visit_count.get(new_sig, 0)) + 1
        self._visit_count[new_sig] = v
        novelty_bonus_term = float(self.novelty_bonus_weight) / float(np.sqrt(float(v)))
        reward += novelty_bonus_term

        next_state = self.state()
        done = (
            self.step_idx >= self.max_steps or next_state.group_features.shape[0] == 0
        )
        info = {
            "current_cost": float(self.current_cost),
            "best_cost": float(self.best_cost),
            "invalid": 0.0,
            "noop": 0.0,
            "oracle_relative_improvement_term": float(oracle_rel_term),
            "oracle_absolute_improvement_term": float(oracle_abs_term),
            "cx_reduction_shaping_term": float(cx_term),
            "gate_reduction_shaping_term": float(gate_term),
            "new_best_bonus_term": float(new_best_bonus_term),
            "worse_cost_penalty_term": float(worse_cost_penalty_term),
            "equal_cost_structural_bonus_term": float(equal_cost_structural_bonus_term),
            "noop_penalty_term": 0.0,
            "novelty_bonus_term": float(novelty_bonus_term),
        }
        return next_state, float(reward), done, info

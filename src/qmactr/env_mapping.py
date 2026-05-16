from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .circuit import QuantumCircuit
from .cost import (
    logical_reuse_aware_cx_counts,
    normalize_cost,
    total_entanglement_cost,
)
from .topology import DQCTopology, topology_feature_matrix


@dataclass
class MappingState:
    qpu_features: np.ndarray
    qpu_adj: np.ndarray
    qubit_features: np.ndarray
    qubit_adj: np.ndarray
    global_features: np.ndarray
    action_mask: np.ndarray
    pair_prior: np.ndarray


class MappingEnv:
    def __init__(
        self,
        circuit: QuantumCircuit,
        topology: DQCTopology,
        final_cost_weight: float = 1.0,
        proxy_reward_weight: float = 0.35,
        qubit_adj_mode: str = "raw",
    ) -> None:
        self.circuit = circuit
        self.topology = topology
        self.final_cost_weight = final_cost_weight
        self.proxy_reward_weight = proxy_reward_weight
        self.qubit_adj_mode = str(qubit_adj_mode)
        self.num_qpus = topology.num_qpus
        self.num_qubits = circuit.num_qubits

        if self.qubit_adj_mode not in {"raw", "reuse_effective"}:
            raise ValueError(f"Unsupported qubit_adj_mode: {self.qubit_adj_mode}")

        self._interaction = circuit.interaction_matrix()
        raw_dir, effective_dir = logical_reuse_aware_cx_counts(self.circuit)
        self._cx_dir_raw = raw_dir.astype(np.float32, copy=False)
        self._cx_dir_effective = effective_dir.astype(np.float32, copy=False)
        if self.qubit_adj_mode == "reuse_effective":
            dir_for_adj = self._cx_dir_effective
        else:
            dir_for_adj = self._cx_dir_raw
        self._cx_dir = dir_for_adj.copy()
        max_cx_dir = float(self._cx_dir.max()) if self._cx_dir.size > 0 else 0.0
        if max_cx_dir > 0:
            self._cx_dir = self._cx_dir / max_cx_dir
        self._qubit_adj = self._cx_dir + 0.5 * self._cx_dir.T
        np.fill_diagonal(self._qubit_adj, self._qubit_adj.diagonal() + 1.0)
        q_row = self._qubit_adj.sum(axis=1, keepdims=True)
        q_row[q_row == 0] = 1.0
        self._qubit_adj = self._qubit_adj / q_row
        self._pair_cost = topology.pair_cost_matrix()
        self._topo_adj = topology.adjacency_matrix()
        self._cap_arr = np.array(
            [self.topology.capacities[i] for i in range(self.num_qpus)], dtype=np.int32
        )
        self._total_capacity = max(1.0, float(self._cap_arr.sum()))
        self._max_cap = max(1.0, float(self._cap_arr.max()))
        self._base_qpu_features = topology_feature_matrix(
            self.topology,
            {i: int(self._cap_arr[i]) for i in range(self.num_qpus)},
        )
        self._qubit_deg = self._interaction.sum(axis=1)
        self._ctrl_deg = self._cx_dir.sum(axis=1)
        self._tgt_deg = self._cx_dir.sum(axis=0)
        self._dir_deg = self._ctrl_deg + self._tgt_deg
        self._max_deg = max(1.0, float(self._qubit_deg.max()))
        self._max_ctrl_deg = max(1.0, float(self._ctrl_deg.max()))
        self._max_tgt_deg = max(1.0, float(self._tgt_deg.max()))
        self._max_dir_deg = max(1.0, float(self._dir_deg.max()))
        self._cx_count = max(1, circuit.cx_count())
        self.reset()

    def reset(self) -> MappingState:
        self.mapping: Dict[int, int] = {}
        self.remaining = {i: int(self._cap_arr[i]) for i in range(self.num_qpus)}
        self._remaining_arr = self._cap_arr.copy()
        self._mapping_arr = np.full((self.num_qubits,), -1, dtype=np.int32)
        self._mapped = np.zeros((self.num_qubits,), dtype=bool)
        self._assigned_weight = np.zeros((self.num_qubits,), dtype=np.float32)
        self._assigned_count = 0
        self._proxy_sum = 0.0
        self.t = 0
        return self.state()

    def is_done(self) -> bool:
        return self._assigned_count == self.num_qubits

    def encode_action(self, qpu: int, qubit: int) -> int:
        return qpu * self.num_qubits + qubit

    def decode_action(self, action: int) -> Tuple[int, int]:
        qpu = action // self.num_qubits
        qubit = action % self.num_qubits
        return qpu, qubit

    def action_mask(self) -> np.ndarray:
        qpu_ok = (self._remaining_arr > 0).astype(np.float32).reshape(self.num_qpus, 1)
        qubit_free = (~self._mapped).astype(np.float32).reshape(1, self.num_qubits)
        return (qpu_ok * qubit_free).reshape(self.num_qpus * self.num_qubits)

    def _partial_proxy_cost(self, mapping: Dict[int, int]) -> float:

        if mapping is self.mapping:
            return float(self._proxy_sum) / float(self._cx_count)

        total = 0.0
        for g in self.circuit.gates:
            if g.op != "cx" or g.q1 is None:
                continue
            if g.q0 not in mapping or g.q1 not in mapping:
                continue
            a = mapping[g.q0]
            b = mapping[g.q1]
            if a == b:
                continue
            total += float(self._pair_cost[a, b])
        return float(total) / float(self._cx_count)

    def _pair_prior(self, mask: np.ndarray) -> np.ndarray:
        rem_ratio = self._remaining_arr.astype(np.float32) / np.maximum(
            1.0, self._cap_arr.astype(np.float32)
        )
        base = 0.05 * rem_ratio.reshape(self.num_qpus, 1)

        assigned_idx = np.flatnonzero(self._mapped)
        if assigned_idx.size > 0:
            assigned_qpu = self._mapping_arr[assigned_idx]

            w = self._cx_dir[:, assigned_idx] + 0.7 * self._cx_dir[assigned_idx, :].T

            inv_dist = 1.0 / (1.0 + self._pair_cost[:, assigned_qpu])

            score = inv_dist @ w.T
            score = score + base
        else:
            avg_dist = self._pair_cost.mean(axis=1).astype(np.float32)
            dir_ratio = (
                (self._dir_deg / self._max_dir_deg)
                .astype(np.float32)
                .reshape(1, self.num_qubits)
            )
            score = dir_ratio / (1.0 + avg_dist.reshape(self.num_qpus, 1))
            score = score + base

        prior = score.reshape(self.num_qpus * self.num_qubits).astype(
            np.float32, copy=False
        )
        prior = np.where(mask > 0, prior, 0.0).astype(np.float32, copy=False)

        valid = mask > 0
        if np.any(valid):
            vals = prior[valid]
            mu = float(vals.mean())
            sigma = float(vals.std())
            if sigma > 1e-6:
                prior[valid] = (vals - mu) / sigma
            else:
                prior[valid] = 0.0
        return prior

    def state(self) -> MappingState:
        mask = self.action_mask()
        qpu_feats = self._base_qpu_features.copy()
        qpu_feats[:, 1] = self._remaining_arr.astype(np.float32) / self._max_cap
        qpu_feats[:, 3] = (self._remaining_arr > 0).astype(np.float32)
        qubit_feats = np.zeros((self.num_qubits, 7), dtype=np.float32)
        qubit_feats[:, 0] = self._mapped.astype(np.float32)
        qubit_feats[:, 1] = self._qubit_deg / self._max_deg
        qubit_feats[:, 2] = -1.0
        mapped_idx = np.flatnonzero(self._mapped)
        if mapped_idx.size > 0:
            qubit_feats[mapped_idx, 2] = self._mapping_arr[mapped_idx].astype(
                np.float32
            ) / max(1.0, float(self.num_qpus - 1))
        qubit_feats[:, 3] = 1.0 - (
            float(self._assigned_count) / max(1.0, float(self.num_qubits))
        )
        if self._assigned_count > 0:
            qubit_feats[:, 4] = self._assigned_weight / np.maximum(
                1e-6, self._qubit_deg
            )
        qubit_feats[:, 5] = self._ctrl_deg / self._max_ctrl_deg
        qubit_feats[:, 6] = self._tgt_deg / self._max_tgt_deg

        global_feats = np.array(
            [
                float(self._assigned_count) / max(1.0, float(self.num_qubits)),
                float(self._remaining_arr.sum()) / self._total_capacity,
                float(self._proxy_sum) / float(self._cx_count),
            ],
            dtype=np.float32,
        )

        return MappingState(
            qpu_features=qpu_feats,
            qpu_adj=self._topo_adj,
            qubit_features=qubit_feats,
            qubit_adj=self._qubit_adj,
            global_features=global_feats,
            action_mask=mask,
            pair_prior=self._pair_prior(mask),
        )

    def step(self, action: int) -> Tuple[MappingState, float, bool, Dict[str, float]]:
        prev_proxy = float(self._proxy_sum) / float(self._cx_count)
        qpu, qubit = self.decode_action(action)
        if self._mapped[qubit] or self._remaining_arr[qpu] <= 0:

            return self.state(), -1.0, False, {"invalid": 1.0}

        assigned_idx = np.flatnonzero(self._mapped)
        assigned_qpu = (
            self._mapping_arr[assigned_idx]
            if assigned_idx.size > 0
            else np.zeros((0,), dtype=np.int32)
        )

        self.mapping[qubit] = qpu
        self._mapped[qubit] = True
        self._mapping_arr[qubit] = int(qpu)
        self.remaining[qpu] -= 1
        self._remaining_arr[qpu] -= 1
        self._assigned_count += 1
        self._assigned_weight += self._interaction[:, qubit]
        self.t += 1

        if assigned_idx.size > 0:
            dist = self._pair_cost[qpu, assigned_qpu]
            w_fwd = self._cx_dir[qubit, assigned_idx]
            w_rev = self._cx_dir[assigned_idx, qubit]
            reward = float(np.sum((w_fwd + 0.7 * w_rev) / (1.0 + dist)))

            out_counts = self._cx_dir_raw[qubit, assigned_idx]
            in_counts = self._cx_dir_raw[assigned_idx, qubit]
            delta_proxy = float(
                np.dot(out_counts, self._pair_cost[qpu, assigned_qpu])
                + np.dot(in_counts, self._pair_cost[assigned_qpu, qpu])
            )
            self._proxy_sum += delta_proxy
        else:
            reward = 0.0

        new_proxy = float(self._proxy_sum) / float(self._cx_count)
        reward += self.proxy_reward_weight * (prev_proxy - new_proxy)

        done = self.is_done()
        info: Dict[str, float] = {"invalid": 0.0}
        if done:
            final_cost = total_entanglement_cost(
                self.circuit, self.mapping, self.topology
            )
            norm_cost = normalize_cost(final_cost, self.circuit)
            reward -= self.final_cost_weight * norm_cost
            info["final_cost"] = float(final_cost)
            info["final_norm_cost"] = float(norm_cost)

        return self.state(), float(reward), done, info

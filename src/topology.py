from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json

import networkx as nx
import numpy as np

DIRECT_HOP_COST = 12.0
EXTRA_HOP_INCREMENT = 10.0


@dataclass
class DQCTopology:
    graph: nx.Graph
    capacities: Dict[int, int]
    name: str = ""

    @property
    def num_qpus(self) -> int:
        return self.graph.number_of_nodes()

    def adjacency_matrix(self) -> np.ndarray:
        n = self.num_qpus
        mat = np.zeros((n, n), dtype=np.float32)
        for i, j in self.graph.edges():
            mat[i, j] = DIRECT_HOP_COST
            mat[j, i] = DIRECT_HOP_COST

        return mat

    def edge_cost(self, qpu_a: int, qpu_b: int) -> float:
        if qpu_a == qpu_b:
            return 0.0
        hops = int(nx.shortest_path_length(self.graph, qpu_a, qpu_b))

        return float(DIRECT_HOP_COST + (hops - 1) * EXTRA_HOP_INCREMENT)

    def pair_cost_matrix(self) -> np.ndarray:
        n = self.num_qpus
        mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                mat[i, j] = self.edge_cost(i, j)
        return mat


def load_topology_file(path: str | Path) -> DQCTopology:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Topology file not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Topology file must contain a JSON object: {p}")

    capacities_raw = data.get("capacities")
    if not isinstance(capacities_raw, dict) or not capacities_raw:
        raise ValueError(f"Topology file must define non-empty 'capacities': {p}")
    capacities = {int(k): int(v) for k, v in capacities_raw.items()}

    num_qpus = int(data.get("num_qpus", max(capacities.keys()) + 1))
    if num_qpus <= 0:
        raise ValueError(f"'num_qpus' must be positive in topology file: {p}")

    graph = nx.Graph()
    graph.add_nodes_from(range(num_qpus))

    edges = data.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"Topology file must define non-empty 'edges': {p}")
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Each edge must be [u, v], got {edge!r} in {p}")
        u, v = int(edge[0]), int(edge[1])
        if u < 0 or v < 0 or u >= num_qpus or v >= num_qpus:
            raise ValueError(
                f"Edge {edge!r} references a QPU outside [0, {num_qpus - 1}] in {p}"
            )
        if u == v:
            raise ValueError(f"Self-loop edge {edge!r} is not allowed in {p}")
        graph.add_edge(u, v)

    for qpu in range(num_qpus):
        if qpu not in capacities:
            raise ValueError(f"Missing capacity for QPU {qpu} in topology file: {p}")
        if capacities[qpu] <= 0:
            raise ValueError(
                f"Capacity for QPU {qpu} must be positive in topology file: {p}"
            )

    if not nx.is_connected(graph):
        raise ValueError(f"Topology graph must be connected: {p}")

    name = str(data.get("name") or p.stem)
    return DQCTopology(graph=graph, capacities=capacities, name=name)


def topology_feature_matrix(
    topology: DQCTopology, remaining: Dict[int, int]
) -> np.ndarray:
    n = topology.num_qpus
    feats = np.zeros((n, 7), dtype=np.float32)
    degree = dict(topology.graph.degree())
    closeness = nx.closeness_centrality(topology.graph)
    betweenness = nx.betweenness_centrality(topology.graph, normalized=True)
    pair_cost = topology.pair_cost_matrix()
    cap_arr = np.array([topology.capacities[i] for i in range(n)], dtype=np.float32)
    rem_arr = np.array([remaining[i] for i in range(n)], dtype=np.float32)
    max_deg = max(1, max(degree.values()))
    max_cap = max(1.0, float(cap_arr.max()))
    max_close = max(1e-6, float(max(closeness.values()) if closeness else 1.0))
    max_between = max(1e-6, float(max(betweenness.values()) if betweenness else 1.0))

    for i in range(n):
        mean_pair = float(pair_cost[i].mean()) if n > 0 else 0.0
        inv_mean_pair = 1.0 / (1.0 + mean_pair)
        feats[i, 0] = cap_arr[i] / max_cap
        feats[i, 1] = rem_arr[i] / max_cap
        feats[i, 2] = degree[i] / float(max_deg)
        feats[i, 3] = 1.0 if rem_arr[i] > 0 else 0.0
        feats[i, 4] = float(closeness.get(i, 0.0)) / max_close
        feats[i, 5] = float(betweenness.get(i, 0.0)) / max_between
        feats[i, 6] = inv_mean_pair
    return feats

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGATLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, num_heads: int = 2, edge_aware: bool = True
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.edge_aware = bool(edge_aware)

        self.w = nn.Parameter(torch.randn(num_heads, in_dim, out_dim) * 0.1)
        self.a_src = nn.Parameter(torch.randn(num_heads, out_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(num_heads, out_dim) * 0.1)
        self.a_edge = nn.Parameter(torch.randn(num_heads) * 0.1)
        self.skip = nn.Linear(in_dim, num_heads * out_dim)

    def forward(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:

        single = x.dim() == 2
        if single:
            x_b = x.unsqueeze(0)
        else:
            x_b = x

        if edge_weight.dim() == 2:
            edge_b = edge_weight.unsqueeze(0).expand(x_b.shape[0], -1, -1)
        else:
            edge_b = edge_weight

        bsz, n, _ = x_b.shape
        mask = edge_b > 0
        eye = torch.eye(n, device=edge_b.device, dtype=torch.bool).unsqueeze(0)
        mask = mask | eye

        h = torch.einsum("bni,hio->bhno", x_b, self.w)
        src_score = torch.einsum("bhno,ho->bhn", h, self.a_src)
        dst_score = torch.einsum("bhno,ho->bhn", h, self.a_dst)
        edge_term = (
            self.a_edge.view(1, -1, 1, 1) * edge_b.unsqueeze(1)
            if self.edge_aware
            else 0.0
        )
        e = F.leaky_relu(
            src_score.unsqueeze(-1) + dst_score.unsqueeze(-2) + edge_term,
            negative_slope=0.2,
        )

        e = e.masked_fill(~mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out = torch.matmul(alpha, h)
        cat = out.permute(0, 2, 1, 3).reshape(bsz, n, self.num_heads * self.out_dim)
        y = F.relu(cat + self.skip(x_b))
        if single:
            return y.squeeze(0)
        return y


class MappingActorCritic(nn.Module):
    def __init__(
        self,
        qpu_feat_dim: int,
        qubit_feat_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        heads: int = 2,
        edge_aware: bool = True,
    ) -> None:
        super().__init__()
        self.qpu_gat1 = EdgeAwareGATLayer(
            qpu_feat_dim, hidden_dim // heads, num_heads=heads, edge_aware=edge_aware
        )
        self.qpu_gat2 = EdgeAwareGATLayer(
            hidden_dim, hidden_dim // heads, num_heads=heads, edge_aware=edge_aware
        )

        self.qubit_gat1 = EdgeAwareGATLayer(
            qubit_feat_dim, hidden_dim // heads, num_heads=heads, edge_aware=edge_aware
        )
        self.qubit_gat2 = EdgeAwareGATLayer(
            hidden_dim, hidden_dim // heads, num_heads=heads, edge_aware=edge_aware
        )
        self.qpu_norm = nn.LayerNorm(hidden_dim)
        self.qubit_norm = nn.LayerNorm(hidden_dim)

        pair_dim = hidden_dim * 2 + global_dim + 1
        qubit_dim = hidden_dim + global_dim
        qpu_dim = hidden_dim + global_dim
        self.qubit_actor = nn.Sequential(
            nn.Linear(qubit_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.qpu_actor = nn.Sequential(
            nn.Linear(qpu_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.conditional_actor = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pair_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.prior_scale = nn.Parameter(torch.tensor(0.75))

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_hier(
        self,
        qpu_features: torch.Tensor,
        qpu_adj: torch.Tensor,
        qubit_features: torch.Tensor,
        qubit_adj: torch.Tensor,
        global_features: torch.Tensor,
        pair_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if qpu_features.dim() == 3:

            hq = self.qpu_gat1(qpu_features, qpu_adj)
            hq = self.qpu_gat2(hq, qpu_adj)
            hq = self.qpu_norm(hq)

            hc = self.qubit_gat1(qubit_features, qubit_adj)
            hc = self.qubit_gat2(hc, qubit_adj)
            hc = self.qubit_norm(hc)

            bsz, n_qpu, hdim = hq.shape
            n_qubit = hc.shape[1]

            hq_exp = hq.unsqueeze(2).expand(bsz, n_qpu, n_qubit, hdim)
            hc_exp = hc.unsqueeze(1).expand(bsz, n_qpu, n_qubit, hdim)
            g_exp = global_features.view(bsz, 1, 1, -1).expand(
                bsz, n_qpu, n_qubit, global_features.shape[-1]
            )
            prior_flat = pair_prior.reshape(bsz, n_qpu * n_qubit)
            prior_exp = prior_flat.view(bsz, n_qpu, n_qubit, 1)
            pair = torch.cat([hq_exp, hc_exp, g_exp, prior_exp], dim=-1).reshape(
                bsz, n_qpu * n_qubit, -1
            )

            qubit_ctx = torch.cat(
                [
                    hc,
                    global_features.unsqueeze(1).expand(
                        bsz, n_qubit, global_features.shape[-1]
                    ),
                ],
                dim=-1,
            )
            qubit_logits = self.qubit_actor(qubit_ctx).squeeze(-1)

            qpu_ctx = torch.cat(
                [
                    hq,
                    global_features.unsqueeze(1).expand(
                        bsz, n_qpu, global_features.shape[-1]
                    ),
                ],
                dim=-1,
            )
            qpu_logits = self.qpu_actor(qpu_ctx).squeeze(-1)

            conditional_logits = (
                self.conditional_actor(pair).squeeze(-1).view(bsz, n_qpu, n_qubit)
            )
            compat_logits = (
                self.pair_bilinear(
                    hq_exp.reshape(bsz, n_qpu * n_qubit, -1),
                    hc_exp.reshape(bsz, n_qpu * n_qubit, -1),
                )
                .squeeze(-1)
                .view(bsz, n_qpu, n_qubit)
            )
            cond_logits = self.logit_scale * (
                conditional_logits + compat_logits + qpu_logits.unsqueeze(-1)
            ) + (self.prior_scale * prior_exp.squeeze(-1))
            pair_logits = cond_logits + qubit_logits.unsqueeze(1)

            pooled = torch.cat(
                [hq.mean(dim=1), hc.mean(dim=1), global_features], dim=-1
            )
            value = self.critic(pooled).squeeze(-1)
            return (
                pair_logits.view(bsz, n_qpu * n_qubit),
                qubit_logits,
                cond_logits,
                value,
            )

        hq = self.qpu_gat1(qpu_features, qpu_adj)
        hq = self.qpu_gat2(hq, qpu_adj)
        hq = self.qpu_norm(hq)

        hc = self.qubit_gat1(qubit_features, qubit_adj)
        hc = self.qubit_gat2(hc, qubit_adj)
        hc = self.qubit_norm(hc)

        n_qpu = hq.shape[0]
        n_qubit = hc.shape[0]

        hq_exp = hq.unsqueeze(1).expand(n_qpu, n_qubit, hq.shape[-1])
        hc_exp = hc.unsqueeze(0).expand(n_qpu, n_qubit, hc.shape[-1])
        g_exp = global_features.view(1, 1, -1).expand(
            n_qpu, n_qubit, global_features.shape[-1]
        )
        prior_exp = pair_prior.view(n_qpu, n_qubit, 1)
        pair = torch.cat([hq_exp, hc_exp, g_exp, prior_exp], dim=-1).reshape(
            n_qpu * n_qubit, -1
        )

        qubit_ctx = torch.cat(
            [
                hc,
                global_features.view(1, -1).expand(n_qubit, global_features.shape[-1]),
            ],
            dim=-1,
        )
        qubit_logits = self.qubit_actor(qubit_ctx).squeeze(-1)

        qpu_ctx = torch.cat(
            [hq, global_features.view(1, -1).expand(n_qpu, global_features.shape[-1])],
            dim=-1,
        )
        qpu_logits = self.qpu_actor(qpu_ctx).squeeze(-1)

        conditional_logits = (
            self.conditional_actor(pair).squeeze(-1).view(n_qpu, n_qubit)
        )
        compat_logits = (
            self.pair_bilinear(
                hq_exp.reshape(n_qpu * n_qubit, -1),
                hc_exp.reshape(n_qpu * n_qubit, -1),
            )
            .squeeze(-1)
            .view(n_qpu, n_qubit)
        )
        cond_logits = self.logit_scale * (
            conditional_logits + compat_logits + qpu_logits.unsqueeze(-1)
        ) + (self.prior_scale * prior_exp.squeeze(-1))
        pair_logits = cond_logits + qubit_logits.unsqueeze(0)

        pooled = torch.cat([hq.mean(dim=0), hc.mean(dim=0), global_features], dim=0)
        value = self.critic(pooled).squeeze(-1)
        return pair_logits.reshape(n_qpu * n_qubit), qubit_logits, cond_logits, value

    def forward(
        self,
        qpu_features: torch.Tensor,
        qpu_adj: torch.Tensor,
        qubit_features: torch.Tensor,
        qubit_adj: torch.Tensor,
        global_features: torch.Tensor,
        pair_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_logits, _, _, value = self.forward_hier(
            qpu_features,
            qpu_adj,
            qubit_features,
            qubit_adj,
            global_features,
            pair_prior,
        )
        return pair_logits, value


class TransformHierarchicalAgent(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        group_feat_dim: int,
        action_feat_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        gnn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.gnn_heads = max(1, int(gnn_heads))
        gnn_out_per_head = max(8, int(hidden_dim) // self.gnn_heads)
        self.node_embed_dim = self.gnn_heads * gnn_out_per_head

        self.gate_gnn1 = EdgeAwareGATLayer(
            node_feat_dim, gnn_out_per_head, num_heads=self.gnn_heads
        )
        self.gate_gnn2 = EdgeAwareGATLayer(
            self.node_embed_dim, gnn_out_per_head, num_heads=self.gnn_heads
        )
        self.node_norm = nn.LayerNorm(self.node_embed_dim)

        group_in = self.node_embed_dim * 2 + group_feat_dim + global_dim
        self.meta_critic = nn.Sequential(
            nn.Linear(group_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.group_actor = nn.Sequential(
            nn.Linear(group_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        action_in = self.node_embed_dim + action_feat_dim + global_dim
        self.intra_actor = nn.Sequential(
            nn.Linear(action_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(
        self, node_features: torch.Tensor, dag_adj: torch.Tensor
    ) -> torch.Tensor:
        if node_features.shape[0] == 0:
            return torch.zeros(
                (0, self.node_embed_dim),
                device=node_features.device,
                dtype=node_features.dtype,
            )
        h = self.gate_gnn1(node_features, dag_adj)
        h = self.gate_gnn2(h, dag_adj)
        return self.node_norm(h)

    def _group_embeddings(
        self, node_embed: torch.Tensor, group_nodes: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if node_embed.shape[0] == 0:
            global_embed = torch.zeros(
                (self.node_embed_dim,), device=node_embed.device, dtype=node_embed.dtype
            )
        else:
            global_embed = node_embed.mean(dim=0)

        if not group_nodes:
            return (
                torch.zeros(
                    (0, self.node_embed_dim),
                    device=node_embed.device,
                    dtype=node_embed.dtype,
                ),
                global_embed,
            )

        group_embed: List[torch.Tensor] = []
        n = int(node_embed.shape[0])
        max_idx = max(0, n - 1)
        for idx in group_nodes:
            if idx.numel() == 0 or n == 0:
                group_embed.append(global_embed)
                continue
            idx_safe = idx.clamp(0, max_idx)
            g_emb = node_embed.index_select(0, idx_safe).mean(dim=0)
            group_embed.append(g_emb)
        return torch.stack(group_embed, dim=0), global_embed

    def group_values_and_logits(
        self,
        node_features: torch.Tensor,
        dag_adj: torch.Tensor,
        group_nodes: List[torch.Tensor],
        group_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if group_features.shape[0] == 0:
            z = torch.zeros(
                (0,), device=group_features.device, dtype=group_features.dtype
            )
            zg = torch.zeros(
                (0, self.node_embed_dim),
                device=group_features.device,
                dtype=group_features.dtype,
            )
            return z, z, zg

        node_embed = self.encode_nodes(node_features, dag_adj)
        group_embed, global_node_embed = self._group_embeddings(node_embed, group_nodes)

        g = global_features.unsqueeze(0).expand(
            group_features.shape[0], global_features.shape[0]
        )
        gn = global_node_embed.unsqueeze(0).expand(
            group_features.shape[0], global_node_embed.shape[0]
        )
        x = torch.cat([group_embed, gn, group_features, g], dim=-1)
        values = self.meta_critic(x).squeeze(-1)
        logits = self.group_actor(x).squeeze(-1)
        return values, logits, group_embed

    def action_logits(
        self,
        selected_group_embed: torch.Tensor,
        action_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        if action_features.shape[0] == 0:
            return torch.zeros(
                (0,), device=action_features.device, dtype=action_features.dtype
            )
        g = global_features.unsqueeze(0).expand(
            action_features.shape[0], global_features.shape[0]
        )
        group = selected_group_embed.unsqueeze(0).expand(
            action_features.shape[0], selected_group_embed.shape[0]
        )
        x = torch.cat([group, action_features, g], dim=-1)
        return self.intra_actor(x).squeeze(-1)

    def selected_logps_and_values_batched(
        self,
        node_features: torch.Tensor,
        dag_adj: torch.Tensor,
        group_features: torch.Tensor,
        group_valid_mask: torch.Tensor,
        group_node_indices: torch.Tensor,
        group_node_mask: torch.Tensor,
        global_features: torch.Tensor,
        selected_group_idx: torch.Tensor,
        action_features: torch.Tensor,
        action_valid_mask: torch.Tensor,
        selected_action_idx: torch.Tensor,
        value_aggregate_mode: str = "max",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_nodes, _ = node_features.shape
        _, n_groups, _ = group_features.shape
        _, n_actions, _ = action_features.shape

        node_embed = self.encode_nodes(node_features, dag_adj)
        hdim = int(node_embed.shape[-1])

        if n_nodes > 0 and n_groups > 0 and group_node_indices.shape[-1] > 0:
            idx = group_node_indices.clamp(0, max(0, n_nodes - 1))
            idx_exp = idx.unsqueeze(-1).expand(bsz, n_groups, idx.shape[-1], hdim)
            node_expand = node_embed.unsqueeze(1).expand(bsz, n_groups, n_nodes, hdim)
            gathered = torch.gather(node_expand, dim=2, index=idx_exp)
            gmask = group_node_mask.unsqueeze(-1).to(gathered.dtype)
            gsum = (gathered * gmask).sum(dim=2)
            gcount = gmask.sum(dim=2).clamp_min(1.0)
            group_embed = gsum / gcount
        else:
            group_embed = torch.zeros(
                (bsz, n_groups, hdim),
                device=node_features.device,
                dtype=node_features.dtype,
            )

        global_node_embed = (
            node_embed.mean(dim=1)
            if n_nodes > 0
            else torch.zeros(
                (bsz, hdim), device=node_features.device, dtype=node_features.dtype
            )
        )
        gn = global_node_embed.unsqueeze(1).expand(bsz, n_groups, hdim)
        gf = global_features.unsqueeze(1).expand(
            bsz, n_groups, global_features.shape[-1]
        )
        gx = torch.cat([group_embed, gn, group_features, gf], dim=-1)

        values = self.meta_critic(gx).squeeze(-1)
        group_logits = self.group_actor(gx).squeeze(-1)
        neg_inf = torch.finfo(group_logits.dtype).min
        group_logits = group_logits.masked_fill(~group_valid_mask, neg_inf)
        group_logp_all = torch.log_softmax(group_logits, dim=-1)
        group_probs = torch.softmax(group_logits, dim=-1)
        group_entropy = -(group_probs * group_logp_all).sum(dim=-1)
        new_logp_group = group_logp_all.gather(
            1, selected_group_idx.view(-1, 1)
        ).squeeze(1)

        if value_aggregate_mode == "expected":
            state_value = (group_probs * values).sum(dim=-1)
        else:
            val_masked = values.masked_fill(~group_valid_mask, neg_inf)
            state_value = torch.max(val_masked, dim=-1).values
            state_value = torch.where(
                torch.isfinite(state_value),
                state_value,
                torch.zeros_like(state_value),
            )

        batch_idx = torch.arange(bsz, device=node_features.device, dtype=torch.long)
        selected_group_embed = group_embed[batch_idx, selected_group_idx]
        sg = selected_group_embed.unsqueeze(1).expand(bsz, n_actions, hdim)
        agf = global_features.unsqueeze(1).expand(
            bsz, n_actions, global_features.shape[-1]
        )
        ax = torch.cat([sg, action_features, agf], dim=-1)
        action_logits = self.intra_actor(ax).squeeze(-1)
        action_logits = action_logits.masked_fill(~action_valid_mask, neg_inf)
        action_logp_all = torch.log_softmax(action_logits, dim=-1)
        action_probs = torch.softmax(action_logits, dim=-1)
        action_entropy = -(action_probs * action_logp_all).sum(dim=-1)
        new_logp_action = action_logp_all.gather(
            1, selected_action_idx.view(-1, 1)
        ).squeeze(1)

        total_entropy = group_entropy + action_entropy
        return new_logp_group, new_logp_action, state_value, total_entropy

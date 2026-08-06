from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


@dataclass
class DWSOutput:
    correction: torch.Tensor
    direction: torch.Tensor
    scale: torch.Tensor
    confidence: torch.Tensor


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeStateDWS(nn.Module):
    """Permutation-equivariant edge-state Deep Weight Space model.

    Weight matrices use the convention W[l][out, in]. Hidden-layer neuron
    permutations act on rows of W[l] and columns of W[l+1]. The model pools
    over neuron axes only after equivariant edge/node message passing, so its
    low-dimensional output is invariant to every hidden-neuron permutation.

    The output is a residual correction in a frozen D-dimensional functional
    basis, plus an explicit nonnegative scale and confidence.
    """

    def __init__(
        self,
        depth: int,
        label_dim: int,
        node_obs_dim: int = 0,
        layer_obs_dim: int = 0,
        edge_channels: int = 8,
        node_channels: int = 12,
        token_channels: int = 48,
        passes: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        if token_channels % transformer_heads:
            raise ValueError("token_channels must be divisible by transformer_heads")
        self.depth = int(depth)
        self.label_dim = int(label_dim)
        self.edge_channels = int(edge_channels)
        self.node_channels = int(node_channels)
        self.passes = int(passes)
        self.node_obs_dim = int(node_obs_dim)
        self.layer_obs_dim = int(layer_obs_dim)

        self.edge_init = MLP(4, 24, edge_channels)
        self.node_obs_proj = nn.Linear(node_obs_dim, node_channels) if node_obs_dim else None
        self.node_seed = nn.Parameter(torch.zeros(node_channels))
        self.node_layer_embedding = nn.Embedding(depth + 1, node_channels)
        self.edge_layer_embedding = nn.Embedding(depth, edge_channels)

        self.node_update = MLP(
            node_channels + 2 * edge_channels + node_channels,
            4 * node_channels,
            node_channels,
        )
        self.node_norm = nn.LayerNorm(node_channels)
        self.edge_update = MLP(
            2 * edge_channels + 2 * node_channels,
            4 * edge_channels,
            edge_channels,
        )
        self.edge_norm = nn.LayerNorm(edge_channels)

        layer_token_in = 3 * edge_channels + 4 * node_channels + layer_obs_dim
        self.layer_token = MLP(layer_token_in, 2 * token_channels, token_channels)
        self.layer_position = nn.Parameter(torch.zeros(depth, token_channels))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_channels,
            nhead=transformer_heads,
            dim_feedforward=4 * token_channels,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        head_in = 2 * token_channels
        self.direction_head = MLP(head_in, 2 * token_channels, label_dim)
        self.scale_head = MLP(head_in, token_channels, 1)
        self.confidence_head = MLP(head_in, token_channels, 1)

    @staticmethod
    def _raw_edge_features(w: torch.Tensor) -> torch.Tensor:
        # Per-layer normalization is invariant to row/column permutations.
        dims = (-2, -1)
        mu = w.mean(dim=dims, keepdim=True)
        sd = w.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-8)
        z = (w - mu) / sd
        return torch.stack(
            [z, z.square() - 1.0, z.abs() - math.sqrt(2.0 / math.pi), torch.sign(z) * torch.sqrt(z.abs() + 1e-6)],
            dim=-1,
        )

    def forward(
        self,
        weights: torch.Tensor,
        node_observables: Optional[torch.Tensor] = None,
        layer_observables: Optional[torch.Tensor] = None,
    ) -> DWSOutput:
        if weights.ndim != 4:
            raise ValueError("weights must have shape [batch, depth, width, width]")
        b, depth, width_out, width_in = weights.shape
        if depth != self.depth or width_out != width_in:
            raise ValueError("unexpected depth or non-square width")

        # Compute and checkpoint each layer independently. This is algebraically
        # identical to the vectorized initialization, but avoids retaining all
        # width-256 MLP activations simultaneously during backward.
        edges = []
        for l in range(depth):
            raw_l = self._raw_edge_features(weights[:, l])
            init_l = checkpoint(self.edge_init, raw_l, use_reentrant=False)
            edges.append(init_l + self.edge_layer_embedding.weight[l])

        nodes = self.node_seed.view(1, 1, 1, -1).expand(b, depth + 1, width_out, -1)
        nodes = nodes + self.node_layer_embedding.weight.view(1, depth + 1, 1, -1)
        if self.node_obs_proj is not None:
            if node_observables is None:
                raise ValueError("node_observables required by configured node_obs_dim")
            if node_observables.shape[:3] != (b, depth + 1, width_out):
                raise ValueError("node_observables must have shape [batch, depth+1, width, features]")
            nodes = nodes + self.node_obs_proj(node_observables)
        elif node_observables is not None and node_observables.shape[-1] != 0:
            raise ValueError("model configured without node observables")

        for _ in range(self.passes):
            row = torch.stack([e.mean(dim=2) for e in edges], dim=1)  # [B,L,out,C]
            col = torch.stack([e.mean(dim=1) for e in edges], dim=1)  # [B,L,in,C]
            incoming = torch.zeros(b, depth + 1, width_out, self.edge_channels, device=weights.device, dtype=weights.dtype)
            outgoing = torch.zeros_like(incoming)
            incoming[:, 1:] = row
            outgoing[:, :-1] = col
            emb = self.node_layer_embedding.weight.view(1, depth + 1, 1, -1).expand(b, -1, width_out, -1)
            node_cat = torch.cat([nodes, incoming, outgoing, emb], dim=-1)
            dn = checkpoint(self.node_update, node_cat, use_reentrant=False)
            nodes = self.node_norm(nodes + dn)

            updated = []
            for l, e in enumerate(edges):
                src = nodes[:, l].unsqueeze(1).expand(-1, width_out, -1, -1)
                dst = nodes[:, l + 1].unsqueeze(2).expand(-1, -1, width_out, -1)
                glob = e.mean(dim=(1, 2), keepdim=True).expand_as(e)
                edge_cat = torch.cat([e, glob, dst, src], dim=-1)
                de = checkpoint(self.edge_update, edge_cat, use_reentrant=False)
                updated.append(self.edge_norm(e + de))
            edges = updated

        tokens = []
        if layer_observables is not None:
            if layer_observables.shape[:2] != (b, depth):
                raise ValueError("layer_observables must have shape [batch, depth, features]")
            if layer_observables.shape[-1] != self.layer_obs_dim:
                raise ValueError("layer_observables feature count mismatch")
        elif self.layer_obs_dim:
            raise ValueError("layer_observables required by configured layer_obs_dim")

        for l, e in enumerate(edges):
            e_mean = e.mean(dim=(1, 2))
            e_std = e.std(dim=(1, 2), unbiased=False)
            e_absmax = e.abs().amax(dim=(1, 2))
            src = nodes[:, l]
            dst = nodes[:, l + 1]
            parts = [e_mean, e_std, e_absmax, src.mean(1), src.std(1, unbiased=False), dst.mean(1), dst.std(1, unbiased=False)]
            if layer_observables is not None:
                parts.append(layer_observables[:, l])
            token_cat = torch.cat(parts, dim=-1)
            tokens.append(checkpoint(self.layer_token, token_cat, use_reentrant=False))
        tok = torch.stack(tokens, dim=1) + self.layer_position.view(1, depth, -1)
        tok = self.layer_encoder(tok)
        pooled = torch.cat([tok.mean(1), tok[:, -1]], dim=-1)

        direction = self.direction_head(pooled)
        scale = F.softplus(self.scale_head(pooled).squeeze(-1))
        confidence = torch.sigmoid(self.confidence_head(pooled).squeeze(-1))
        unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        correction = unit * (scale * confidence).unsqueeze(-1)
        return DWSOutput(correction=correction, direction=direction, scale=scale, confidence=confidence)


def permute_hidden_layers(
    weights: torch.Tensor,
    permutations: list[torch.Tensor],
    node_observables: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply hidden permutations; input and output permutations should be identity."""
    b, depth, width, _ = weights.shape
    if len(permutations) != depth + 1:
        raise ValueError("need depth+1 node-layer permutations")
    out = torch.empty_like(weights)
    for l in range(depth):
        p_src = permutations[l]
        p_dst = permutations[l + 1]
        out[:, l] = weights[:, l][:, p_dst][:, :, p_src]
    obs_out = None
    if node_observables is not None:
        obs_out = torch.empty_like(node_observables)
        for l, p in enumerate(permutations):
            obs_out[:, l] = node_observables[:, l][:, p]
    return out, obs_out

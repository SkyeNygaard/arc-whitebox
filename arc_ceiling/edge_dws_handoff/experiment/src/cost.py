from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostEstimate:
    multiply_adds: int
    flops: int
    effective_compute_B: float


def estimate_dws_flops(
    depth: int,
    width: int,
    edge_channels: int,
    node_channels: int,
    token_channels: int,
    passes: int,
    label_dim: int,
) -> int:
    # Conservative dense-operation accounting. One multiply-add = 2 FLOPs.
    e = depth * width * width
    n = (depth + 1) * width
    edge_init = e * (4 * 24 + 24 * 24 + 24 * edge_channels)
    per_pass_node = n * ((node_channels + 2 * edge_channels + node_channels) * (4 * node_channels) + (4 * node_channels) ** 2 + (4 * node_channels) * node_channels)
    per_pass_edge = e * ((2 * edge_channels + 2 * node_channels) * (4 * edge_channels) + (4 * edge_channels) ** 2 + (4 * edge_channels) * edge_channels)
    token = depth * ((3 * edge_channels + 4 * node_channels) * (2 * token_channels) + (2 * token_channels) ** 2 + (2 * token_channels) * token_channels)
    attention = 2 * depth * depth * token_channels + 8 * depth * token_channels * token_channels
    heads = 2 * token_channels * (2 * token_channels + token_channels + label_dim + 2)
    macs = int(edge_init + passes * (per_pass_node + per_pass_edge) + token + attention + heads)
    return 2 * macs


def effective_compute_b(baseline_b: float, anchor_extra_b: float, inference_flops: int, replay_extra_b: float = 0.0) -> float:
    return float(baseline_b + anchor_extra_b + replay_extra_b + inference_flops / 1e9)

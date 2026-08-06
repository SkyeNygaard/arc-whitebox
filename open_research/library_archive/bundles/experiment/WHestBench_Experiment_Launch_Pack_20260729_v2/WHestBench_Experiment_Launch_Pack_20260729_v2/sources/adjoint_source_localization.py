from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/data/arc_research/agent34/agent3_agent4_repro')
sys.path.insert(0, str(ROOT))
from sampling import full_real_kerdock_bases, haar_rotation, chi_mean  # noqa: E402
import agent34_screen_fast as a  # noqa: E402

D = 256
DEPTH = 32


def make_kerdock(rotation_seed: int, max_bases: int | None) -> torch.Tensor:
    bases = full_real_kerdock_bases(D)
    if max_bases is not None:
        bases = bases[:max_bases]
    q = haar_rotation(D, rotation_seed)
    radius = chi_mean(D)
    blocks = [torch.cat([b @ q, -(b @ q)], dim=0) * radius for b in bases]
    return torch.cat(blocks, dim=0).contiguous()


def forward_to_target(
    x: torch.Tensor,
    ws: list[torch.Tensor],
    target_layer: int,
    collect_gates: bool,
) -> tuple[torch.Tensor, list[np.ndarray] | None]:
    gates: list[np.ndarray] = []
    with torch.no_grad():
        for layer in range(target_layer + 1):
            h = x @ ws[layer]
            if collect_gates:
                gates.append((h > 0).double().mean(0).cpu().numpy())
            x = torch.relu(h)
    return x, gates if collect_gates else None


def c21_matrix(x: torch.Tensor) -> np.ndarray:
    xd = x.double()
    xc = xd - xd.mean(0)
    return ((xc * xc).T @ xc / len(xc)).cpu().numpy()


def build_backward_maps(
    ws: list[torch.Tensor], gates: list[np.ndarray], target_layer: int
) -> list[np.ndarray]:
    """B[l] maps centered activation at l to target centered activation, row convention."""
    B: list[np.ndarray] = [np.empty((0, 0)) for _ in range(target_layer + 1)]
    B[target_layer] = np.eye(D, dtype=np.float64)
    for layer in range(target_layer - 1, -1, -1):
        w_next = ws[layer + 1].double().cpu().numpy()
        A = w_next * gates[layer + 1][None, :]
        B[layer] = A @ B[layer + 1]
    return B


def contractions_by_layer(
    x0: torch.Tensor,
    ws: list[torch.Tensor],
    B: list[np.ndarray],
    U: np.ndarray,
    V: np.ndarray,
    target_layer: int,
) -> np.ndarray:
    """Return T[layer, direction] without forming a third-order tensor."""
    out = np.empty((target_layer + 1, U.shape[1]), dtype=np.float64)
    x = x0
    with torch.no_grad():
        for layer in range(target_layer + 1):
            x = torch.relu(x @ ws[layer])
            xd = x.double()
            xc = xd - xd.mean(0)
            Bt = torch.from_numpy(B[layer]).to(dtype=torch.float64)
            z = xc @ Bt
            Ut = torch.from_numpy(U).to(dtype=torch.float64)
            Vt = torch.from_numpy(V).to(dtype=torch.float64)
            quad = (z * z) @ Ut
            linear = z @ Vt
            out[layer] = (quad * linear).mean(0).cpu().numpy()
    return out


def summarize_direction(delta: np.ndarray, target_layer: int) -> dict:
    source = np.diff(delta)
    total = float(delta[-1])
    inherited = float(delta[0])
    eps = 1e-30
    suffix = {}
    for d in [1, 2, 3, 4, 6, 8, 12, 16, 24, target_layer]:
        d = min(d, target_layer)
        val = float(source[-d:].sum())
        suffix[str(d)] = {
            'value': val,
            'signed_fraction_of_final': val / (total + math.copysign(eps, total or 1.0)),
            'absolute_source_fraction': float(np.abs(source[-d:]).sum() / max(np.abs(source).sum(), eps)),
        }
    abs_sum = float(np.abs(source).sum())
    sq_sum = float(np.square(source).sum())
    return {
        'final_defect': total,
        'inherited_layer0_defect': inherited,
        'inherited_fraction': inherited / (total + math.copysign(eps, total or 1.0)),
        'source_sum': float(source.sum()),
        'telescoping_error': float(inherited + source.sum() - total),
        'source_l1_over_final_abs': abs_sum / max(abs(total), eps),
        'source_effective_layers': abs_sum * abs_sum / max(sq_sum, eps),
        'largest_source_layers': [
            {'transition_to_activation_layer': int(i + 1), 'value': float(source[i])}
            for i in np.argsort(np.abs(source))[::-1][:8]
        ],
        'suffix': suffix,
        'delta_by_activation_layer': delta.tolist(),
        'source_by_transition': source.tolist(),
    }


def run_one(
    network: int,
    xk: torch.Tensor,
    qmc_n: int,
    target_layer: int,
    qmc_seed_base: int,
) -> dict:
    t0 = time.time()
    ws = a.make_weights(51000 + network)
    xq0 = a.sobol_normal(qmc_n, qmc_seed_base + network)

    xq_target, _ = forward_to_target(xq0.clone(), ws, target_layer, False)
    xk_target, gates = forward_to_target(xk.clone(), ws, target_layer, True)
    assert gates is not None

    c_truth = c21_matrix(xq_target)
    c_kerdock = c21_matrix(xk_target)
    defect = c_truth - c_kerdock
    U, s, Vt = np.linalg.svd(defect, full_matrices=False)
    U4 = U[:, :4].copy()
    V4 = Vt[:4].T.copy()

    B = build_backward_maps(ws, gates, target_layer)
    tq = contractions_by_layer(xq0, ws, B, U4, V4, target_layer)
    tk = contractions_by_layer(xk, ws, B, U4, V4, target_layer)
    delta = tq - tk

    direct = np.array([U4[:, r] @ defect @ V4[:, r] for r in range(4)])
    direct_err = delta[-1] - direct
    dirs = [summarize_direction(delta[:, r], target_layer) for r in range(4)]

    # Aggregate the four positive singular components, matching the rank-4 defect captured.
    agg_delta = delta.sum(axis=1)
    aggregate = summarize_direction(agg_delta, target_layer)
    captured_fraction = float(s[:4].sum() / max(np.linalg.svd(defect, compute_uv=False).sum(), 1e-30))
    frob_captured = float(np.square(s[:4]).sum() / max(np.square(s).sum(), 1e-30))

    return {
        'network': network,
        'qmc_n': qmc_n,
        'kerdock_n': int(len(xk)),
        'target_layer_zero_based': target_layer,
        'top_singular_values': s[:16].tolist(),
        'rank4_nuclear_fraction': captured_fraction,
        'rank4_frobenius_fraction': frob_captured,
        'direct_terminal_contractions': direct.tolist(),
        'terminal_identity_errors': direct_err.tolist(),
        'directions': dirs,
        'aggregate_rank4': aggregate,
        'backward_map_spectra': [
            {
                'layer': i,
                'top_sv': float(np.linalg.svd(B[i], compute_uv=False)[0]),
                'effective_rank': float(
                    (np.square(np.linalg.svd(B[i], compute_uv=False)).sum() ** 2)
                    / max(np.power(np.linalg.svd(B[i], compute_uv=False), 4).sum(), 1e-30)
                ),
            }
            for i in [0, 4, 8, 12, 16, 20, 24, 28, target_layer]
            if i <= target_layer
        ],
        'runtime_seconds': time.time() - t0,
    }


def aggregate(records: list[dict]) -> dict:
    suffix_depths = ['1', '2', '3', '4', '6', '8', '12', '16', '24']
    out = {}
    for d in suffix_depths:
        vals = [r['aggregate_rank4']['suffix'][d]['signed_fraction_of_final'] for r in records if d in r['aggregate_rank4']['suffix']]
        absvals = [r['aggregate_rank4']['suffix'][d]['absolute_source_fraction'] for r in records if d in r['aggregate_rank4']['suffix']]
        out[d] = {
            'median_signed_fraction': float(np.median(vals)),
            'mean_signed_fraction': float(np.mean(vals)),
            'min_signed_fraction': float(np.min(vals)),
            'max_signed_fraction': float(np.max(vals)),
            'median_absolute_source_fraction': float(np.median(absvals)),
        }
    return {
        'networks': len(records),
        'suffix_summary': out,
        'median_inherited_fraction': float(np.median([r['aggregate_rank4']['inherited_fraction'] for r in records])),
        'median_source_l1_over_final_abs': float(np.median([r['aggregate_rank4']['source_l1_over_final_abs'] for r in records])),
        'median_source_effective_layers': float(np.median([r['aggregate_rank4']['source_effective_layers'] for r in records])),
        'median_rank4_frobenius_fraction': float(np.median([r['rank4_frobenius_fraction'] for r in records])),
        'max_terminal_identity_error': float(max(max(abs(x) for x in r['terminal_identity_errors']) for r in records)),
        'total_runtime_seconds': float(sum(r['runtime_seconds'] for r in records)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--networks', type=int, nargs='+', default=[0])
    p.add_argument('--qmc-n', type=int, default=4096)
    p.add_argument('--target-layer', type=int, default=29)
    p.add_argument('--rotation-seed', type=int, default=3)
    p.add_argument('--max-bases', type=int, default=17)
    p.add_argument('--qmc-seed-base', type=int, default=220000)
    p.add_argument('--out', type=Path, default=Path('/mnt/data/arc_research/adjoint_source_localization.json'))
    args = p.parse_args()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    max_bases = None if args.max_bases <= 0 else args.max_bases
    xk = make_kerdock(args.rotation_seed, max_bases)
    records = []
    for n in args.networks:
        rec = run_one(n, xk, args.qmc_n, args.target_layer, args.qmc_seed_base)
        records.append(rec)
        a4 = rec['aggregate_rank4']
        print(json.dumps({
            'network': n,
            'runtime': rec['runtime_seconds'],
            'rank4_frob': rec['rank4_frobenius_fraction'],
            'inherited': a4['inherited_fraction'],
            'suffix4': a4['suffix']['4']['signed_fraction_of_final'],
            'suffix8': a4['suffix']['8']['signed_fraction_of_final'],
            'suffix16': a4['suffix']['16']['signed_fraction_of_final'],
            'cancellation': a4['source_l1_over_final_abs'],
        }), flush=True)
    payload = {'config': vars(args) | {'out': str(args.out)}, 'records': records, 'summary': aggregate(records)}
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()

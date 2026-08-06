#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import radial_core as r

D = r.D
DEPTH = r.DEPTH
TARGET = r.TARGET
P = 32
ROTATIONS_CACHE: dict[int, np.ndarray] = {}


def make_kerdock(rotation_seed: int) -> np.ndarray:
    if rotation_seed in ROTATIONS_CACHE:
        return ROTATIONS_CACHE[rotation_seed]
    radius = r.chi_mean(D)
    H = r.walsh_hadamard() / math.sqrt(D)
    rotation = r.haar_rotation(rotation_seed)
    blocks: list[np.ndarray] = []
    for u in range(128):
        chirp = r.kerdock_chirp(u)
        basis = (H * chirp[None, :]) @ rotation
        blocks.extend([(radius * basis).astype(np.float32), (-radius * basis).astype(np.float32)])
    coordinate = (radius * rotation).astype(np.float32)
    blocks.extend([coordinate, -coordinate])
    x = np.concatenate(blocks, axis=0)
    assert x.shape == (r.N_ROWS, D)
    ROTATIONS_CACHE[rotation_seed] = x
    return x


def stream_reference_lower(ws: list[torch.Tensor], n: int, seed: int, chunk: int) -> dict[str, np.ndarray]:
    eng = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed)
    y_sum = np.zeros(D, dtype=np.float64)
    mu_sum = np.zeros(D, dtype=np.float64)
    M_sum = np.zeros((D, D), dtype=np.float64)
    done = 0
    with torch.no_grad():
        while done < n:
            b = min(chunk, n - done)
            u = eng.draw(b, dtype=torch.float32).clamp_(1e-7, 1 - 1e-7)
            x = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
            h, y = r.forward_target_final(x, ws)
            H = h.double().numpy()
            Y = y.double().numpy()
            y_sum += Y.sum(axis=0)
            mu_sum += H.sum(axis=0)
            M_sum += H.T @ H
            done += b
    return {'y': y_sum / n, 'mu': mu_sum / n, 'M': M_sum / n}


def normalize_rms(v: np.ndarray) -> np.ndarray:
    scale = float(np.sqrt(np.mean(v * v)))
    return v / max(scale, 1e-12)


def weight_global_and_node_features(ws: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    wnp = [w.numpy().astype(np.float64, copy=False) for w in ws]
    gl: list[float] = []
    for W in wnp:
        rn = np.linalg.norm(W, axis=1)
        cn = np.linalg.norm(W, axis=0)
        rs = W.sum(axis=1)
        cs = W.sum(axis=0)
        gl.extend([
            float(W.mean()), float(W.std()), float(np.mean(np.abs(W))), float(np.max(np.abs(W))),
            float(rn.mean()), float(rn.std()), float(rn.min()), float(rn.max()),
            float(cn.mean()), float(cn.std()), float(cn.min()), float(cn.max()),
            float(rs.mean()), float(rs.std()), float(cs.mean()), float(cs.std()),
        ])

    # Forward equivariant summaries at the target layer.
    fs = np.ones(D, dtype=np.float64)
    fa = np.ones(D, dtype=np.float64)
    fq = np.ones(D, dtype=np.float64)
    for W in wnp[: TARGET + 1]:
        fs = normalize_rms(W.T @ fs)
        fa = normalize_rms(np.abs(W).T @ fa)
        fq = normalize_rms((W * W).T @ fq)

    # Backward summaries from final outputs to target-layer coordinates.
    bs = np.ones(D, dtype=np.float64)
    ba = np.ones(D, dtype=np.float64)
    bq = np.ones(D, dtype=np.float64)
    for W in reversed(wnp[TARGET + 1 :]):
        bs = normalize_rms(W @ bs)
        ba = normalize_rms(np.abs(W) @ ba)
        bq = normalize_rms((W * W) @ bq)

    Win = wnp[TARGET]
    Wout = wnp[TARGET + 1] if TARGET + 1 < DEPTH else np.zeros_like(Win)
    Wlast = wnp[-1]
    in_col = Win
    out_row = Wout
    node = np.column_stack([
        fs, fa, fq, bs, ba, bq,
        in_col.mean(axis=0), in_col.std(axis=0), np.mean(np.abs(in_col), axis=0), np.linalg.norm(in_col, axis=0),
        out_row.mean(axis=1), out_row.std(axis=1), np.mean(np.abs(out_row), axis=1), np.linalg.norm(out_row, axis=1),
        Wlast.mean(axis=1), Wlast.std(axis=1), np.mean(np.abs(Wlast), axis=1), np.linalg.norm(Wlast, axis=1),
    ]).astype(np.float32)
    return np.asarray(gl, dtype=np.float32), node


def lower_defect_matrix(mu: np.ndarray, M: np.ndarray, m: np.ndarray) -> np.ndarray:
    diag = np.diag(M)
    return (
        diag[:, None] * (mu - m)[None, :]
        + 2.0 * (mu - m)[:, None] * M
        + 2.0 * ((m * m)[:, None] * mu[None, :] - (mu * mu)[:, None] * mu[None, :])
    ) / (D + 1.0)


def aggregate_node(node: np.ndarray, v: np.ndarray) -> np.ndarray:
    av = np.abs(v)
    vv = v * v
    return np.concatenate([
        v @ node,
        av @ node / max(float(av.sum()), 1e-12),
        vv @ node / max(float(vv.sum()), 1e-12),
    ]).astype(np.float32)


def build_token_features(
    Q: np.ndarray,
    indices: np.ndarray,
    directions: np.ndarray,
    m: np.ndarray,
    sample_M: np.ndarray,
    beta_bar: np.ndarray,
    node: np.ndarray,
) -> np.ndarray:
    diagM = np.diag(sample_M)
    tokens: list[np.ndarray] = []
    for rank, (i, v) in enumerate(zip(indices, directions)):
        qrow = Q[i]
        b = beta_bar[rank]
        av = np.abs(v)
        vv = v * v
        scalars = np.asarray([
            rank / max(P - 1, 1),
            float(np.sum(qrow * v)), float(np.linalg.norm(qrow)),
            float(m[i]), float(diagM[i]),
            float(v @ m), float(av @ m), float(vv @ m),
            float(v @ diagM), float(av @ diagM), float(vv @ diagM),
            float(sample_M[i] @ v), float(v @ sample_M @ v),
            float(np.linalg.norm(v, 1)), float(np.max(av)), float(np.sum(v)),
            float(np.linalg.norm(b)), float(b.mean()), float(b.std()), float(np.max(np.abs(b))), float(b.sum()),
        ], dtype=np.float32)
        tokens.append(np.concatenate([scalars, node[i], aggregate_node(node, v)]))
    return np.stack(tokens).astype(np.float32)


def run_rotation(
    network_id: int,
    rotation_seed: int,
    ws: list[torch.Tensor],
    ref1: dict[str, np.ndarray],
    ref2: dict[str, np.ndarray],
    global_features: np.ndarray,
    node_static: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    xk = make_kerdock(rotation_seed)
    Ht, Yt = r.forward_target_final(torch.from_numpy(xk), ws)
    H = Ht.double().numpy()
    Y = Yt.double().numpy()
    m = H.mean(axis=0)
    base = Y.mean(axis=0)
    rho = r.chi_mean(D)
    Q = r.sample_anchor_matrix(H, m, rho)
    indices, directions = r.sample_row_probes(Q, P)
    X = r.radial_features_sample_rows(H, m, indices, directions, rho)
    fit = r.fit_crossfit(X, Y)
    q_anchor = r.contract_rows(Q, indices, directions)
    sample_pred, _ = r.estimate_from_fit(fit, q_anchor)
    fw = fit['fold_sizes'] / fit['fold_sizes'].sum()
    beta_bar = np.einsum('f,fpd->pd', fw, fit['betas'])

    mu = 0.5 * (ref1['mu'] + ref2['mu'])
    M = 0.5 * (ref1['M'] + ref2['M'])
    defect = lower_defect_matrix(mu, M, m)
    delta = r.contract_rows(defect, indices, directions)
    oracle_pred = sample_pred + delta @ beta_bar
    y1, y2 = ref1['y'], ref2['y']
    base_mse = r.pooled_mse(base, y1, y2)
    sample_mse = r.pooled_mse(sample_pred, y1, y2)
    oracle_mse = r.pooled_mse(oracle_pred, y1, y2)
    base_unbiased = r.unbiased_mse(base, y1, y2)
    oracle_unbiased = r.unbiased_mse(oracle_pred, y1, y2)

    sample_M = H.T @ H / len(H)
    node = np.column_stack([node_static, m.astype(np.float32), np.diag(sample_M).astype(np.float32)])
    token = build_token_features(Q, indices, directions, m, sample_M, beta_bar, node)

    return {
        'network_id': np.asarray(network_id, dtype=np.int64),
        'rotation_seed': np.asarray(rotation_seed, dtype=np.int64),
        'global_features': global_features,
        'token_features': token,
        'target_delta': delta.astype(np.float32),
        'beta_bar': beta_bar.astype(np.float32),
        'probe_indices': indices.astype(np.int64),
        'probe_directions': directions.astype(np.float32),
        'q_anchor': q_anchor.astype(np.float32),
        'sample_prediction': sample_pred.astype(np.float64),
        'baseline_prediction': base.astype(np.float64),
        'oracle_prediction': oracle_pred.astype(np.float64),
        'truth_half1': y1.astype(np.float64),
        'truth_half2': y2.astype(np.float64),
        'base_mse': np.asarray(base_mse),
        'sample_mse': np.asarray(sample_mse),
        'oracle_mse': np.asarray(oracle_mse),
        'base_unbiased_mse': np.asarray(base_unbiased),
        'oracle_unbiased_mse': np.asarray(oracle_unbiased),
        'oracle_ratio': np.asarray(oracle_mse / max(base_mse, 1e-300)),
        'headroom_label': np.asarray(float(oracle_mse < base_mse), dtype=np.float32),
        'target_correction': (oracle_pred - sample_pred).astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--split', choices=['train','validation','test'], required=True)
    ap.add_argument('--network-id', type=int, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--threads', type=int, default=5)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.network_id not in cfg['base_network_ids'][args.split]:
        raise ValueError('network not in frozen split')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.outdir.mkdir(parents=True, exist_ok=True)
    outfile = args.outdir / f'{args.split}_network_{args.network_id}.npz'
    if outfile.exists():
        print(json.dumps({'status':'exists','file':str(outfile)}))
        return
    t0 = time.time()
    ws, whash, wseed = r.make_weights(args.network_id)
    n = int(cfg['reference_n_per_half'][args.split])
    seed0 = 21_000_000 + 2 * args.network_id
    ref1 = stream_reference_lower(ws, n, seed0, args.chunk)
    ref2 = stream_reference_lower(ws, n, seed0 + 1, args.chunk)
    gf, node = weight_global_and_node_features(ws)
    rotations = cfg['rotations'][args.split]
    examples = [run_rotation(args.network_id, int(rot), ws, ref1, ref2, gf, node) for rot in rotations]
    payload: dict[str, np.ndarray] = {
        'split': np.asarray(args.split),
        'network_id': np.asarray(args.network_id),
        'weight_seed': np.asarray(wseed),
        'weight_sha256': np.asarray(whash),
        'reference_n_per_half': np.asarray(n),
        'reference_seed1': np.asarray(seed0),
        'reference_seed2': np.asarray(seed0+1),
        'rotation_seeds': np.asarray(rotations, dtype=np.int64),
        'global_features': np.stack([e['global_features'] for e in examples]),
        'token_features': np.stack([e['token_features'] for e in examples]),
        'target_delta': np.stack([e['target_delta'] for e in examples]),
        'beta_bar': np.stack([e['beta_bar'] for e in examples]),
        'probe_indices': np.stack([e['probe_indices'] for e in examples]),
        'probe_directions': np.stack([e['probe_directions'] for e in examples]),
        'q_anchor': np.stack([e['q_anchor'] for e in examples]),
        'sample_prediction': np.stack([e['sample_prediction'] for e in examples]),
        'baseline_prediction': np.stack([e['baseline_prediction'] for e in examples]),
        'oracle_prediction': np.stack([e['oracle_prediction'] for e in examples]),
        'truth_half1': np.stack([e['truth_half1'] for e in examples]),
        'truth_half2': np.stack([e['truth_half2'] for e in examples]),
        'base_mse': np.asarray([e['base_mse'] for e in examples]),
        'sample_mse': np.asarray([e['sample_mse'] for e in examples]),
        'oracle_mse': np.asarray([e['oracle_mse'] for e in examples]),
        'base_unbiased_mse': np.asarray([e['base_unbiased_mse'] for e in examples]),
        'oracle_unbiased_mse': np.asarray([e['oracle_unbiased_mse'] for e in examples]),
        'oracle_ratio': np.asarray([e['oracle_ratio'] for e in examples]),
        'headroom_label': np.asarray([e['headroom_label'] for e in examples]),
        'target_correction': np.stack([e['target_correction'] for e in examples]),
        'runtime_seconds': np.asarray(time.time()-t0),
        'freeze_sha256': np.asarray(cfg['freeze_sha256']),
    }
    np.savez_compressed(outfile, **payload)
    print(json.dumps({
        'status':'written','file':str(outfile),'split':args.split,'network':args.network_id,
        'rotations':rotations,'oracle_ratios':payload['oracle_ratio'].tolist(),
        'seconds':float(payload['runtime_seconds']),
    }), flush=True)

if __name__ == '__main__':
    main()

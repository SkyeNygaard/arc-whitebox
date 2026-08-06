#!/usr/bin/env python3
"""Evaluate the K3/Gaussian hybrid against the POST-ReLU layer means.

The grader scores post-activation means: whestbench.simulation runs
``x = relu(x @ w)`` at every layer, including the last, and compares the
per-layer sample means. The earlier stage scored ``pre_mean`` instead, which is
a different target, so gamma tuned there does not transfer automatically.

Layer index l here is the activation after weight matrix l, matching
``moments['mean'][l]``. The reported final_mean_mse is layer depth-1.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from hybrid_k3_coefnet_v2 import HybridConfig, HybridK3CoefNetV2


def load_moments(path):
    with np.load(path) as d:
        return {'global_index': int(d['global_index']),
                'post_mean': np.asarray(d['mean'], np.float64)}


def load_weights(path, device, dtype):
    w = np.asarray(np.load(path), np.float64)
    if w.shape != (32, 256, 256) or not np.isfinite(w).all():
        raise ValueError(f'bad weights {path}: {w.shape}')
    return torch.as_tensor(w.transpose(0, 2, 1).copy(), device=device, dtype=dtype)


def parse_config(text):
    p = [float(x) for x in text.split(',')]
    if len(p) < 3:
        raise ValueError(text)
    return HybridConfig(alpha=p[0], beta=p[1], gamma=p[2],
                        corr_cap=p[3] if len(p) > 3 else .999,
                        x_clip=p[4] if len(p) > 4 else 20.,
                        residual_clip=p[5] if len(p) > 5 else .5)


def key(c):
    return f'a{c.alpha:g}_b{c.beta:g}_g{c.gamma:g}'


def parse_indices(spec):
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = map(int, part.split('-', 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def run_one(moment_path, weights_dir, model_path, config, device, dtype, calibration):
    from mlp_kprop.kprop_harmonic import SIMPLE, coerce_input, linear_kprop, nonlin_kprop
    from mlp_kprop.wick import relu_wick_coef
    m = load_moments(moment_path)
    idx = m['global_index']
    W = load_weights(weights_dir / f'mlp_{idx:05d}.npy', device, dtype)
    K = coerce_input({1: torch.zeros(256, device=device, dtype=dtype),
                      2: torch.eye(256, device=device, dtype=dtype)}, k_max=3, kind=SIMPLE)
    patch = None if config is None else HybridK3CoefNetV2(
        model_path, config, 32, device, dtype, calibration_path=calibration)
    layers, safety = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for l in range(32):
            Kpre = linear_kprop(K, W[l], k_max=3)
            Kpost = nonlin_kprop(Kpre, nonlin_wick_coef=relu_wick_coef, k_max=3,
                                 kind=SIMPLE, use_pK=True, factor=True)
            if patch is not None:
                nxt = W[l + 1] if l + 1 < 32 else None
                patch.apply_(Kpre, Kpost, l, next_weights=nxt)
                d = dict(patch.last_diagnostics)
                d['layer'] = l
                safety.append(d)
                if d['nonfinite_mean'] or d['nonfinite_covariance']:
                    raise FloatingPointError((idx, l, d))
            pred = Kpost[1].core.detach().cpu().numpy().astype(np.float64)
            mse = float(np.mean((pred - m['post_mean'][l]) ** 2))
            layers.append({'layer': l, 'mean_mse': mse})
            K = Kpost
    return {'global_index': idx, 'config': 'upstream' if config is None else key(config),
            'runtime_seconds': time.perf_counter() - start, 'layers': layers,
            'final_mean_mse': layers[-1]['mean_mse'],
            'all_layers_mse': float(np.mean([x['mean_mse'] for x in layers])),
            'safety': safety}


def summarize(rows, base=None):
    mse = float(np.mean([r['final_mean_mse'] for r in rows]))
    o = {'mlps': len(rows), 'final_mean_mse': mse,
         'all_layers_mse': float(np.mean([r['all_layers_mse'] for r in rows])),
         'median_final_mean_mse': float(np.median([r['final_mean_mse'] for r in rows])),
         'runtime_seconds': float(sum(r['runtime_seconds'] for r in rows))}
    if base:
        bm = float(np.mean([base[r['global_index']]['final_mean_mse'] for r in rows]))
        ratios = [base[r['global_index']]['final_mean_mse'] / max(r['final_mean_mse'], 1e-300)
                  for r in rows]
        o.update({'baseline_mean_mse': bm, 'gain_vs_upstream': bm / max(mse, 1e-300),
                  'fraction_mlps_improved': float(np.mean(np.array(ratios) > 1)),
                  'median_mlp_gain': float(np.median(ratios)),
                  'worst_mlp_gain': float(np.min(ratios))})
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-json', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--moments-dir', type=Path, required=True)
    p.add_argument('--weights-dir', type=Path, required=True)
    p.add_argument('--split', choices=['valid', 'test'])
    p.add_argument('--indices', default='')
    p.add_argument('--configs', default='0,0,0.6')
    p.add_argument('--calibration', type=Path)
    p.add_argument('--start-mlp', type=int, default=0)
    p.add_argument('--max-mlps', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.indices:
        ids = parse_indices(a.indices)
        label = 'fresh_indices'
    elif a.split:
        ids = list(map(int, json.loads(a.results_json.read_text())[f'{a.split}_ids']))
        label = a.split
    else:
        raise SystemExit('provide --split or --indices')
    ids = ids[a.start_mlp:]
    ids = ids[:a.max_mlps] if a.max_mlps else ids
    configs = [parse_config(x) for x in a.configs.split(';') if x.strip()]
    device = torch.device(a.device)
    dtype = torch.float64 if a.dtype == 'float64' else torch.float32
    base = []
    for i, idx in enumerate(ids, 1):
        r = run_one(a.moments_dir / f'mlp_{idx:05d}.npz', a.weights_dir, a.model, None,
                    device, dtype, a.calibration)
        base.append(r)
        print(json.dumps({'stage': 'upstream', 'loaded': i, 'global_index': idx,
                          'mse': r['final_mean_mse']}), flush=True)
    bmap = {r['global_index']: r for r in base}
    entries = {}
    for c in configs:
        rows = []
        for i, idx in enumerate(ids, 1):
            r = run_one(a.moments_dir / f'mlp_{idx:05d}.npz', a.weights_dir, a.model, c,
                        device, dtype, a.calibration)
            rows.append(r)
            print(json.dumps({'stage': 'hybrid', 'config': key(c), 'loaded': i,
                              'global_index': idx, 'mse': r['final_mean_mse']}), flush=True)
        entries[key(c)] = {'config': c.__dict__, 'rows': rows, 'summary': summarize(rows, bmap)}
        print(json.dumps({'config': key(c), **entries[key(c)]['summary']}), flush=True)
    result = {'split': label, 'target': 'post_relu_layer_mean',
              'calibration': str(a.calibration) if a.calibration else None,
              'upstream': {'rows': base, 'summary': summarize(base)}, 'hybrid': entries}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'output': str(a.output)}))


if __name__ == '__main__':
    main()

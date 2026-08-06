#!/usr/bin/env python3
"""Print a compact go/no-go decision from eval_next_variance_x1.py output."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def verdict(summary: dict) -> tuple[str,list[str]]:
    gain=float(summary.get('relative_variance_gain',0.0))
    fraction=float(summary.get('relative_variance_fraction_improved',0.0))
    sigma=float(summary.get('model_sigma_relative_rms',999.0))
    orientation=float(summary.get('max_orientation_relative_rms',999.0))
    negative=float(summary.get('negative_variance_fraction',1.0))
    reasons=[]
    if orientation>0.005:reasons.append(f'weight/layer orientation mismatch ({orientation:.2%})')
    if gain<1.3:reasons.append(f'contraction gain too small ({gain:.2f}x)')
    if fraction<0.8:reasons.append(f'too few cases improve ({fraction:.1%})')
    if negative>0:reasons.append(f'negative predicted variances ({negative:.2%})')
    if reasons:return 'STOP',reasons
    if gain>=1.5 and fraction>=0.8:
        return 'GO TO ROLLOUT',[f'{gain:.2f}x relative-variance gain',f'{fraction:.1%} cases improve',f'model sigma RMS {sigma:.2%}']
    return 'BORDERLINE',[f'{gain:.2f}x relative-variance gain',f'{fraction:.1%} cases improve']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('result',type=Path);a=ap.parse_args();d=json.loads(a.result.read_text())
    for mode,m in d['modes'].items():
        for label in ['test_alpha1','test']:
            s=m[label]['summary'];v,r=verdict(s)
            print(f'{mode} / {label} / alpha={1.0 if label=="test_alpha1" else m["alpha"]:.4g}: {v}')
            for x in r:print('  -',x)
            print('  relative variance gain:',f'{s["relative_variance_gain"]:.3f}x')
            print('  sigma gain:',f'{s["sigma_gain"]:.3f}x')
            print('  base/model sigma RMS:',f'{s["base_sigma_relative_rms"]:.3%}',f'{s["model_sigma_relative_rms"]:.3%}')
if __name__=='__main__':main()

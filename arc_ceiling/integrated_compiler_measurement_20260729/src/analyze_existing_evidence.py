#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def interval(values,reps=20000,seed=20260729):
    x=np.asarray(values,float);rng=np.random.default_rng(seed);means=np.empty(reps);n=len(x)
    for i in range(reps):means[i]=x[rng.integers(0,n,n)].mean()
    return [float(q) for q in np.quantile(means,[.025,.975])]

def main():
    p=ROOT/'results'/'adaptive_suffix_holdouts_60.csv'
    with p.open() as f:rows=list(csv.DictReader(f))
    raw=np.array([float(r['raw_mse_ratio']) for r in rows]);ideal=np.array([float(r['ideal_cost_ratio']) for r in rows]);cal=np.array([float(r['calibrated_score_ratio']) for r in rows]);depth=np.array([int(r['chosen_depth']) for r in rows])
    guarded=cal.copy();fallback=ideal>0.995;guarded[fallback]=1.0
    payload={'label':'historical frozen full-width evidence only; not current-package official measurement','networks':len(rows),'adaptive_2_6':{'mean_raw_mse_ratio':float(raw.mean()),'raw_mse_ratio_interval':interval(raw),'raw_wins':int(np.sum(raw<1)),'median_raw_ratio':float(np.median(raw)),'worst_raw_ratio':float(raw.max()),'mean_ideal_cost_ratio':float(ideal.mean()),'mean_calibrated_score_ratio':float(cal.mean()),'calibrated_interval':interval(cal),'calibrated_wins':int(np.sum(cal<1)),'median_calibrated_ratio':float(np.median(cal)),'worst_calibrated_ratio':float(cal.max()),'guard_fallback_count_reconstructed':int(fallback.sum()),'guarded_mean_ratio_reconstructed':float(guarded.mean()),'guarded_interval_reconstructed':interval(guarded),'guarded_wins_reconstructed':int(np.sum(guarded<1)),'selected_depth_counts':{str(k):int(np.sum(depth==k)) for k in sorted(set(depth))}}}
    (ROOT/'results'/'historical_evidence_reanalysis.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()

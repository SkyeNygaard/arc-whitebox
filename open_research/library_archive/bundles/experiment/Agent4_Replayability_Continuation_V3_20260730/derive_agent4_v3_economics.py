#!/usr/bin/env python3
"""Derive architecture-specific WHestBench score frontiers for rank-linear replay."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

TARGET = 0.2304147465437788
BASE_EFFECTIVE_B = 174.5
PER_SHIFT_B = 0.0847
OLD_FIXED_MULT = 1.0312478632478632

# Latest Agent-5 continuation exact nonlinear-secant frontier.
ROWS = [
    {"rank": 4, "rstar": 0.441716, "worst": 0.846},
    {"rank": 5, "rstar": 0.404465, "worst": 0.812},
    {"rank": 8, "rstar": 0.290770, "worst": 0.663},
    {"rank": 12, "rstar": 0.220745271, "worst": 0.513},
    {"rank": 20, "rstar": 0.127663, "worst": 0.290},
    {"rank": 24, "rstar": 0.109470, "worst": 0.274},
    {"rank": 32, "rstar": 0.081667, "worst": 0.221},
]


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--json',type=Path,required=True); p.add_argument('--csv',type=Path,required=True)
    a=p.parse_args()
    out=[]
    for row in ROWS:
        r=row['rank']; rs=row['rstar']; delta=r*PER_SHIFT_B
        c0=(BASE_EFFECTIVE_B+delta)/BASE_EFFECTIVE_B
        adjusted=rs*c0
        old_adjusted=rs*OLD_FIXED_MULT
        noise_headroom=max(0.0, TARGET/c0-rs)
        max_total_replay_B=BASE_EFFECTIVE_B*(TARGET/rs-1.0) if rs>0 else math.inf
        per_shift_ceiling=max_total_replay_B/r
        # Shared-vector frontier: if one sample costs gamma baseline units and has
        # normalized covariance trace u, winning requires
        # (sqrt(c0*r*) + sqrt(gamma*u))^2 < TARGET.
        sqrt_margin=max(0.0, math.sqrt(TARGET)-math.sqrt(c0*rs))
        gamma_u_ceiling=sqrt_margin**2
        out.append({
            **row,
            'replay_added_B':delta,
            'replay_fraction_of_current':delta/BASE_EFFECTIVE_B,
            'c0_rank_linear':c0,
            'replay_adjusted_ratio':adjusted,
            'old_fixed_replay_adjusted_ratio':old_adjusted,
            'adjusted_ratio_improvement_vs_old':old_adjusted-adjusted,
            'normalized_additive_risk_headroom':noise_headroom,
            'max_total_replay_B_before_zero_noise_loss':max_total_replay_B,
            'max_per_shift_B_before_zero_noise_loss':per_shift_ceiling,
            'actual_per_shift_fraction_of_ceiling':PER_SHIFT_B/per_shift_ceiling if per_shift_ceiling>0 else math.inf,
            'shared_estimator_gamma_times_trace_ceiling':gamma_u_ceiling,
            'aggregate_zero_noise_win':adjusted<TARGET,
            'worst_case_raw_below_target':row['worst']<TARGET,
        })
    result={
        'target_ratio':TARGET,'base_effective_compute_B':BASE_EFFECTIVE_B,'validated_per_shift_replay_B':PER_SHIFT_B,
        'old_fixed_multiplier':OLD_FIXED_MULT,'rows':out,
        'claims':{
            'rank_linear_cost_formula':'c0(r)=(174.5+0.0847*r)/174.5',
            'additive_risk_gate':'u < target/c0-rstar',
            'shared_vector_gate':'gamma*trSigma < (sqrt(target)-sqrt(c0*rstar))^2',
        }
    }
    a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    with a.csv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Exact/projected flopscope 0.9.1 cost certificate for sorted stop-loss replay.

Uses the published official family formulas.  Sort and cumsum costs are exact for
specified shapes. Query arithmetic is explicitly enumerated; because the exact dtype
resolution of searchsorted/indexing can only make this tiny term differ, both a
conservative upper bound and a dominant-core value are emitted.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
N=66048; M=256; BASE_B=174.5; TARGET=0.2304147465437788; PER_SHIFT_B=.0847
RSTAR={12:.220745271,20:.127663,24:.109470,32:.081667}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 logn=math.ceil(math.log2(N)); num=N*M
 # Official v0.9.1: sort cost = slices*n*ceil(log2 n), weight 4, fp32 rate 1.
 sort=4*M*N*logn
 # cumsum float64 along N: (numel - num_slices), weight 1, rate 2.
 cumsum=2*(num-M)
 # Optional concatenation/output writes and per-query ops.  We give a deliberately
 # conservative 128 billed fp32-equivalent ops per scalar stop-loss query, including
 # searchsorted (<= 4*2*logn=136 if int64 rate were used), gather/select/arithmetic.
 # Use 192 to safely cover dtype/indexing ambiguity and column stack writes.
 cache_z_write=num  # conservative: materialize/cache Z once (fp32 sequential write)
 core=sort+cumsum+cache_z_write
 rows=[]
 for r,rs in RSTAR.items():
  query_upper=192*r*M
  output_write=2*r*M
  total=core+query_upper+output_write
  scan=r*PER_SHIFT_B*1e9
  c0=(BASE_B+total/1e9)/BASE_B
  adj=rs*c0
  rows.append({'rank':r,'rstar':rs,'sort_flops':sort,'cumsum_flops':cumsum,'cache_z_write_flops':cache_z_write,'query_upper_flops':query_upper,'output_write_flops':output_write,'sorted_total_upper_flops':total,'sorted_total_upper_B':total/1e9,'serialized_translation_B':scan/1e9,'sorted_saving_B_vs_serialized':scan/1e9-total/1e9,'sorted_cost_ratio_vs_serialized':total/scan,'sorted_c0_upper':c0,'sorted_replay_adjusted_ratio_upper':adj,'sorted_zero_noise_win':adj<TARGET})
 result={'flopscope_version':'0.9.1','shape':[N,M],'ceil_log2_N':logn,'formula_sources':{'sort':'4 * M * N * ceil(log2 N), fp32','cumsum':'2 * (N*M-M), float64 accumulator','cache_Z':'N*M sequential fp32 writes','query_upper':'192 * rank * M conservative envelope'},'dominant_core_flops':core,'dominant_core_B':core/1e9,'break_even_rank_vs_0.0847B_per_shift':core/(PER_SHIFT_B*1e9),'rows':rows}
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()

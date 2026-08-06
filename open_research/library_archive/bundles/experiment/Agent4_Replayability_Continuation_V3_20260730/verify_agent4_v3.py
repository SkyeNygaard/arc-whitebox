#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def load(root,name): return json.loads((root/name).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root
 rep=load(r,'replay_benchmark_results.json'); blk=load(r,'blockwise_sorted_benchmark.json'); fl=load(r,'sorted_stoploss_flops.json'); ec=load(r,'agent4_v3_economics.json'); ds=load(r,'direct_output_source_economics.json'); bs=load(r,'finite_design_relu_blind_spot.json'); sk=load(r,'sorted_stoploss_kernel_selftest.json')
 checks={}
 checks['scan_batches_exact']=all(rep['ranks'][str(rank)][f'batched_{b}']['max_abs_vs_batch1']==0 for rank in [20,24,32] for b in [1,2,4,8])
 checks['batch1_fastest']=all(rep['ranks'][str(rank)]['batched_1']['median_seconds']<=min(rep['ranks'][str(rank)][f'batched_{b}']['median_seconds'] for b in [1,2,4,8]) for rank in [20,24,32])
 checks['sorted_scan_match']=rep['sorted_stoploss']['queries']['32']['max_abs_vs_scan']<2e-8 and max(v['max_abs_vs_scan'] for v in blk['blocks'].values())<2e-8
 checks['kernel_selftest']=sk['all_pass']
 checks['flop_core_formula']=fl['dominant_core_flops']==4*256*66048*17+2*(66048*256-256)+66048*256
 checks['sort_break_even']=14<fl['break_even_rank_vs_0.0847B_per_shift']<15
 er={int(x['rank']):x for x in ec['rows']}; checks['rank_costs']=abs(er[24]['replay_added_B']-2.0328)<1e-12 and er[32]['worst_case_raw_below_target']
 dr={x['name']:x for x in ds['rows']}; checks['direct_source_priority']=dr['direct_rank32']['tail_gate_pass'] and dr['direct_rank40']['tail_gate_pass'] and dr['adaptive_direct']['tail_gate_pass'] and not dr['direct_rank20']['tail_gate_pass']
 checks['blind_on_design']=bs['max_value_on_design_nodes']==0 and bs['design_nodes_above_1e_13']==0
 checks['blind_positive_mean']=bs['exact_standard_gaussian_mean']>0 and bs['network_formula_max_error']<1e-15
 checks['transcript_no_go']=bs['direct_output_pair']['max_group_transcript_difference_h_vs_h_plus_g']==0 and bs['direct_output_pair']['rank1_contraction_difference']>0
 out={'checks':checks,'all_pass':all(checks.values())};a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True));
 if not out['all_pass']: raise SystemExit(1)
if __name__=='__main__': main()

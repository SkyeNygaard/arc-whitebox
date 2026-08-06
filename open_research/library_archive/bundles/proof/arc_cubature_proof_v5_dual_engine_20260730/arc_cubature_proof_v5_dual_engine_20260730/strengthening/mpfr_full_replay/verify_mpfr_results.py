#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from decimal import Decimal as D
from pathlib import Path

def load(p:Path): return json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--proof-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); r=a.run_dir; p=a.proof_dir
    v=load(r/'verify_certified.json'); g=load(r/'regenerate_mesh.json'); m=load(r/'global_minorant.json'); km=load(r/'kernel_mean_512.json'); ft=load(r/'final_theorem.json'); cov=load(r/'exact_coverage.json')
    assert v['passed'] and v['initial_intervals']==1421 and v['accepted_intervals']==1421
    assert g['passed'] and g['initial_intervals']==1079 and g['accepted_intervals']==1421 and g['splits']==342 and g['maximum_depth']==4
    assert m['passed'] and m['rows']==13 and D(m['global_candidate_upper_bound'])<0
    assert cov['passed'] and cov['full_domain_coverage_minus1_to1'] and cov['no_gaps_or_overlaps']
    orig_mean=load(p/'results/FORMAL_KERNEL_MEAN_D256_L32.json')['A0_certified']
    assert D(orig_mean['lower']) <= D(km['A0_lower']) <= D(km['A0_upper']) <= D(orig_mean['upper'])
    orig_energy=load(p/'results/FORMAL_DELSARTE_BOUND_D256_L32.json')
    assert D(orig_energy['kerdock_energy']['lower']) <= D(ft['kerdock_energy_lower']) <= D(ft['kerdock_energy_upper']) <= D(orig_energy['kerdock_energy']['upper'])
    assert D(orig_energy['universal_energy_lower_bound']['lower']) <= D(ft['universal_energy_bound_lower']) <= D(ft['universal_energy_bound_upper']) <= D(orig_energy['universal_energy_lower_bound']['upper'])
    orig_theorem=load(p/'results/FORMAL_NEAR_OPTIMALITY_THEOREM_D256_L32.json')
    published_ratio=D(orig_theorem['actual_multiplicative_ratio_kerdock_over_infimum']['upper'])
    published_percent=D(orig_theorem['actual_relative_excess_percent']['upper'])
    published_add=D(orig_theorem['actual_additive_suboptimality']['upper'])
    assert D(1) <= D(ft['ratio_upper']) <= published_ratio
    assert D(ft['relative_excess_percent_upper']) <= published_percent
    assert D(ft['additive_suboptimality_upper']) <= published_add
    out={
      'full_curvature_mesh_second_engine':True,
      'global_minorant_second_engine':True,
      'spherical_mean_second_engine':True,
      'delsarte_and_final_ratio_second_engine':True,
      'mpfr_mean_interval_nested_in_original':True,
      'mpfr_energy_intervals_nested_in_original':True,
      'mpfr_ratio_upper':ft['ratio_upper'],
      'published_ratio_upper':str(published_ratio),
      'mpfr_relative_excess_percent_upper':ft['relative_excess_percent_upper'],
      'published_relative_excess_percent_upper':str(published_percent),
      'passed':True,
    }
    a.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

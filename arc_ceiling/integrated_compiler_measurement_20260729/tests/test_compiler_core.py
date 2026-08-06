from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compiler_core import CandidateSpec, classify_layer, compile_suffix_mean, cost_proxy, exact_pilot_suffix, run_candidate
from fast_matmul_generic import winograd_hybrid_p3_d5_partial_tree


def relu_forward(anchor,weights):
    x=anchor
    for w in weights:x=np.maximum(x@w,0)
    return x


def test_partial_tree_matches_dense():
    rng=np.random.default_rng(1);a=rng.standard_normal((64,32),dtype=np.float32);b=rng.standard_normal((32,32),dtype=np.float32)
    got=winograd_hybrid_p3_d5_partial_tree(a,b,np)
    ref=a@b
    assert np.max(np.abs(got-ref)) < 3e-4


def test_all_kink_compiler_is_exact():
    rng=np.random.default_rng(2);n,d,k=48,12,3
    anchor=np.maximum(rng.standard_normal((n,d)),0).astype(np.float32)
    weights=[(rng.standard_normal((d,d))*0.2).astype(np.float32) for _ in range(k)]
    pilot=np.arange(n,dtype=np.int64);pres,y=exact_pilot_suffix(anchor[pilot],weights,np)
    all_idx=np.arange(d,dtype=np.int64);empty=np.empty(0,dtype=np.int64)
    classes=[(empty,empty,all_idx) for _ in range(k)]
    got=compile_suffix_mean(anchor,weights,classes,pilot,y,0.875,np)
    ref=relu_forward(anchor,weights).mean(0,dtype=np.float64)
    assert np.max(np.abs(got-ref)) < 1e-8


def test_full_row_classification_is_exact():
    rng=np.random.default_rng(3);n,d,k=80,16,4
    anchor=np.maximum(rng.standard_normal((n,d)),0).astype(np.float32)
    weights=[(rng.standard_normal((d,d))*0.15).astype(np.float32) for _ in range(k)]
    pilot=np.arange(n,dtype=np.int64);pres,y=exact_pilot_suffix(anchor,weights,np)
    classes=[classify_layer(h,0,np) for h in pres]
    got=compile_suffix_mean(anchor,weights,classes,pilot,y,1.0,np)
    ref=relu_forward(anchor,weights).mean(0,dtype=np.float64)
    assert np.max(np.abs(got-ref)) < 1e-8


def test_cost_proxy_and_guard_fallback():
    rng=np.random.default_rng(4);n,d,depth=64,32,8
    first=np.maximum(rng.standard_normal((n,d)),0).astype(np.float32)
    weights=[(rng.standard_normal((d,d))*0.1).astype(np.float32) for _ in range(depth-1)]
    spec=CandidateSpec('guard_test',(3,),8,1,1.0,0.0)
    pilot=np.arange(8,dtype=np.int64)
    got,diag=run_candidate(first,weights,spec,lambda a,b:winograd_hybrid_p3_d5_partial_tree(a,b,np),np,pilot_rows=pilot,total_depth=depth)
    ref=relu_forward(first,weights).mean(0,dtype=np.float64)
    assert diag['fallback'] is True
    assert np.max(np.abs(got-ref)) < 2e-5


def test_adaptive_selects_legal_depth_and_is_finite():
    rng=np.random.default_rng(5);n,d,depth=64,32,8
    first=np.maximum(rng.standard_normal((n,d)),0).astype(np.float32)
    weights=[(rng.standard_normal((d,d))*0.08).astype(np.float32) for _ in range(depth-1)]
    spec=CandidateSpec('adaptive_test',(2,3,4,5,6),8,1,1.0,None)
    pilot=np.arange(16,dtype=np.int64)
    got,diag=run_candidate(first,weights,spec,lambda a,b:winograd_hybrid_p3_d5_partial_tree(a,b,np),np,pilot_rows=pilot,total_depth=depth)
    assert diag['selected_depth'] in spec.depths
    assert np.all(np.isfinite(got))
    assert got.shape==(d,)


def test_partition_counts_cover_width():
    h=np.array([[1,-1,1],[1,-2,-1],[-1,-3,2]],dtype=float)
    on,off,kink=classify_layer(h,1,np)
    assert len(on)+len(off)+len(kink)==3

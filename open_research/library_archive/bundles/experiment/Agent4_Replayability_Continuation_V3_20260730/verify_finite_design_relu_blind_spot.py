#!/usr/bin/env python3
"""Construct a positive homogeneous ReLU blind spot for the public Kerdock design.

The function g(x)=min((a^T x)_+,(b^T x)_+) is represented by a width-2,
three-ReLU-layer bias-free network.  A two-dimensional wedge is chosen inside the
largest angular gap among projections of all 66,048 public baseline nodes. Therefore
g is zero on every baseline node but has strictly positive Gaussian expectation.

This proves that finite node outputs alone cannot determine the Gaussian expectation
for the full homogeneous ReLU network class. It does not rule out identities that use
additional weight information or distributional assumptions on random He networks.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def hadamard(n: int) -> np.ndarray:
    H=np.array([[1.0]],dtype=np.float64)
    while H.shape[0]<n:
        H=np.block([[H,H],[H,-H]])
    return H


def main():
    p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=np.load(a.asset)
    chirps=np.asarray(data['chirps'],dtype=np.float64)
    rotation=np.asarray(data['rotation'],dtype=np.float64)
    width=rotation.shape[0]
    rng=np.random.default_rng(40420260730)
    U=rng.normal(size=(width,2)); U,_=np.linalg.qr(U)
    Rproj=rotation@U
    H=hadamard(width)
    positive=[]
    for c in chirps:
        positive.append((H*c[None,:])@Rproj/math.sqrt(width))
    positive.append(Rproj)  # coordinate basis rows; radial scale irrelevant
    pos=np.concatenate(positive,axis=0)
    proj=np.concatenate([pos,-pos],axis=0)
    norms=np.linalg.norm(proj,axis=1)
    assert float(norms.min())>1e-12
    angles=np.mod(np.arctan2(proj[:,1],proj[:,0]),2*math.pi)
    order=np.argsort(angles); s=angles[order]
    extended=np.concatenate([s,[s[0]+2*math.pi]])
    gaps=np.diff(extended); i=int(np.argmax(gaps)); gap=float(gaps[i]); left=float(s[i]);
    # Use the central half of the empty gap to create strict margin from both endpoint nodes.
    alpha=left+gap/4; beta=left+3*gap/4; delta=beta-alpha
    # Inward halfspace normals for wedge angles alpha < theta < beta.
    av=np.array([math.cos(alpha+math.pi/2),math.sin(alpha+math.pi/2)])
    bv=np.array([math.cos(beta-math.pi/2),math.sin(beta-math.pi/2)])
    p1=np.maximum(proj@av,0.0); p2=np.maximum(proj@bv,0.0); g=np.minimum(p1,p2)
    max_node=float(g.max()); count_positive=int(np.count_nonzero(g>1e-13))
    # Exact expectation for a standard 2-D Gaussian projection.
    exact_mean=math.sqrt(math.pi/2)/math.pi * (1-math.cos(delta/2))
    # Direct-output contraction no-go: choose a scalar ridge h with nonconstant
    # basis-group means and compare h against h + lambda*g.  Because g vanishes
    # on every node, all 129 group means and their PCA source are identical, while
    # the Gaussian target and hence the rank-1 contraction differ.
    cv=np.array([math.cos(0.371),math.sin(0.371)])
    group_h=[]
    for block in positive:
        vals=np.maximum(block@cv,0.0)
        vals_neg=np.maximum((-block)@cv,0.0)
        group_h.append(float((vals.sum()+vals_neg.sum())/(2*len(block))))
    group_h=np.asarray(group_h)
    transcript_variance=float(np.var(group_h))
    lam=1.0
    transcript_max_difference=0.0  # exact from g(node)=0
    target_h=1/math.sqrt(2*math.pi)
    target_f=target_h+lam*exact_mean
    # Verify the explicit width-2, depth-3 network algebra on random points.
    X=rng.normal(size=(100000,2)); h1=np.maximum(np.stack([X@av,X@bv],axis=1),0.0)
    h2=np.maximum(np.stack([h1[:,0],h1[:,0]-h1[:,1]],axis=1),0.0)
    h3=np.maximum(h2[:,0]-h2[:,1],0.0)
    direct=np.minimum(np.maximum(X@av,0.0),np.maximum(X@bv,0.0))
    network_err=float(np.max(np.abs(h3-direct)))
    mc=float(direct.mean()); mc_se=float(direct.std(ddof=1)/math.sqrt(len(direct)))
    out={
      'protected_data_opened':False,
      'asset':str(a.asset),
      'node_count':int(len(proj)),
      'positive_pair_count':int(len(pos)),
      'projection_min_norm':float(norms.min()),
      'largest_angular_gap_radians':gap,
      'wedge_width_radians':delta,
      'max_value_on_design_nodes':max_node,
      'design_nodes_above_1e_13':count_positive,
      'exact_standard_gaussian_mean':exact_mean,
      'monte_carlo_mean_100k':mc,
      'monte_carlo_standard_error':mc_se,
      'network_formula_max_error':network_err,
      'direct_output_pair': {
        'group_count': int(len(group_h)),
        'group_mean_variance_for_ridge_h': transcript_variance,
        'max_group_transcript_difference_h_vs_h_plus_g': transcript_max_difference,
        'gaussian_target_h': target_h,
        'gaussian_target_h_plus_g': target_f,
        'rank1_contraction_difference': exact_mean,
        'arbitrary_amplification_note': 'multiplying g by any positive scalar leaves the finite transcript unchanged and scales the target gap'
      },
      'network_width':2,
      'relu_layers':3,
      'theorem_scope':'finite-node transcript no-go; does not exclude weight-aware identities or high-probability random-ensemble estimators'
    }
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))

if __name__=='__main__':main()

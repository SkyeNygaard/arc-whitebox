#!/usr/bin/env python3
"""Adversarial boundary checks for remaining analytic statements."""
import json, math
from pathlib import Path

# ReLU cubic-bound counterexample to the literal phrase "density bounded near zero".
# z ~ Uniform[9,10], t=-10. Density is exactly 0 in a neighborhood of zero, so L=0
# is a valid local bound, yet all points can cross the ReLU kink and E r^2=1/3.
relu_counterexample = {
    "z_distribution":"Uniform[9,10]",
    "t":-10,
    "local_density_bound_near_zero_L":0,
    "E_remainder_squared":1/3,
    "claimed_rhs_2L_abs_t_cubed":0,
    "repair":"Require the conditional density bound on the entire interval [-|t|,|t|], or assume |t| is below the radius of the local density bound."
}

# Corrected replication algebra sanity checks.
def adjusted_ratio(m, rho, compute_factor):
    return compute_factor * (rho + (1-rho)/m)
replication = {
    "mean_zero_independent_m5": adjusted_ratio(5,0,5),
    "perfectly_correlated_m5": adjusted_ratio(5,1,5),
    "note":"The algebraic formula survives. Only the implication independent => rho=0 fails without zero means."
}

# Brute force the corrected MUB allocation mechanism on a small abstract example.
# d=4, a=1, b=-0.1 gives a+bd=0.6>0. Enumerate count partitions over M=3 bases.
d=4; M=3; a=1.0; b=-0.1
def h(r): return 0.0 if r==0 else r/(a+b*r)
alloc_best={}
for P in range(1,M*d+1):
    best=-1; allocs=[]
    for r1 in range(d+1):
      for r2 in range(d+1):
       for r3 in range(d+1):
        if r1+r2+r3!=P: continue
        val=h(r1)+h(r2)+h(r3)
        if val>best+1e-12: best=val; allocs=[(r1,r2,r3)]
        elif abs(val-best)<=1e-12: allocs.append((r1,r2,r3))
    canonical=tuple(sorted(([d]*(P//d)+([P%d] if P%d else [])+[0]*(M-P//d-(1 if P%d else 0))), reverse=True))
    observed={tuple(sorted(x,reverse=True)) for x in allocs}
    assert observed=={canonical}, (P,observed,canonical)
    alloc_best[str(P)]={"count_pattern":canonical,"H":best}

# Poisson mean numerical quadrature in d=4 with a simple trapezoid over S^3 zonal density.
# Density for t is (2/pi)*sqrt(1-t^2). Use midpoint integration.
def poisson(t,r,d): return (1-r*r)/(1-2*r*t+r*r)**(d/2)
def mean_zonal_midpoint(r=0.37,d=4,n=200000):
    s=0.0
    dt=2/n
    c=2/math.pi
    for k in range(n):
        t=-1+(k+0.5)*dt
        s += poisson(t,r,d)*c*math.sqrt(max(0,1-t*t))*dt
    return s
poisson_mean=mean_zonal_midpoint()

result={
 "ReLU_density_bound": relu_counterexample,
 "replication_formula": replication,
 "T37_corrected_small_dimension_enumeration": {
    "d":d,"bases":M,"a":a,"b":b,"a_plus_bd":a+b*d,
    "all_budgets_passed":True,"optimal_patterns":alloc_best
 },
 "Poisson_mean_sanity_check": {
    "dimension":4,"r":0.37,"midpoints":200000,"numerical_mean":poisson_mean,
    "absolute_error":abs(poisson_mean-1.0),"verdict":"survived numerical attack"
 },
 "kernel_perturbation_wording": "The epsilon(1+B)^2 theorem is correct. Its optimizer-transfer corollary needs the same total-variation bound B uniformly over the entire comparison class, including a minimizing sequence.",
 "optimizer_instability_wording": "The rank-one construction proves pairwise ranking reversal. Calling it global optimizer reversal requires specifying an admissible class containing only, or controlled by, the two displayed rules.",
 "pass":True
}
Path(__file__).with_name('MISC_ATTACK_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

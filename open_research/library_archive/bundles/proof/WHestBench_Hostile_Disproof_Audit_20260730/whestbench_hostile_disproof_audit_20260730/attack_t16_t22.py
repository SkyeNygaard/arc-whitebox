#!/usr/bin/env python3
"""Independent endpoint attack plus recorded fast rerun results for T16/T22.

The expensive full Python primal/all-degree regeneration is intentionally not launched by
this script; those jobs exceeded the current execution window. The fast C++ and T22
commands were rerun separately and their outcomes are recorded below.
"""
import json
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import mpmath as mp
from pathlib import Path

HERE=Path(__file__).resolve().parent
C1=Path('/mnt/data/whestbench_closure_20260730')
cert=json.loads((C1/'T16_PRIMAL_DUAL_CERTIFICATE.json').read_text())
coeff=cert['hermite_coefficient_certificate']['coefficient_intervals']
lo=[Decimal(x[0]) for x in coeff]
hi=[Decimal(x[1]) for x in coeff]

mp.mp.dps=100
def kappa(t):
    return (mp.sqrt(1-t*t)+(mp.pi-mp.acos(t))*t)/mp.pi
km=mp.mpf(-1)
for _ in range(32):
    km=kappa(km)
Kminus=Decimal(mp.nstr(km,100))
Kplus=Decimal(1)

with localcontext() as ctx:
    ctx.prec=100; ctx.rounding=ROUND_FLOOR
    hp_lo=sum(lo,Decimal(0)); hm_lo=sum((lo[i] if i%2==0 else -hi[i]) for i in range(6))
with localcontext() as ctx:
    ctx.prec=100; ctx.rounding=ROUND_CEILING
    hp_hi=sum(hi,Decimal(0)); hm_hi=sum((hi[i] if i%2==0 else -lo[i]) for i in range(6))
# 1e-90 deliberately dwarfs plausible 100-digit evaluation roundoff while remaining
# far below the 2.2e-7 negative-endpoint margin. This is a robust high-precision attack,
# not a directed-interval proof.
eps=Decimal('1e-90')
rp_lo=Kplus-hp_hi; rp_hi=Kplus-hp_lo
rm_lo=(Kminus-eps)-hm_hi; rm_hi=(Kminus+eps)-hm_lo
assert rp_lo>0 and rm_lo>0

result={
 "T16_endpoint_attack": {
   "method":"100-digit independent mpmath recurrence plus certified coefficient intervals; not directed interval arithmetic",
   "K32_plus_1":str(Kplus),
   "h_plus_1_interval":[str(hp_lo),str(hp_hi)],
   "residual_plus_1_interval":[str(rp_lo),str(rp_hi)],
   "K32_minus_1_100digit":str(Kminus),
   "h_minus_1_interval":[str(hm_lo),str(hm_hi)],
   "robust_residual_minus_1_envelope":[str(rm_lo),str(rm_hi)],
   "verdict":"Endpoint equality counterexample not found. Both endpoint residuals are strictly positive. The prose's appeal to continuity alone was incomplete, but the claimed equality set survives this attack."
 },
 "fast_reruns_performed_separately": [
   {"command":"./t16_independent_cpp_audit","returncode":0,"elapsed_seconds":1.40,"result":"best degree 7 exact negative fraction"},
   {"command":"python arc_cubature_proof_v5_1/verify_theorem_package.py","returncode":0,"elapsed_seconds":27.32,"result":"one_sided_logic_verified=true"},
   {"command":"python arc_cubature_proof_v5_1/verify_manifest.py","returncode":0,"elapsed_seconds":0.54,"result":"manifest verified: 59 files"}
 ],
 "slow_rerun_status": {
   "prove_t16_all_degree.py":"Full fresh run exceeded this execution window; archived rerun output exists and the independent C++ finite/tail audit passed.",
   "prove_t16_primal_dual.py":"Full fresh run exceeded this execution window; no second arithmetic/interval implementation of the primal sixth-derivative/Hermite/Krawczyk step was found."
 },
 "verdict":"T22 survived the scoped attack. T16 was not disproved, but its primal computer-assisted step remains insufficiently independent for external release."
}
(HERE/'T16_T22_ATTACK_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

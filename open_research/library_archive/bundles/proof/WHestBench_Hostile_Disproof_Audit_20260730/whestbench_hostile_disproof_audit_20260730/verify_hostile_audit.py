#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess, sys
from decimal import Decimal
from pathlib import Path

HERE=Path(__file__).resolve().parent
scripts=['attack_t29.py','attack_t38.py','attack_information_replication.py','attack_misc_theorems.py','attack_t16_t22.py']
runs=[]
for name in scripts:
    p=subprocess.run([sys.executable,str(HERE/name)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    row={'script':name,'returncode':p.returncode,'stdout_tail':p.stdout[-300:],'stderr_tail':p.stderr[-300:]}
    runs.append(row)
    if p.returncode:
        raise SystemExit(json.dumps(row,indent=2))

t29=json.loads((HERE/'T29_ATTACK_RESULTS.json').read_text())
t38=json.loads((HERE/'T38_ATTACK_RESULTS.json').read_text())
info=json.loads((HERE/'INFORMATION_REPLICATION_ATTACK_RESULTS.json').read_text())
misc=json.loads((HERE/'MISC_ATTACK_RESULTS.json').read_text())
t16=json.loads((HERE/'T16_T22_ATTACK_RESULTS.json').read_text())

assert Decimal(t29['counterexample']['uniform_risk'])==0
assert Decimal(t29['counterexample']['nonuniform_risk'])==0
assert t38['association_values']['dimension_256']['A_minus_O_plus_d_times_O_minus_C']==0
assert abs(info['haar_orientation_counterexample']['conditional_mean_error']-0.5)<1e-15
assert info['replication_counterexample']['adjusted_ratio']==5.0
assert misc['ReLU_density_bound']['E_remainder_squared']>misc['ReLU_density_bound']['claimed_rhs_2L_abs_t_cubed']
assert misc['T37_corrected_small_dimension_enumeration']['all_budgets_passed']
assert Decimal(t16['T16_endpoint_attack']['robust_residual_minus_1_envelope'][0])>0

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

artifacts=['T29_ATTACK_RESULTS.json','T38_ATTACK_RESULTS.json','INFORMATION_REPLICATION_ATTACK_RESULTS.json','MISC_ATTACK_RESULTS.json','T16_T22_ATTACK_RESULTS.json','T29_CORRECTED_THEOREM.md','T38_CORRECTED_THEOREM.md','REQUIRED_PATCHES.md','CLAIM_SURVIVAL_MATRIX.md','WHESTBENCH_COMPLETE_PROOF_PACKAGE_HOSTILE_PATCHED.md']
cert={
 'status':'PASS',
 'scripts':runs,
 'confirmed_false_as_written':[
   'T29 free-mass every-minimizer uniqueness',
   'original broad T38 nondegeneracy implication',
   'Haar conditional no-value corollary with independence from runtime information alone',
   'independent-replica score-neutral corollary without mean-zero errors',
   'global cubic ReLU remainder bound from a merely local density bound'
 ],
 'surviving_core':['T22 scoped one-sided theorem','T16 core certificate subject to independent-primal limitation','T27/T37 strict-sign allocation theorem','T39/T40 scoped identities'],
 'sha256':{name:sha256(HERE/name) for name in artifacts}
}
(HERE/'HOSTILE_AUDIT_CERTIFICATE.json').write_text(json.dumps(cert,indent=2)+'\n')
print(json.dumps(cert,indent=2))

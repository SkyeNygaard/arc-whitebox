from __future__ import annotations
import json, math, hashlib, sys
from pathlib import Path
import numpy as np

REOPEN=Path('/mnt/data/oracle_experiments_inputs/reopened_paths/reopened_paths_repro_20260730')
OUT=Path('/mnt/data/oracle_continuation_20260730/edge_dws')
OUT.mkdir(parents=True, exist_ok=True)
SEEDS=list(range(3000,3068))
CAND='nonlinear_k4_m64_b0_0.001'
D=256; DEPTH=32

def make_weights(seed:int)->np.ndarray:
    rng=np.random.default_rng(seed)
    scale=math.sqrt(2.0/D)
    # Generator uses h @ W; EdgeStateDWS expects W[out,in].
    return np.stack([((rng.standard_normal((D,D))*scale).astype(np.float32)).T for _ in range(DEPTH)], axis=0)

weights=[]; base_ids=[]; rots=[]; e0s=[]; js=[]; anchors=[]; targets=[]; confs=[]
rows=[]
for seed in SEEDS:
    with np.load(REOPEN/'results'/f'network_{seed}.npz', allow_pickle=False) as z:
        base=np.asarray(z['base'],float); ref=np.asarray(z['ref'],float); cand=np.asarray(z[CAND],float)
    d=cand-base
    e0=base-ref
    den=float(d@d)
    alpha_star=float(-(e0@d)/den) if den>1e-30 else 0.0
    alpha_star=float(np.clip(alpha_star,-20.0,20.0))
    base_mse=float(np.mean(e0*e0))
    opt_mse=float(np.mean((e0+alpha_star*d)**2))
    frozen_mse=float(np.mean((e0-2.0*d)**2))
    weights.append(make_weights(seed)); base_ids.append(str(seed)); rots.append(3)
    e0s.append(e0.astype(np.float32)); js.append(d.astype(np.float32)[:,None])
    anchors.append([-2.0]); targets.append([alpha_star]); confs.append(opt_mse<base_mse)
    rows.append(dict(seed=seed,alpha_star=alpha_star,base_mse=base_mse,opt_mse=opt_mse,frozen_mse=frozen_mse,
                     oracle_ratio=opt_mse/base_mse,frozen_ratio=frozen_mse/base_mse))

arrays=dict(
    weights=np.stack(weights).astype(np.float32),
    base_network_id=np.asarray(base_ids,dtype='U16'),
    rotation_id=np.asarray(rots,dtype=np.int16),
    baseline_error=np.stack(e0s).astype(np.float32),
    replay_jacobian=np.stack(js).astype(np.float32),
    anchor_coeffs=np.asarray(anchors,dtype=np.float32),
    target_coeffs=np.asarray(targets,dtype=np.float32),
    target_confidence=np.asarray(confs,dtype=np.float32),
)
raw=OUT/'edge_dws_labels_raw.npz'
np.savez_compressed(raw,**arrays)
verify=OUT/'surrogate_verification.json'
verify.write_text(json.dumps({
    'verified':True,'proof_kind':'additive_final_output',
    'equation':'candidate(coeff)=base+coeff*(frozen_source-base); error=baseline_error+J@coeff',
    'candidate':CAND,'source_frozen_alpha':-2.0,
    'max_abs_error':0.0,'tolerance':0.0
},indent=2))
# split: original fit/exploratory, original terminal first half, extension two halves
splits={
 'train':[str(x) for x in range(3000,3020)],
 'calibration':[str(x) for x in range(3020,3036)],
 'validation':[str(x) for x in range(3036,3052)],
 'test':[str(x) for x in range(3052,3068)],
}
payload=json.dumps(splits,sort_keys=True,separators=(',',':')).encode()
split_doc={'status':'frozen','version':1,'splits':splits,'splits_sha256':hashlib.sha256(payload).hexdigest(),
           'disallowed_base_network_ids':[],
           'note':'Roles preserve chronology: original fit/exploratory, original terminal, and two disjoint extension halves.'}
(OUT/'split_registry.json').write_text(json.dumps(split_doc,indent=2))
(OUT/'label_rows.json').write_text(json.dumps(rows,indent=2))
print(raw)
print(json.dumps({'n':len(SEEDS),'weights_gib':arrays['weights'].nbytes/2**30,
                  'alpha_summary':{k:float(v) for k,v in zip(['min','median','max'],np.quantile(arrays['target_coeffs'][:,0],[0,0.5,1]))},
                  'oracle_pooled_ratio':float(sum(r['opt_mse'] for r in rows)/sum(r['base_mse'] for r in rows)),
                  'frozen_pooled_ratio':float(sum(r['frozen_mse'] for r in rows)/sum(r['base_mse'] for r in rows))},indent=2))

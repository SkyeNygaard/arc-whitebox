#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from scipy.linalg import hadamard
ROOT=Path(__file__).resolve().parent
ASSET=Path('/mnt/data/priority6_inputs/kerdock_mub5_seed3.npz')
sys.path.insert(0,str(ROOT))
import orientation_codebook_experiment as ex

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 z=np.load(ASSET);chirps=z['chirps'].astype(np.float64);rot=z['rotation'].astype(np.float64);H=hadamard(256,dtype=np.float64)/16
 bases=np.asarray([(H*c[None,:])@rot for c in chirps[:8]]+[rot])
 orth=float(np.max(np.abs(bases@np.swapaxes(bases,-1,-2)-np.eye(256))))
 cross=np.abs(bases[0]@bases[1:].transpose(0,2,1));mub=float(np.max(np.abs(cross-1/16)))
 rng=np.random.default_rng(7);mu=rng.normal(size=256);M=rng.normal(size=(256,256));M=M@M.T/256;m=mu.copy();L=rng.normal(size=(256,8));R=rng.normal(size=(256,8))
 zero=float(np.max(np.abs(ex.lower_anchor(mu,M,m,L,R))))
 prereg=ROOT/'VALIDATION_PREREGISTRATION.json';prereg_hash=(ROOT/'VALIDATION_PREREGISTRATION.sha256').read_text().split()[0]
 full=json.loads((ROOT/'raw_full'/'freeze_manifest.json').read_text());val=json.loads((ROOT/'raw_validation'/'freeze_manifest.json').read_text()) if (ROOT/'raw_validation'/'freeze_manifest.json').exists() else {}
 overlap=sorted(set(full['base_ids'])&set(val.get('base_ids',[])))
 # Exact deterministic comparison between two independently executed duplicate development cases.
 a=json.loads((ROOT/'raw_full'/'case_3680907127_0.json').read_text());b=json.loads((ROOT/'raw_v2'/'case_3680907127_0.json').read_text())
 numeric=[]
 for key in ('baseline_mse','baseline_mse_nc','y0_mse','truth_noise_mse'):numeric.append(abs(a[key]-b[key]))
 for i in range(8):numeric.append(np.max(np.abs(np.asarray(a['orientations'][i]['correction'])-np.asarray(b['orientations'][i]['correction']))))
 out={'asset':{'shape_chirps':list(chirps.shape),'shape_rotation':list(rot.shape),'rotation_orthogonality_max_abs':float(np.max(np.abs(rot@rot.T-np.eye(256)))),'sample_basis_orthogonality_max_abs':orth,'sample_mutual_unbiasedness_max_abs_error':mub},
 'algebra':{'lower_anchor_zero_when_mu_equals_m_max_abs':zero},
 'freeze':{'preregistration_sha256_actual':sha(prereg),'preregistration_sha256_recorded':prereg_hash,'matches':sha(prereg)==prereg_hash,'development_validation_base_overlap':overlap,'preregistration_mtime_ns':prereg.stat().st_mtime_ns,'first_validation_case_mtime_ns':min((p.stat().st_mtime_ns for p in (ROOT/'raw_validation').glob('case_*.json')),default=None)},
 'repeatability':{'max_numeric_difference_duplicate_case':float(max(numeric))},
 'source_hashes':{p.name:sha(p) for p in [ROOT/'orientation_codebook_experiment.py',ROOT/'orientation_codebook_validation.py',ROOT/'analyze_orientation_codebook.py',ROOT/'analyze_validation.py',ASSET]}}
 out['pass']=(out['asset']['rotation_orthogonality_max_abs']<1e-5 and orth<1e-5 and mub<1e-5 and zero<1e-12 and out['freeze']['matches'] and not overlap and out['repeatability']['max_numeric_difference_duplicate_case']<1e-12 and (out['freeze']['first_validation_case_mtime_ns'] is None or out['freeze']['preregistration_mtime_ns']<out['freeze']['first_validation_case_mtime_ns']))
 (ROOT/'VALIDATION_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

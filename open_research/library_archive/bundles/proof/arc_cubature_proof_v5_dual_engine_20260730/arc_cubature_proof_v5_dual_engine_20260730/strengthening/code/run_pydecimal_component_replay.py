#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
NAMES=['FORMAL_CERTIFICATE_D256_L32.json','FORMAL_SIGN_LOGIC_AUDIT.json','KERDOCK_MULTIPLICITY_PROOF.json','FORMAL_KERNEL_MEAN_D256_L32.json']
COMMANDS=[['assemble_formal_certificate.py'],['verify_sign_logic.py'],['generate_kerdock_multiplicity.py'],['formal_kernel_mean.py']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proof-dir',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 root=a.proof_dir.resolve(); shim=root/'strengthening/pydecimal_shim'
 with tempfile.TemporaryDirectory(prefix='pydecimal-proof-replay-') as td:
  run=Path(td)/'proof'; shutil.copytree(root,run,ignore=shutil.ignore_patterns('__pycache__','strengthening','FULL_ARTIFACT_MANIFEST.sha256','FULL_ARTIFACT_FILES.json','verify_full_artifact_manifest.py','HARDENING_NOTES.md'))
  for n in NAMES:(run/'results'/n).unlink(missing_ok=True)
  env=os.environ.copy();env['PYTHONPATH']=str(shim)
  logs=[]
  for cmd in COMMANDS:
   p=subprocess.run([sys.executable,*cmd],cwd=run,env=env,text=True,capture_output=True,check=True)
   logs.append({'command':[sys.executable,*cmd],'returncode':p.returncode,'stdout_tail':p.stdout[-500:]})
  rows={}
  for n in NAMES:
   orig=root/'results'/n;new=run/'results'/n
   rows[n]={'original_sha256':sha(orig),'replay_sha256':sha(new),'byte_identical':orig.read_bytes()==new.read_bytes()}
  assert all(v['byte_identical'] for v in rows.values())
  out={'title':'Pure-_pydecimal component regeneration','python':sys.version.splitlines()[0],'engine':'standard-library pure-Python _pydecimal via strengthening/pydecimal_shim/decimal.py','artifacts':rows,'commands':logs,'passed':True}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

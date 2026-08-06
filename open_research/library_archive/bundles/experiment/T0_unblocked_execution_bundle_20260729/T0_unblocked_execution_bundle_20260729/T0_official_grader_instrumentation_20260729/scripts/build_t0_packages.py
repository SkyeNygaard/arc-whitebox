from pathlib import Path
import shutil, json, hashlib, tarfile, datetime, re
SRC=Path('/mnt/data/t0_work/a43_bundle/packages/A43_exact_source')
OUT=Path('/mnt/data/T0_official_grader_instrumentation_20260729/packages')
OUT.mkdir(parents=True,exist_ok=True)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def write_manifest(d,desc):
 m=json.loads((SRC/'manifest.json').read_text())
 m['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 m['description']=desc
 m['files']=[{'name':n,'sha256':sha(d/n)} for n in ['estimator.py','fast_matmul.py','kerdock_mub5_seed3.npz']]
 (d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')

def tar_dir(d,out):
 with tarfile.open(out,'w:gz',format=tarfile.PAX_FORMAT) as tf:
  for p in sorted(d.iterdir()): tf.add(p,arcname=p.name,recursive=True)

def make_basis(k):
 name=f'A43_basis{k:03d}'
 d=OUT/name
 if d.exists(): shutil.rmtree(d)
 shutil.copytree(SRC,d)
 text=(d/'estimator.py').read_text()
 # constants and first layer are replaced deterministically.  For k<129 use
 # the literal prefix of the original row order: chirp bases 0..k-1.
 active=128 if k==129 else k
 include=(k==129)
 text=text.replace('_KERDOCK_BASES = 128',f'_KERDOCK_BASES = {active}')
 text=text.replace('_TOTAL_ROWS = 66_048',f'_TOTAL_ROWS = {k*512}')
 text=text.replace('weighted = self._chirps[:, :, None] * effective_weight[None, :, :]',
                   'weighted = self._chirps[:_KERDOCK_BASES, :, None] * effective_weight[None, :, :]')
 old='''        coordinate_rows = fnp.stack((radius * effective_weight, -radius * effective_weight), axis=1).reshape((-1, _WIDTH))\n        return fnp.maximum(fnp.concatenate((kerdock_rows, coordinate_rows), axis=0), 0.0)'''
 if include:
  new=old
 else:
  new='''        return fnp.maximum(kerdock_rows, 0.0)'''
 if old not in text: raise RuntimeError('first layer block not found')
 text=text.replace(old,new)
 text=text.replace('"""Kerdock 5-design with exact fused, weight-preencoded row streaming."""',
                   f'"""Frozen T0 A43 arm: {k} complete bases, literal prefix, no correction."""')
 (d/'estimator.py').write_text(text)
 write_manifest(d,f'T0.1 frozen A43 arm with {k} complete 512-row bases; literal prefix of original basis order; no statistical correction.')
 tar_dir(d,OUT/f'{name}.tar.gz')
 return d

def make_copy(srcname,name,desc):
 src=Path('/mnt/data/t0_work/a43_bundle/packages')/srcname
 d=OUT/name
 if d.exists(): shutil.rmtree(d)
 shutil.copytree(src,d)
 # rebuild manifest against copied source, preserving its source manifest fields
 m=json.loads((d/'manifest.json').read_text());m['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['description']=desc
 m['files']=[{'name':n,'sha256':sha(d/n)} for n in ['estimator.py','fast_matmul.py','kerdock_mub5_seed3.npz']]
 (d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 tar_dir(d,OUT/f'{name}.tar.gz')
 return d

for k in [129,96,64,32]: make_basis(k)
make_copy('production_partial_tree_source','production_baseline','T0.3 frozen production partial-tree baseline source.')
make_copy('A42_exact_source','A42','T0.3 frozen A42 cached-right full-depth stream source.')
make_copy('A43_exact_source','A43','T0.3 frozen A43 fused full-depth stream source.')
# T0.2 exact null-work calibration arm: 64 tracked fp32 256x256 matmuls,
# followed by two elementwise operations that add exact zeros to first_weight.
d=make_copy('A43_exact_source','A43_delta64','T0.2 A43 plus exact legal null-work calibration delta: 64 fp32 256x256 matmuls and two 256x256 elementwise ops.')
p=d/'estimator.py'; text=p.read_text()
needle='''        first_weight = mlp.weights[0].astype(fnp.float32)\n        prepared_weights = tuple('''
repl='''        first_weight = mlp.weights[0].astype(fnp.float32)\n        # T0.2 preregistered tracked-operation calibration.  The zero matrix\n        # makes this output-preserving bit-for-bit; all work remains on\n        # flopscope arrays.  Expected delta under flopscope 0.9.1:\n        # 64*(2*256^3) + 2*256^2 = 2,147,614,720 operations.\n        calibration_zero = fnp.zeros((_WIDTH, _WIDTH), dtype=fnp.float32)\n        calibration_probe = first_weight @ calibration_zero\n        for _calibration_i in range(63):\n            calibration_probe = calibration_probe + (first_weight @ calibration_zero)\n        first_weight = first_weight + calibration_probe * 0.0\n        prepared_weights = tuple('''
if needle not in text: raise RuntimeError('predict insertion point missing')
p.write_text(text.replace(needle,repl))
write_manifest(d,'T0.2 A43 plus exact legal null-work calibration delta: 64 fp32 256x256 matmuls, 63 accumulation adds, one multiply-by-zero, and one final add. Expected delta 2,151,743,488 operations if every accumulation is billed; core matmul delta 2,147,483,648.')
tar_dir(d,OUT/'A43_delta64.tar.gz')
# note: exact delta including accumulation is 64*33554432 + 63*65536 + 65536 + 65536
exact=64*(2*256**3)+63*256**2+2*256**2
(OUT/'CALIBRATION_DELTA.json').write_text(json.dumps({'matmuls':64,'matmul_flops':64*(2*256**3),'accumulation_adds':63,'final_elementwise_ops':2,'expected_total_delta_flops':exact},indent=2)+'\n')
# hashes
rows=[]
for p in sorted(OUT.glob('*.tar.gz')): rows.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
(OUT/'PACKAGE_HASHES.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))

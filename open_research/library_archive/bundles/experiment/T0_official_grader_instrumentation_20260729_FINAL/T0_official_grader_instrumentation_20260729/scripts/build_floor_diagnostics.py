from pathlib import Path
import shutil,json,hashlib,tarfile,datetime
SRC=Path('/mnt/data/t0_work/a43_bundle/packages/A43_exact_source');OUT=Path('/mnt/data/T0_official_grader_instrumentation_20260729/packages')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for k in [16,20]:
 d=OUT/f'A43_diagnostic_basis{k:03d}'
 if d.exists():shutil.rmtree(d)
 shutil.copytree(SRC,d);p=d/'estimator.py';t=p.read_text()
 t=t.replace('_KERDOCK_BASES = 128',f'_KERDOCK_BASES = {k}').replace('_TOTAL_ROWS = 66_048',f'_TOTAL_ROWS = {k*512}')
 t=t.replace('weighted = self._chirps[:, :, None] * effective_weight[None, :, :]','weighted = self._chirps[:_KERDOCK_BASES, :, None] * effective_weight[None, :, :]')
 old='''        coordinate_rows = fnp.stack((radius * effective_weight, -radius * effective_weight), axis=1).reshape((-1, _WIDTH))\n        return fnp.maximum(fnp.concatenate((kerdock_rows, coordinate_rows), axis=0), 0.0)'''
 t=t.replace(old,'        return fnp.maximum(kerdock_rows, 0.0)')
 t=t.replace('"""Kerdock 5-design with exact fused, weight-preencoded row streaming."""',f'"""T0 local-only floor diagnostic: {k} complete prefix bases."""')
 p.write_text(t)
 m=json.loads((d/'manifest.json').read_text());m['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['description']=f'T0 local-only scoring-floor diagnostic, {k} complete prefix bases; not allocated an official submission.';m['files']=[{'name':n,'sha256':sha(d/n)} for n in ['estimator.py','fast_matmul.py','kerdock_mub5_seed3.npz']];(d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 out=OUT/f'A43_diagnostic_basis{k:03d}.tar.gz'
 with tarfile.open(out,'w:gz') as tf:
  for q in sorted(d.iterdir()):tf.add(q,arcname=q.name)
 print(k,sha(out))

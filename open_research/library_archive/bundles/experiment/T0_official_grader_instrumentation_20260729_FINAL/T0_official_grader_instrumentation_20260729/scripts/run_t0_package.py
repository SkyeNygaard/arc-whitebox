import argparse, importlib.util, sys, types, numpy as np, time, json, hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--seed',type=int,default=51000);ap.add_argument('--out',required=True);a=ap.parse_args()
class BaseEstimator: pass
class SetupContext:
 def __init__(self,d):self.submission_dir=d
class MLP:
 def __init__(self,w):self.width=256;self.depth=32;self.weights=w
fl=types.ModuleType('flopscope');fl.numpy=np;sys.modules['flopscope']=fl;sys.modules['flopscope.numpy']=np
wh=types.ModuleType('whestbench');wh.BaseEstimator=BaseEstimator;wh.SetupContext=SetupContext;sys.modules['whestbench']=wh
dom=types.ModuleType('whestbench.domain');dom.MLP=MLP;sys.modules['whestbench.domain']=dom
pkg=Path(a.package);sys.path.insert(0,str(pkg));sys.modules.pop('fast_matmul',None)
spec=importlib.util.spec_from_file_location('estimator',pkg/'estimator.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(a.seed);weights=list((rng.standard_normal((32,256,256))*np.sqrt(2/256)).astype(np.float32))
e=m.Estimator();e.setup(SetupContext(pkg));t=time.perf_counter();y=e.predict(MLP(weights),272_000_000_000);sec=time.perf_counter()-t
v=np.asarray(y[-1],dtype=np.float64); rec={'package':pkg.name,'seed':a.seed,'seconds':sec,'final_digest':hashlib.sha256(np.ascontiguousarray(v).view(np.uint8)).hexdigest(),'final_norm':float(np.linalg.norm(v)),'final_sum':float(v.sum()),'shape':list(y.shape)}
Path(a.out).write_text(json.dumps(rec,indent=2));np.save(Path(a.out).with_suffix('.npy'),v);print(json.dumps(rec))

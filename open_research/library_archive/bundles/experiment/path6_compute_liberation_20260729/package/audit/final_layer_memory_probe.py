import sys, math, time, resource, json
import numpy as np
from compute_liberation_experiment import N_POINTS, WIDTH, partial_tree_matmul
mode=sys.argv[1]
chunk=int(sys.argv[2]) if len(sys.argv)>2 else N_POINTS
rng=np.random.default_rng(29)
h=np.maximum(rng.normal(0.15,0.7,size=(N_POINTS,WIDTH)),0).astype(np.float32)
w=(rng.standard_normal((WIDTH,WIDTH),dtype=np.float32)/math.sqrt(WIDTH)).astype(np.float32)
t=time.perf_counter()
if mode=='materialize':
 z=partial_tree_matmul(h,w)
 out=np.maximum(z,0).astype(np.float64).mean(axis=0)
elif mode=='direct':
 total=np.zeros(WIDTH,dtype=np.float64)
 for s in range(0,N_POINTS,chunk):
  z=partial_tree_matmul(h[s:min(s+chunk,N_POINTS)],w)
  total += np.maximum(z,0).astype(np.float64).sum(axis=0)
 out=total/N_POINTS
else: raise ValueError(mode)
print(json.dumps({'mode':mode,'chunk_rows':chunk,'elapsed_s':time.perf_counter()-t,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'checksum':float(out.sum())}))

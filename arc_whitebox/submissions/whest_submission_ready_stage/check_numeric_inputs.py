#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--weights-dir',type=Path,required=True);p.add_argument('--results-json',type=Path,required=True);a=p.parse_args()
r=json.loads(a.results_json.read_text());ids=sorted(set(r['valid_ids']+r['test_ids']))
for idx in ids:
 w=np.load(a.weights_dir/f'mlp_{idx:05d}.npy')
 print(json.dumps({'id':idx,'dtype':str(w.dtype),'finite':bool(np.isfinite(w).all()),'min':float(np.nanmin(w)),'max':float(np.nanmax(w)),'rms':float(np.sqrt(np.mean(w.astype(np.float64)**2))) }))

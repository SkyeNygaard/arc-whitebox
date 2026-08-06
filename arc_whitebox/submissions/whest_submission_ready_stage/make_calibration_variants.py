#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('layerwise',type=Path);p.add_argument('--global-output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.layerwise.read_text())
depth=len(r['x1_scale'])
g={'x1_scale':[r['global_x1_scale']]*depth,'x1a_scale':[r['global_x1a_scale']]*depth,
   'source':str(a.layerwise),'mode':'global','global_x1_scale':r['global_x1_scale'],'global_x1a_scale':r['global_x1a_scale']}
a.global_output.parent.mkdir(parents=True,exist_ok=True);a.global_output.write_text(json.dumps(g,indent=2));print(a.global_output)

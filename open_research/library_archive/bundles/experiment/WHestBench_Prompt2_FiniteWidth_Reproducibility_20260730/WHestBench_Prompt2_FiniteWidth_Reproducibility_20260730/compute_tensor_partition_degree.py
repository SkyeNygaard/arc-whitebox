#!/usr/bin/env python3
import sys,json
from pathlib import Path
sys.path.insert(0,'/mnt/data')
import prompt2_full_hermite_core as c
from prompt2_tensor_partition_core import TensorEnergy
rank=int(sys.argv[1]);degree=int(sys.argv[2]);support=int(sys.argv[3]) if len(sys.argv)>3 else None
t=TensorEnergy(rank)
x=c.round_interval_outward(t.energy(degree,support),60)
d={'rank':rank,'degree':degree,'max_support':support,'lo':f'{x.lo.numerator}/{x.lo.denominator}','hi':f'{x.hi.numerator}/{x.hi.denominator}','bounds':c.decimal_bounds(x,50)}
suf=f'_s{support}' if support is not None else ''
out=Path(f'/mnt/data/tensor{rank}_degree_{degree}{suf}.json');out.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))

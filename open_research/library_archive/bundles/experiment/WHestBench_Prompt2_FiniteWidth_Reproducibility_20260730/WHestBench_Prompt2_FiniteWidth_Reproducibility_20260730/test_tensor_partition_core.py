import json,sys
sys.path.insert(0,'/mnt/data')
import prompt2_full_hermite_core as c
from prompt2_tensor_partition_core import TensorEnergy
for rank,maxd in [(3,4),(4,4),(5,3),(6,3)]:
 t=TensorEnergy(rank)
 for n in range(maxd+1):
  x=c.round_interval_outward(t.energy(n),60)
  path=f'/mnt/data/tensor{rank}_degree_{n}.json'
  d=json.load(open(path))
  def q(s):
   a,b=s.split('/');return c.Fraction(int(a),int(b))
  old=c.I(q(d['lo']),q(d['hi']))
  ok=x.lo==old.lo and x.hi==old.hi
  print(rank,n,ok,c.decimal_bounds(x,30),c.decimal_bounds(old,30))
  assert ok

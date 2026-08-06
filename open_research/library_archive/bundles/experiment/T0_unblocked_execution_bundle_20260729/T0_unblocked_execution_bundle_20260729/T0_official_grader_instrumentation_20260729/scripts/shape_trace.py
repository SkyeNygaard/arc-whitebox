import sys, types, importlib.util, math, json
from dataclasses import dataclass

class Counter:
 def __init__(self): self.total=0; self.by={}
 def add(self,name,n): self.total+=int(n); self.by[name]=self.by.get(name,0)+int(n)
C=Counter()

def rate(dtype): return 2 if dtype=='float64' else 1
@dataclass
class A:
 shape: tuple
 dtype: str='float32'
 def __getitem__(self,key):
  # support ellipsis, slices, ints
  if not isinstance(key,tuple): key=(key,)
  # expand ellipsis
  key=list(key)
  if Ellipsis in key:
   i=key.index(Ellipsis); missing=len(self.shape)-(len(key)-1); key=key[:i]+[slice(None)]*missing+key[i+1:]
  key += [slice(None)]*(len(self.shape)-len(key))
  out=[]
  for dim,k in zip(self.shape,key):
   if isinstance(k,int): continue
   if isinstance(k,slice):
    start,stop,step=k.indices(dim); out.append(max(0,(stop-start+(step-1))//step))
   else: raise NotImplementedError(k)
  return A(tuple(out),self.dtype)
 def reshape(self,shape):
  shape=list(shape); prod=1; neg=None
  for i,x in enumerate(shape):
   if x==-1: neg=i
   else: prod*=x
  old=math.prod(self.shape)
  if neg is not None: shape[neg]=old//prod
  return A(tuple(shape),self.dtype)
 def astype(self,dtype):
  d='float64' if '64' in str(dtype) else 'float32'; C.add('astype',math.prod(self.shape)*rate(d)); return A(self.shape,d)
 def _bin(self,o,name):
  sh=broadcast(self.shape,o.shape if isinstance(o,A) else ())
  C.add(name,math.prod(sh)*rate(self.dtype)); return A(sh,self.dtype)
 def __add__(self,o): return self._bin(o,'add')
 def __radd__(self,o): return self._bin(o,'add')
 def __sub__(self,o): return self._bin(o,'sub')
 def __rsub__(self,o): return self._bin(o,'sub')
 def __mul__(self,o): return self._bin(o,'mul')
 def __rmul__(self,o): return self._bin(o,'mul')
 def __truediv__(self,o): return self._bin(o,'div')
 def __neg__(self): C.add('neg',math.prod(self.shape)*rate(self.dtype)); return A(self.shape,self.dtype)
 def __matmul__(self,o):
  # batch broadcasting
  batch=broadcast(self.shape[:-2],o.shape[:-2]); m,k=self.shape[-2:]; k2,n=o.shape[-2:]; assert k==k2,(self.shape,o.shape)
  sh=batch+(m,n); C.add('matmul',2*math.prod(batch or (1,))*m*k*n*rate(self.dtype)); return A(sh,self.dtype)

def broadcast(a,b):
 n=max(len(a),len(b)); aa=(1,)*(n-len(a))+a; bb=(1,)*(n-len(b))+b; out=[]
 for x,y in zip(aa,bb):
  assert x==y or x==1 or y==1,(a,b); out.append(max(x,y))
 return tuple(out)

def stack(xs,axis=0):
 xs=list(xs); sh=list(xs[0].shape); axis=axis if axis>=0 else len(sh)+1+axis; sh.insert(axis,len(xs)); C.add('stack',math.prod(sh)*rate(xs[0].dtype)); return A(tuple(sh),xs[0].dtype)
def concatenate(xs,axis=0):
 xs=list(xs); sh=list(xs[0].shape); axis=axis if axis>=0 else len(sh)+axis; sh[axis]=sum(x.shape[axis] for x in xs); C.add('concatenate',math.prod(sh)*rate(xs[0].dtype)); return A(tuple(sh),xs[0].dtype)
def block(grid):
 # only 2d grids of arrays, possibly 4x4
 rows=[]
 for row in grid:
  h=row[0].shape[-2]; w=sum(x.shape[-1] for x in row); rows.append((h,w))
 sh=grid[0][0].shape[:-2]+(sum(h for h,w in rows),rows[0][1]); C.add('block',math.prod(sh)*rate(grid[0][0].dtype)); return A(sh,grid[0][0].dtype)
def maximum(a,b): C.add('maximum',math.prod(a.shape)*rate(a.dtype)); return A(a.shape,a.dtype)
def sum_(a,axis=None):
 axes=(axis,) if isinstance(axis,int) else tuple(axis or range(len(a.shape))); axes=tuple(x if x>=0 else len(a.shape)+x for x in axes)
 sh=tuple(d for i,d in enumerate(a.shape) if i not in axes); C.add('sum',math.prod(a.shape)*rate(a.dtype)); return A(sh,a.dtype)
def sqrt(a): C.add('sqrt',math.prod(a.shape)*rate(a.dtype)); return A(a.shape,a.dtype)
def zeros(shape): return A(tuple(shape) if isinstance(shape,tuple) else (shape,),'float64')
def load(*a,**k): raise RuntimeError

mod=types.ModuleType('flopscope.numpy'); mod.ndarray=A; mod.float32='float32'; mod.float64='float64'; mod.stack=stack; mod.concatenate=concatenate; mod.block=block; mod.maximum=maximum; mod.sum=sum_; mod.sqrt=sqrt; mod.zeros=zeros; mod.load=load
fl=types.ModuleType('flopscope'); fl.numpy=mod
sys.modules['flopscope']=fl;sys.modules['flopscope.numpy']=mod
spec=importlib.util.spec_from_file_location('fm','/mnt/data/t0_work/a43_bundle/packages/A43_exact_source/fast_matmul.py'); fm=importlib.util.module_from_spec(spec);sys.modules['fast_matmul']=fm;spec.loader.exec_module(fm)

def first_layer(k):
 # k basis ids: k<=128 chirps, at 129 includes coordinate. For trace full.
 chirp=min(k,128); coord=1 if k==129 else 0
 w=A((256,256)); rot=A((256,256)); eff=rot@w
 if chirp:
  weighted=A((chirp,256,1))*eff # broadcast
  v=weighted
  span=1
  while span<256:
   g=v.reshape((chirp,256//(2*span),2,span,256)); left=g[:,:,0,:,:]; right=g[:,:,1,:,:]
   v=stack((left+right,left-right),axis=2).reshape((chirp,256,256)); span*=2
  pre=v*(1.0)
  kr=stack((pre,-pre),axis=2).reshape((-1,256))
 else: kr=None
 if coord:
  cr=stack((1.0*eff,-1.0*eff),axis=1).reshape((-1,256))
  design=concatenate((kr,cr),axis=0)
 else: design=kr
 return maximum(design,0.0)

def trace(k):
 global C; C=Counter()
 # weights astype and prepare 31
 first=A((256,256)).astype('float32')
 ps=[]
 for i in range(31): ps.append(fm.prepare_right_p3_d5(A((256,256)).astype('float32')))
 a=first_layer(k)
 total=None
 for s in range(k):
  e=fm.first_layer_chunk_to_relu_encoding(a[s*512:(s+1)*512],ps[0])
  for p in ps[1:-1]: e=fm.encoded_chunk_to_relu_encoding(e,p)
  cs=fm.encoded_chunk_to_final_sum(e,ps[-1]); total=cs if total is None else total+cs
  
 final=total/(k*512)
 firstmean=sqrt(sum_(first*first,axis=0))*0.3989
 rows=[zeros(256) for _ in range(32)]; rows[0]=firstmean; rows[-1]=final; out=stack(rows,axis=0)
 return C.total,C.by
for k in [1,32,64,96,129]:
 t,b=trace(k); print(k,t, json.dumps(dict(sorted(b.items(),key=lambda x:-x[1])),indent=2))

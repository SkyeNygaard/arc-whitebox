from __future__ import annotations
import argparse,json,math,sys,time,gc
from pathlib import Path
import numpy as np, torch
from scipy.stats import norm
SRC=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code')
sys.path.insert(0,str(SRC))
import frozen_reference_impl as fr
from base_nested_impl import lower_anchor, estimate
D=fr.D;TARGET=fr.TARGET

def design(seed):
 radius=fr.chi_mean(D);H=fr.walsh_hadamard()/math.sqrt(D);R=fr.haar_rotation(seed);blocks=[]
 for u in range(128):
  b=(H*fr.kerdock_chirp(u)[None,:])@R;blocks += [(radius*b).astype(np.float32),(-radius*b).astype(np.float32)]
 c=(radius*R).astype(np.float32);blocks += [c,-c]
 return np.concatenate(blocks)

def forward_capture(x,ws):
 with torch.no_grad():
  zt=None;ht=None
  for li,w in enumerate(ws):
   z=x@w;x=torch.relu(z)
   if li==TARGET:zt=z.clone();ht=x.clone()
 return zt,ht,x

def ref(ws,n,nid,chunk):
 r1=fr.stream_reference(ws,n,13_000_000+2*nid,chunk);r2=fr.stream_reference(ws,n,13_000_001+2*nid,chunk)
 return r1,r2,{k:.5*(r1[k]+r2[k]) for k in r1}

def center_estimates(Z,m):
 mu=Z.mean(0);x=Z-mu;v=np.mean(x*x,0);sd=np.sqrt(np.maximum(v,1e-30));t=mu/sd
 k3=np.mean(x**3,0);k4=np.mean(x**4,0)-3*v*v
 g=mu*norm.cdf(t)+sd*norm.pdf(t)
 e3=g-(k3/6)*t*norm.pdf(t)/(sd*sd)
 e4=e3+(k4/24)*((t*t-1)*norm.pdf(t))/(sd**3)
 # winsorize only catastrophic marginal expansions, using a broad same-design scale envelope.
 lo=np.quantile(m,.001)-4*np.std(m);hi=np.quantile(m,.999)+4*np.std(m)
 return ({'sample':m,'gaussian':np.clip(g,lo,hi),'edgeworth3':np.clip(e3,lo,hi),'edgeworth4':np.clip(e4,lo,hi)}, {'mean_abs_skew':float(np.mean(np.abs(k3/sd**3))),'mean_abs_excess':float(np.mean(np.abs(k4/sd**4)))})

def mse(a,b):return float(np.mean((a-b)**2))
def run(nid,rots,truth_n,chunk,outdir):
 ws,_,_=fr.make_weights(nid);r1,r2,pool=ref(ws,truth_n,nid,chunk);truth=pool['y'];rho=fr.chi_mean(D)
 for rot in rots:
  t=time.time();zt,ht,yt=forward_capture(torch.from_numpy(design(rot)),ws);Z=zt.double().numpy();H=ht.double().numpy();Y=yt.double().numpy();base=Y.mean(0);m=H.mean(0)
  Q=fr.sample_anchor_matrix(H,m,rho);idx,dirs=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,idx,dirs,rho);fit=fr.fit_crossfit(X,Y);sample_anchor=X.mean(0)
  pair=(D/(rho*rho))*(H.T@H/len(H));ests,diag=center_estimates(Z,m);ests['oracle']=pool['mu']
  bm=mse(base,truth);rows={}
  for name,cen in ests.items():
   a=lower_anchor(m,cen,pair,idx,dirs)
   full=estimate(fit,sample_anchor+a)
   corr=full-base
   item={'center_relerr':float(np.linalg.norm(cen-pool['mu'])/max(np.linalg.norm(pool['mu']),1e-30)),'corr_norm':float(np.linalg.norm(corr)),'alpha':{}}
   for alpha in [.05,.1,.2,.3,.4,.5,.6,.75,1.0]:
    mm=mse(base+alpha*corr,truth);item['alpha'][str(alpha)]={'ratio':mm/bm,'mse':mm}
   e=base-truth;astar=-float(e@corr)/max(float(corr@corr),1e-30);item['oracle_alpha']=astar;item['oracle_scalar_ratio']=mse(base+np.clip(astar,-2,2)*corr,truth)/bm;rows[name]=item
  # small blends from sample center toward analytic centers.
  for src in ['gaussian','edgeworth3','edgeworth4']:
   for lam in [.1,.25,.5,.75]:
    cen=m+lam*(ests[src]-m);a=lower_anchor(m,cen,pair,idx,dirs);corr=estimate(fit,sample_anchor+a)-base
    rows[f'{src}_centerblend{lam:g}']={'alpha':{'1.0':{'ratio':mse(base+corr,truth)/bm,'mse':mse(base+corr,truth)}},'corr_norm':float(np.linalg.norm(corr)),'center_relerr':float(np.linalg.norm(cen-pool['mu'])/max(np.linalg.norm(pool['mu']),1e-30))}
  out={'network_id':nid,'rot':rot,'truth_n_each':truth_n,'base_mse':bm,'reference_noise':mse(r1['y'],r2['y'])/4,'diagnostics':diag,'rows':rows,'seconds':time.time()-t}
  (outdir/f'n{nid}_r{rot}.json').write_text(json.dumps(out,indent=2));print(json.dumps({'network':nid,'rot':rot,'base':bm,'best':min(v['alpha']['1.0']['ratio'] if '1.0' in v['alpha'] else min(x['ratio'] for x in v['alpha'].values()) for v in rows.values()),'e4half':rows['edgeworth4']['alpha']['0.5']['ratio']}),flush=True);del Z,H,Y,zt,ht,yt;gc.collect()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=7300);ap.add_argument('--networks',type=int,default=2);ap.add_argument('--rots',type=int,nargs='+',default=[3,11,97]);ap.add_argument('--truth-n',type=int,default=4096);ap.add_argument('--chunk',type=int,default=4096);ap.add_argument('--outdir',type=Path,default=Path('/mnt/data/competition_relevance_20260730/edgeworth_radial_center'));a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
 for n in range(a.start,a.start+a.networks):run(n,a.rots,a.truth_n,a.chunk,a.outdir)
if __name__=='__main__':main()

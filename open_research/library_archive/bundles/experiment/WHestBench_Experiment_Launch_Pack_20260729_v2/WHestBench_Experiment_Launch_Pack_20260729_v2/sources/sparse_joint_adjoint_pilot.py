from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,'/mnt/data/arc_research')
import sparse_adjoint_control as s
import sparse_crossfit_lower_fast as f
from sparse_lower_moment_pilot import make_pilot,forward_target,lower_matrix
D=s.D;TARGET=29;PROBES=32;RIDGE=.1

def run(net,truth_n=65536,g_n=4096,lower_grid=(0,.01,.02,.05),conn_grid=(0,.05,.1,.2,.33)):
    t=time.time();ws=s.a.make_weights(51000+net);xk=s.make_kerdock();ht,Yt,km,kc,kg=s.collect_all(xk,ws,TARGET)
    H=ht.double().cpu().numpy();Y=Yt.double().cpu().numpy();m=H.mean(0);rho=s.chi_mean(D);Q=s.sample_anchor_matrix(H,m,rho)
    ix=np.argsort(np.linalg.norm(Q,axis=1))[::-1][:PROBES];rr=Q[ix].copy();rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),1e-30);V=rr.T
    X=s.radial_features(H,m,ix,V,rho);qa=np.einsum('ij,ij->i',Q[ix],rr);Zk=H-m;sampleC=np.mean((Zk[:,ix]**2)*(Zk@V),axis=0);stats=s.fit_fold_stats(X,Y,6,RIDGE)
    # Adjoint-connected contraction, antithetic source integration.
    B=s.backward_maps(ws,kg,TARGET);zh=s.a.sobol_normal((g_n+1)//2,1810000+net).double().cpu().numpy();Z=np.concatenate([zh,-zh],axis=0)[:g_n]
    src=[s.gaussian_source(km[l],kc[l],B[l],ix,V,Z) for l in range(TARGET+1)];predC=np.sum(src,axis=0);dc=(predC-sampleC)/(D+1)
    # One independent Kerdock basis for only lower moments.
    hp=forward_target(make_pilot(4,1),ws,TARGET).double().cpu().numpy();mp=hp.mean(0);Mp=(hp.T@hp/len(hp))*(D/(rho*rho));Lp=lower_matrix(mp,Mp,m);dl=np.einsum('ij,ij->i',Lp[ix],rr)
    anchors={}
    for al in lower_grid:
      for ac in conn_grid: anchors[f'l{al:g}_c{ac:g}']=qa+al*dl+ac*dc
    preds={k:s.estimate(stats,a) for k,a in anchors.items()}
    y1=f.final_mean_stream(ws,truth_n,1820000+2*net);y2=f.final_mean_stream(ws,truth_n,1820001+2*net);truth=.5*(y1+y2);base=s.estimate(stats,qa);bm=float(np.mean((base-truth)**2));ms={k:float(np.mean((v-truth)**2)) for k,v in preds.items()}
    return {'network':net,'baseline_mse':bm,'truth_noise_mse':float(.5*np.mean((y1-y2)**2)),'mse':ms,'ratio':{k:v/bm for k,v in ms.items()},'fixed_candidate':'l0.02_c0.1','delta_norms':{'lower':float(np.linalg.norm(dl)),'connected':float(np.linalg.norm(dc)),'cos':float(np.dot(dl,dc)/max(np.linalg.norm(dl)*np.linalg.norm(dc),1e-30))},'indices':ix.tolist(),'runtime':time.time()-t}

def main():
 p=argparse.ArgumentParser();p.add_argument('--nets',nargs='+',type=int,required=True);p.add_argument('--truth-n',type=int,default=65536);p.add_argument('--g-n',type=int,default=4096);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(8);rs=[]
 for n in a.nets:
  r=run(n,a.truth_n,a.g_n);rs.append(r);print(n,'fixed',round(r['ratio'][r['fixed_candidate']],4),'best',min(r['ratio'].items(),key=lambda z:z[1]),'sec',round(r['runtime'],1),flush=True)
 ks=rs[0]['mse'];sm={k:{'aggregate':sum(r['mse'][k] for r in rs)/sum(r['baseline_mse'] for r in rs),'mean':float(np.mean([r['ratio'][k] for r in rs])),'median':float(np.median([r['ratio'][k] for r in rs])),'wins':sum(r['ratio'][k]<1 for r in rs),'worst':max(r['ratio'][k] for r in rs)} for k in ks};a.out.write_text(json.dumps({'records':rs,'summary':sm,'fixed_candidate':'l0.02_c0.1'},indent=2));print('FIXED',json.dumps(sm['l0.02_c0.1'],indent=2))
if __name__=='__main__':main()

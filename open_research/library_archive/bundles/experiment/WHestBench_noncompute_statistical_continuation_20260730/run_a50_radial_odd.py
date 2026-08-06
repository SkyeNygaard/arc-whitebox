from __future__ import annotations
import argparse,json,math,sys,time,gc
from pathlib import Path
import numpy as np, torch
SRC=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code')
sys.path.insert(0,str(SRC))
import frozen_reference_impl as fr
D=fr.D;TARGET=fr.TARGET

def make_kerdock_rot(seed):
 radius=fr.chi_mean(D);H=fr.walsh_hadamard()/math.sqrt(D);R=fr.haar_rotation(seed);blocks=[]
 for u in range(128):
  basis=(H*fr.kerdock_chirp(u)[None,:])@R;blocks += [(radius*basis).astype(np.float32),(-radius*basis).astype(np.float32)]
 coordinate=(radius*R).astype(np.float32);blocks += [coordinate,-coordinate]
 return np.concatenate(blocks)

def prepare_ref(ws,n,network_id,chunk):
 r1=fr.stream_reference(ws,n,12_000_000+2*network_id,chunk);r2=fr.stream_reference(ws,n,12_000_001+2*network_id,chunk)
 return r1,r2,{k:.5*(r1[k]+r2[k]) for k in r1}

def orient_rows(U):
 U=U.copy()
 for j in range(len(U)):
  k=int(np.argmax(np.abs(U[j])));s=1. if U[j,k]>=0 else -1.;U[j]*=s
 return U

def run_case(network_id,rot,xk,ws,r1,r2,pooled,outdir):
 t=time.time();hk,yk=fr.forward_target_final(torch.from_numpy(xk),ws);H=hk.double().numpy();Y=yk.double().numpy();m=H.mean(0);base=Y.mean(0);rho=fr.chi_mean(D)
 Q=fr.sample_anchor_matrix(H,m,rho);indices,directions=fr.sample_row_probes(Q);X=fr.radial_features_sample_rows(H,m,indices,directions,rho)
 sample_M=(D/(rho*rho))*(H.T@H/len(H));sample_raw=((D+1)/(rho*rho))*((H*H).T@H/len(H))
 comp=fr.anchor_component_matrices(pooled['mu'],pooled['M'],pooled['raw'],m,sample_M,sample_raw)
 anchors={'sample_anchor':fr.contract_rows(Q,indices,directions)};anchors.update({k:fr.contract_rows(v,indices,directions) for k,v in comp.items()})
 fit=fr.fit_crossfit(X,Y);pred={k:fr.estimate_from_fit(fit,a)[0] for k,a in anchors.items()};oracle=pred['complete_exact']-base
 fw=fit['fold_sizes']/fit['fold_sizes'].sum();beta=np.einsum('f,fpd->pd',fw,fit['betas']);_,s,vt=np.linalg.svd(beta,full_matrices=False);U=orient_rows(vt[:4])
 coeff=U@oracle;proj=coeff@U;truth=.5*(r1['y']+r2['y']);err=base-truth;err_coeff=U@err;outside_sq=max(float(err@err-err_coeff@err_coeff),0.0);base_mse=np.mean(err**2);oracle_mse=np.mean((err+.5*oracle)**2);proj_mse=(outside_sq+float(np.sum((err_coeff+.5*coeff)**2)))/D
 # Canonical orientation-odd and even features for each mode.
 Wlast=ws[-1].double().numpy();sens=np.linalg.norm(Wlast,axis=0);features=[]
 for j,u in enumerate(U):
  features.append({
   'mode':j,'target':float(coeff[j]),'sv':float(s[j]),
   'odd_base':float(u@base),'odd_center':float(u@m),'odd_sens':float(u@sens),
   'odd_center_weighted':float(u@(m*sens)),'odd_sample_anchor':float(u@(pred['sample_anchor']-base)),
   'even_base_norm':float(np.linalg.norm(base)),'even_center_norm':float(np.linalg.norm(m)),
   'even_sens_norm':float(np.linalg.norm(sens)),'even_sv':float(s[j]),
  })
 out={'network_id':network_id,'rot':rot,'base_mse':float(base_mse),'oracle_half_ratio':float(oracle_mse/base_mse),'rank4_half_ratio':float(proj_mse/base_mse),'oracle_norm':float(np.linalg.norm(oracle)),'rank4_capture':float(np.dot(proj,proj)/max(np.dot(oracle,oracle),1e-30)),'error_coeff':[float(x) for x in err_coeff],'outside_error_sq':outside_sq,'features':features,'seconds':time.time()-t}
 (outdir/f'n{network_id}_r{rot}.json').write_text(json.dumps(out,indent=2));del H,Y,hk,yk,X,Q;gc.collect();return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--networks',type=int,default=6);ap.add_argument('--start',type=int,default=7000);ap.add_argument('--truth-n',type=int,default=8192);ap.add_argument('--chunk',type=int,default=4096);ap.add_argument('--rots',type=int,nargs='+',default=[3,11,97]);ap.add_argument('--outdir',type=Path,default=Path('/mnt/data/competition_relevance_20260730/a50_radial_odd'));a=ap.parse_args();a.outdir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(min(8,torch.get_num_threads()))
 designs={r:make_kerdock_rot(r) for r in a.rots};rows=[]
 for nid in range(a.start,a.start+a.networks):
  ws,_,_=fr.make_weights(nid);r1,r2,pool=prepare_ref(ws,a.truth_n,nid,a.chunk)
  for r in a.rots:
   row=run_case(nid,r,designs[r],ws,r1,r2,pool,a.outdir);rows.append(row);print(json.dumps({'done':len(rows),'network':nid,'rot':r,'oracle_half_ratio':row['oracle_half_ratio'],'rank4_half_ratio':row['rank4_half_ratio']}),flush=True)
 (a.outdir/'all_rows.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()

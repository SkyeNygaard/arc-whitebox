from __future__ import annotations
import argparse,json,time,random,gc
from pathlib import Path
import numpy as np, torch
from torch import nn
from src.contracts import load_bundle
from src.edge_dws import EdgeStateDWS
from src.metrics import evaluate

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True,warn_only=True)

def pred(model,a,idx,dev):
 model.eval(); cs=[];ss=[];fs=[]
 with torch.no_grad():
  for i in idx:
   w=torch.from_numpy(a['weights'][i:i+1].astype(np.float32,copy=False)).to(dev)
   no=torch.from_numpy(a['node_observables'][i:i+1].astype(np.float32,copy=False)).to(dev)
   lo=torch.from_numpy(a['layer_observables'][i:i+1].astype(np.float32,copy=False)).to(dev)
   o=model(w,no,lo);cs.append(o.correction.cpu().numpy()[0]);ss.append(float(o.scale.cpu()[0]));fs.append(float(o.confidence.cpu()[0]))
 return np.asarray(cs),np.asarray(ss),np.asarray(fs)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--state',type=Path,default=Path('results/run_001/train_state.pt'));args=ap.parse_args()
 cfg=json.load(open('frozen_config.json'));seed_all(int(cfg['seed']))
 b=load_bundle(Path('inputs/frozen_labels.npz'),Path('inputs/frozen_label_manifest.json'),Path('inputs/canonical_split_registry.json'));a=b.arrays;dev=torch.device('cpu')
 m=EdgeStateDWS(depth=32,label_dim=1,node_obs_dim=a['node_observables'].shape[-1],layer_obs_dim=a['layer_observables'].shape[-1],edge_channels=8,node_channels=8,token_channels=48,passes=1,transformer_heads=4).to(dev)
 opt=torch.optim.AdamW(m.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
 epoch0=0;best=float('inf');best_state=None;hist=[]
 if args.state.exists():
  st=torch.load(args.state,map_location='cpu',weights_only=False);m.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);epoch0=st['epoch'];best=st['best'];best_state=st['best_state'];hist=st['history']
 train=b.splits['train'];cal=b.splits['calibration'];rng=np.random.default_rng(cfg['seed']+epoch0)
 for ep in range(epoch0,min(epoch0+args.epochs,cfg['training']['epochs'])):
  m.train();order=rng.permutation(train);opt.zero_grad(set_to_none=True);losses=[];t=time.time();acc=int(cfg['training']['gradient_accumulation'])
  for step,i in enumerate(order):
   w=torch.from_numpy(a['weights'][i:i+1].astype(np.float32,copy=False));no=torch.from_numpy(a['node_observables'][i:i+1].astype(np.float32,copy=False));lo=torch.from_numpy(a['layer_observables'][i:i+1].astype(np.float32,copy=False))
   e0=torch.from_numpy(a['baseline_error'][i:i+1].astype(np.float32,copy=False));j=torch.from_numpy(a['replay_jacobian'][i:i+1].astype(np.float32,copy=False));anchor=torch.from_numpy(a['anchor_coeffs'][i:i+1].astype(np.float32,copy=False));target=torch.from_numpy(a['target_coeffs'][i:i+1].astype(np.float32,copy=False))
   o=m(w,no,lo);total=anchor+o.correction;err=e0+torch.einsum('bod,bd->bo',j,total);replay=err.square().mean();resid=target-anchor;aux=(o.correction-resid).square().mean();benefit=((e0+torch.einsum('bod,bd->bo',j,target)).square().mean(1)<e0.square().mean(1)).float();cl=nn.functional.binary_cross_entropy(o.confidence,benefit);loss=replay+cfg['training']['aux_coefficient_weight']*aux+cfg['training']['confidence_weight']*cl;(loss/acc).backward()
   if (step+1)%acc==0 or step+1==len(order):nn.utils.clip_grad_norm_(m.parameters(),cfg['training']['grad_clip']);opt.step();opt.zero_grad(set_to_none=True)
   losses.append(float(loss.detach())); del w,no,lo,e0,j,anchor,target,o,total,err,replay,resid,aux,benefit,cl,loss; gc.collect()
  pc,_,cf=pred(m,a,cal,dev);coeff=a['anchor_coeffs'][cal]+pc;met=evaluate(a,cal,coeff,cf,175.5,175.5);cm=met['candidate_raw_mse']
  if cm<best:best=cm;best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  hist.append({'epoch':ep+1,'train_loss':float(np.mean(losses)),'calibration_raw_mse':cm,'calibration_gain':met['raw_gain_baseline_over_candidate'],'seconds':time.time()-t});epoch0=ep+1
  args.state.parent.mkdir(parents=True,exist_ok=True);torch.save({'epoch':epoch0,'model':m.state_dict(),'optimizer':opt.state_dict(),'best':best,'best_state':best_state,'history':hist},args.state)
  print(json.dumps(hist[-1]),flush=True)
 print(json.dumps({'epoch':epoch0,'best_calibration_mse':best,'complete':epoch0>=cfg['training']['epochs']}))
if __name__=='__main__':main()

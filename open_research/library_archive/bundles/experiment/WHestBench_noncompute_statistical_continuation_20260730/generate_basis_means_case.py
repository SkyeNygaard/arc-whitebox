from __future__ import annotations
import argparse,math,sys
from pathlib import Path
import numpy as np, torch
SRC=Path('/mnt/data/work/T4_legal_layer31_anchor_hedge_20260729_review/T4_legal_layer31_anchor_hedge_20260729/code');sys.path.insert(0,str(SRC));import frozen_reference_impl as fr
D=fr.D
def design(seed):
 r=fr.chi_mean(D);H=fr.walsh_hadamard()/math.sqrt(D);R=fr.haar_rotation(seed);bs=[]
 for u in range(128):
  b=(H*fr.kerdock_chirp(u)[None,:])@R;bs += [(r*b).astype(np.float32),(-r*b).astype(np.float32)]
 c=(r*R).astype(np.float32);bs += [c,-c];return np.concatenate(bs)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--network',type=int,required=True);ap.add_argument('--rot',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(4);ws,_,_=fr.make_weights(a.network);x=torch.from_numpy(design(a.rot));
 with torch.no_grad():
  for w in ws:x=torch.relu(x@w)
 B=x.double().numpy().reshape(129,2,256,D).mean((1,2));a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,B=B)
if __name__=='__main__':main()

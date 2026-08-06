from pathlib import Path
import json, glob
import numpy as np

ROOT=Path('/mnt/data/continued_review_inputs/oracle_gap_full/whest_experiments_oracle_gap_20260730/results')
OUT=Path('/mnt/data/continued_review_outputs')
OUT.mkdir(exist_ok=True)

def spectrum(U):
    # rows = incremental correction vectors
    G=U@U.T
    ev=np.linalg.eigvalsh(G)[::-1]
    ev=np.maximum(ev,0)
    tr=ev.sum()
    frac=ev/tr if tr>0 else ev
    er=(tr*tr/np.sum(ev*ev)) if np.sum(ev*ev)>0 else 0
    rank90=int(np.searchsorted(np.cumsum(frac),.9)+1) if tr>0 else 0
    rank95=int(np.searchsorted(np.cumsum(frac),.95)+1) if tr>0 else 0
    den=np.sqrt(np.outer(np.diag(G),np.diag(G)))
    cos=np.divide(G,den,out=np.zeros_like(G),where=den>0)
    off=cos.copy(); np.fill_diagonal(off,0)
    return {
        'effective_rank':float(er),
        'rank90':rank90,'rank95':rank95,
        'eigenvalue_fractions':frac.tolist(),
        'increment_energy_fractions':(np.diag(G)/np.trace(G)).tolist(),
        'max_abs_offdiag_cosine':float(np.max(np.abs(off))),
        'mean_abs_offdiag_cosine':float(np.sum(np.abs(off))/(off.size-len(off))),
        'cosine':cos.tolist(),
    }

def load_case(p):
    with np.load(p,allow_pickle=False) as z:
        C=z['checkpoint_corrections'].astype(float)
        depths=z['checkpoint_depths'].astype(int).tolist()
    U=np.diff(np.vstack([np.zeros((1,C.shape[1])),C]),axis=0)
    return depths,U

allout={'definition':'Effective rank and spectral energy of six successive checkpoint-repair increments. Per-case uses R^256 output vectors; per-base concatenates the three predetermined rotations into R^768.','splits':{}}
rows=[]
for split in ['development','validation','confirmation']:
    files=sorted((ROOT/split).glob('seed_*/*.npz'))
    cases=[]; byseed={}
    for p in files:
        depths,U=load_case(p)
        seed=int(p.parent.name.split('_')[-1]); rot=int(p.stem.split('rot')[-1])
        s=spectrum(U)
        rec={'seed':seed,'rotation':rot,**{k:v for k,v in s.items() if k!='cosine'}}
        cases.append(rec); byseed.setdefault(seed,[]).append((rot,U))
    bases=[]
    for seed, vals in sorted(byseed.items()):
        vals=sorted(vals)
        # same increments, concatenate output vectors across the 3 rotations
        Ucat=np.concatenate([u for _,u in vals],axis=1)
        s=spectrum(Ucat)
        bases.append({'seed':seed,'rotations':[r for r,_ in vals],**{k:v for k,v in s.items() if k!='cosine'},'cosine':s['cosine']})
    def summary(items):
        out={}
        for k in ['effective_rank','rank90','rank95','max_abs_offdiag_cosine','mean_abs_offdiag_cosine']:
            x=np.array([r[k] for r in items],float)
            out[k]={'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max()),'values':x.tolist()}
        out['fraction_effective_rank_ge_3']=float(np.mean([r['effective_rank']>=3 for r in items]))
        out['fraction_rank90_ge_3']=float(np.mean([r['rank90']>=3 for r in items]))
        return out
    allout['splits'][split]={'n_cases':len(cases),'n_bases':len(bases),'depths':depths,'case_summary':summary(cases),'base_summary':summary(bases),'per_case':cases,'per_base':bases}
    for b in bases:
        rows.append([split,b['seed'],b['effective_rank'],b['rank90'],b['rank95'],b['max_abs_offdiag_cosine'],b['mean_abs_offdiag_cosine'],*b['eigenvalue_fractions']])

json_path=OUT/'WITHIN_NETWORK_CHECKPOINT_RANK.json'
json_path.write_text(json.dumps(allout,indent=2))
csv_path=OUT/'WITHIN_NETWORK_CHECKPOINT_RANK_PER_BASE.csv'
with csv_path.open('w') as f:
    f.write('split,seed,effective_rank,rank90,rank95,max_abs_offdiag_cosine,mean_abs_offdiag_cosine,eigfrac1,eigfrac2,eigfrac3,eigfrac4,eigfrac5,eigfrac6\n')
    for r in rows:f.write(','.join(map(str,r))+'\n')
print(json.dumps({s:{'case':allout['splits'][s]['case_summary'],'base':allout['splits'][s]['base_summary']} for s in allout['splits']},indent=2))
print(json_path,csv_path)

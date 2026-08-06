from __future__ import annotations
import json,glob,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/whest_reopened/analytic_companion_control_20260729/analytic_companion_control_20260729')
OUT=Path('/mnt/data/whest_reopened/reopened_path_results/downstream_rescore');OUT.mkdir(parents=True,exist_ok=True)

GROUPS={
 'train_dev':(ROOT/'records_dev',range(0,16)),
 'calib_dev':(ROOT/'records_dev',range(16,24)),
 'validation':(ROOT/'records_validation',range(100,108)),
 'expansion':(ROOT/'records_expansion',range(300,308)),
 'rotation':(ROOT/'records_rotation',range(200,204)),
 'rotation_expansion':(ROOT/'records_rotation_expansion',range(400,408)),
}

def truth_for(d,group,case):
    hp=None
    if group=='validation': hp=ROOT/'highref_validation'/f'case_{case:03d}.npz'
    elif group=='expansion': hp=ROOT/'highref_expansion'/f'case_{case:03d}.npz'
    if hp is not None and hp.exists():
        z=np.load(hp);a=np.asarray(z['half1'],float);b=np.asarray(z['half2'],float);return .5*(a+b),a,b,'highref'
    a=np.asarray(d['truth_half1'],float);b=np.asarray(d['truth_half2'],float)
    return .5*(a+b),a,b,'record'

def arms(block):
    out={}
    out['analytic_total']=(np.asarray(block['analytic_anchor']['total'],float),np.asarray(block['analytic_output']['total'],float))
    out['analytic_mean']=(np.asarray(block['analytic_anchor']['mean'],float),np.asarray(block['analytic_output']['mean'],float))
    out['analytic_cov']=(np.asarray(block['analytic_anchor']['cov_residual'],float),np.asarray(block['analytic_output']['cov_residual'],float))
    for k,v in block['companion_anchor'].items():out[f'companion_{k}']=(np.asarray(v,float),np.asarray(block['companion_output'][k],float))
    out['paired']=(np.asarray(block['paired_anchor'],float),np.asarray(block['paired_output'],float))
    return out

def load_records():
    rows=[]
    for group,(folder,cases) in GROUPS.items():
        for case in cases:
            p=folder/f'case_{case:03d}.json'
            if not p.exists():continue
            d=json.load(open(p));truth,ta,tb,ts=truth_for(d,group,case)
            for design in ['full129','reduced112']:
                bl=d[design];base=np.asarray(bl['baseline_prediction'],float);beta=np.asarray(bl['beta'],float)
                err=truth-base
                # Minimum-norm coefficient vector whose downstream map best matches the true output error.
                G=beta@beta.T;scale=max(np.trace(G)/len(G),1e-30)
                ao=np.linalg.solve(G+1e-10*scale*np.eye(len(G)),beta@err)
                oo=ao@beta
                sample=np.asarray(bl['sample_anchor'],float)
                sample_out=sample@beta
                for arm,(a,o_stored) in arms(bl).items():
                    o=a@beta
                    store_gap=float(np.linalg.norm(o-o_stored)/max(np.linalg.norm(o),1e-30))
                    rows.append(dict(group=group,case=case,seed=d['seed'],design=design,arm=arm,
                      truth=truth,ta=ta,tb=tb,base=base,beta=beta,anchor=a,output=o,oracle_anchor=ao,oracle_output=oo,
                      sample_anchor=sample,sample_output=sample_out,truth_source=ts,stored_output_gap=store_gap))
    return rows

def fit_alpha(rs):
    num=sum(float(np.dot(r['output'],r['truth']-r['base'])) for r in rs)
    den=sum(float(np.dot(r['output'],r['output'])) for r in rs)
    return num/max(den,1e-30)

def summarize(rs,alpha):
    br=[];cr=[];bc=[];cc=[];dw=[];uw=[];cos=[];ip=[];og=[];sg=[]
    per=[]
    for r in rs:
        e=r['truth']-r['base'];pred=r['base']+alpha*r['output']
        brv=np.mean((r['base']-r['truth'])**2);crv=np.mean((pred-r['truth'])**2)
        bcv=np.mean((r['base']-r['ta'])*(r['base']-r['tb']))
        ccv=np.mean((pred-r['ta'])*(pred-r['tb']))
        br.append(brv);cr.append(crv);bc.append(bcv);cc.append(ccv)
        da=r['anchor']-r['oracle_anchor'];do=da@r['beta'];oo=r['oracle_output']
        dw.append(np.dot(do,do)/max(np.dot(oo,oo),1e-30))
        uw.append(np.dot(da,da)/max(np.dot(r['oracle_anchor'],r['oracle_anchor']),1e-30))
        c=alpha*r['output'];cos.append(np.dot(c,e)/max(np.linalg.norm(c)*np.linalg.norm(e),1e-30));ip.append(np.dot(c,e)/len(e))
        og.append(np.mean((r['base']+r['oracle_output']-r['truth'])**2)/max(brv,1e-30))
        sg.append(r['stored_output_gap'])
        per.append(dict(case=r['case'],ratio=crv/max(brv,1e-30),cross_ratio=ccv/max(bcv,1e-30),downstream_anchor_error_ratio=dw[-1],unweighted_anchor_error_ratio=uw[-1],cosine=cos[-1]))
    br=np.array(br);cr=np.array(cr);bc=np.array(bc);cc=np.array(cc)
    rat=cr/br
    return dict(n=len(rs),pooled_raw_ratio=float(cr.sum()/br.sum()),mean_raw_ratio=float(rat.mean()),wins=int(np.sum(rat<1)),p90=float(np.quantile(rat,.9)),worst=float(rat.max()),pooled_cross_ratio=float(cc.sum()/bc.sum()),
      mean_downstream_anchor_error_ratio=float(np.mean(dw)),median_downstream_anchor_error_ratio=float(np.median(dw)),
      mean_unweighted_anchor_error_ratio=float(np.mean(uw)),median_unweighted_anchor_error_ratio=float(np.median(uw)),
      mean_correction_cosine=float(np.mean(cos)),mean_error_correction_inner_product=float(np.mean(ip)),
      oracle_span_pooled_ratio=float(np.sum(np.array(og)*br)/br.sum()),max_stored_output_relative_gap=float(max(sg)),per_case=per)

def main():
    rows=load_records(); keys=sorted(set((r['design'],r['arm']) for r in rows))
    summary={'protocol':{'alpha_fit':'train_dev cases 0-15 only','calibration':'cases 16-23 reported but not used to refit','external_groups':['validation','expansion','rotation','rotation_expansion'],'truth':'highref used when available'},'arms':{}}
    csv=['design,arm,alpha,group,n,pooled_raw_ratio,mean_raw_ratio,wins,p90,worst,pooled_cross_ratio,mean_downstream_anchor_error_ratio,mean_unweighted_anchor_error_ratio,mean_cosine']
    for design,arm in keys:
        tr=[r for r in rows if r['design']==design and r['arm']==arm and r['group']=='train_dev']
        alpha=float(np.clip(fit_alpha(tr),-4,4))
        ent={'alpha':alpha,'groups':{}}
        for group in GROUPS:
            rs=[r for r in rows if r['design']==design and r['arm']==arm and r['group']==group]
            if not rs:continue
            z=summarize(rs,alpha);ent['groups'][group]=z
            csv.append(','.join(map(str,[design,arm,alpha,group,z['n'],z['pooled_raw_ratio'],z['mean_raw_ratio'],z['wins'],z['p90'],z['worst'],z['pooled_cross_ratio'],z['mean_downstream_anchor_error_ratio'],z['mean_unweighted_anchor_error_ratio'],z['mean_correction_cosine']])))
        summary['arms'][f'{design}/{arm}']=ent
    # Rank by pooled validation+expansion (fixed alpha), never by protected/official data.
    rank=[]
    for k,e in summary['arms'].items():
        vals=[]
        for g in ['validation','expansion']:
            if g in e['groups']:vals.append((e['groups'][g]['pooled_raw_ratio'],e['groups'][g]['n']))
        rank.append((sum(x*n for x,n in vals)/sum(n for x,n in vals),k))
    summary['rank_validation_expansion']=rank=sorted(rank)
    # Explicit reopening diagnostic: high unweighted coefficient error but useful output correction.
    diag=[]
    for k,e in summary['arms'].items():
        for g,z in e['groups'].items():
            if g in ['validation','expansion','rotation','rotation_expansion'] and z['pooled_raw_ratio']<1 and z['mean_unweighted_anchor_error_ratio']>z['mean_downstream_anchor_error_ratio']*2:
                diag.append({'arm':k,'group':g,'raw_ratio':z['pooled_raw_ratio'],'unweighted':z['mean_unweighted_anchor_error_ratio'],'downstream':z['mean_downstream_anchor_error_ratio']})
    summary['directional_reopen_examples']=diag
    (OUT/'DOWNSTREAM_RESCORE_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    (OUT/'DOWNSTREAM_RESCORE_ROWS.csv').write_text('\n'.join(csv)+'\n')
    print(json.dumps({'records':len(rows),'best':rank[:8],'directional_examples':len(diag)},indent=2))

if __name__=='__main__':main()

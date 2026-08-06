#!/usr/bin/env python3
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SRC=Path('/mnt/data/new_unresolved_questions/source_material/TARGET_FREE_LAYER31_SUPPORT_HIGHREF_RESULTS.json')
d=json.load(open(SRC))
records=d['records']; fams=d['protocol']['selector_families']; ks=[str(x) for x in d['protocol']['k_values']]

def agg(rows, chooser):
    B=sum(x['baseline_mse'] for x in rows); F=sum(x['full_oracle_mse'] for x in rows)
    C=sum(chooser(x) for x in rows)
    return {'candidate_over_baseline':C/B,'gap_capture':(B-C)/(B-F),'raw_gain':B/C,'baseline_sum':B,'candidate_sum':C,'full_oracle_sum':F}

out={'protocol':d['protocol'],'per_rotation':{},'pooled':{},'support_relationships':{},'theorem_interpretation':{}}
for rot in d['protocol']['rotations']:
    rows=[x for x in records if x['rotation_seed']==rot]
    rr={}
    for k in ks:
        rr[k]={}
        for f in fams:
            rr[k][f]=agg(rows,lambda x,f=f,k=k:x['selectors'][f][k]['mse'])
        rr[k]['oracle_best_family_per_record']=agg(rows,lambda x,k=k:min(x['selectors'][f][k]['mse'] for f in fams))
        # Oracle union support: use union of indices from the four named selectors, but amplitude is only available
        # indirectly for each selected support, so no union replay is claimed.
    out['per_rotation'][str(rot)]=rr
# pooled rotations
for k in ks:
    out['pooled'][k]={f:agg(records,lambda x,f=f,k=k:x['selectors'][f][k]['mse']) for f in fams}
    out['pooled'][k]['oracle_best_family_per_record']=agg(records,lambda x,k=k:min(x['selectors'][f][k]['mse'] for f in fams))
# Same-network cross-rotation support overlaps and capture changes.
by={(x['network'],x['rotation_seed']):x for x in records}
for f in fams:
    out['support_relationships'][f]={}
    for k in ks:
        vals=[]; cap_pairs=[]
        for net in d['protocol']['networks']:
            a=by[(net,d['protocol']['rotations'][0])];b=by[(net,d['protocol']['rotations'][1])]
            A=set(a['selectors'][f][k]['indices']);B=set(b['selectors'][f][k]['indices'])
            vals.append(len(A&B)/int(k))
            cap_pairs.append([a['selectors'][f][k]['full_oracle_gap_capture'],b['selectors'][f][k]['full_oracle_gap_capture']])
        diffs=[abs(a-b) for a,b in cap_pairs]
        out['support_relationships'][f][k]={
            'mean_overlap_fraction':sum(vals)/len(vals),'median_overlap_fraction':statistics.median(vals),'min_overlap_fraction':min(vals),
            'mean_absolute_per_network_capture_change':sum(diffs)/len(diffs),'median_absolute_per_network_capture_change':statistics.median(diffs),
            'capture_pairs':cap_pairs}
# Cross-family overlap at k=32, to see whether radial and PCA are effectively same menu.
for rot in d['protocol']['rotations']:
    vals=[]
    for net in d['protocol']['networks']:
        x=by[(net,rot)];A=set(x['selectors']['radial_h3_row_sens']['32']['indices']);B=set(x['selectors']['basis_pca12_sens']['32']['indices'])
        vals.append({'network':net,'intersection':len(A&B),'jaccard':len(A&B)/len(A|B),'overlap_fraction_of_32':len(A&B)/32})
    out.setdefault('cross_family_radial_vs_pca_k32',{})[str(rot)]={
        'mean_intersection':sum(v['intersection'] for v in vals)/len(vals),
        'mean_jaccard':sum(v['jaccard'] for v in vals)/len(vals),
        'mean_overlap_fraction':sum(v['overlap_fraction_of_32'] for v in vals)/len(vals),'rows':vals}
# Marginal oracle capture increments by K for best two stable families.
for rot in d['protocol']['rotations']:
    for f in ['radial_h3_row_sens','basis_pca12_sens']:
        caps=[out['per_rotation'][str(rot)][k][f]['gap_capture'] for k in ks]
        out.setdefault('marginal_capture',{}).setdefault(str(rot),{})[f]={
            'K':[int(k) for k in ks],'capture':caps,'increments':[caps[0]]+[caps[i]-caps[i-1] for i in range(1,len(caps))]}
# Direct upper-bound conclusions.
best19=out['per_rotation']['19']['32']['oracle_best_family_per_record']['gap_capture']
bestpooled=out['pooled']['32']['oracle_best_family_per_record']['gap_capture']
out['theorem_interpretation']={
 'support_capacity_principle':'For a frozen support S, the exact-amplitude replay is an upper bound on every deployable coefficient policy whose correction is restricted to S. Failure of this oracle support gate closes all amplitude-learning descendants on that support.',
 'four_family_menu_rotation19_ceiling':best19,
 'four_family_menu_pooled_ceiling':bestpooled,
 'rotation19_gate_half_passed':best19>=.5,
 'strongest_honest_conclusion':'The tested target-free selectors identify reproducible support, but the selected 32-coordinate subspace does not uniformly contain half of the full layer-31 oracle repair. Stable index selection and signed repair capacity are distinct properties.'}
(ROOT/'TARGET_FREE_SUPPORT_CAPACITY_ANALYSIS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'rotation3_k32':out['per_rotation']['3']['32'],'rotation19_k32':out['per_rotation']['19']['32'],'pooled_k32':out['pooled']['32'],'interpretation':out['theorem_interpretation']},indent=2))

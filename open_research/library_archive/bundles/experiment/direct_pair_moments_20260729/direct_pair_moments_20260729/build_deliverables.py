from __future__ import annotations
import csv, hashlib, json, math, os
from pathlib import Path
import numpy as np

OUT=Path('/mnt/data/direct_pair_moments_20260729')
raw=json.load(open(OUT/'raw/PAIR_COMPANION_ABLATION_24_COMBINED.json'))
hi=json.load(open(OUT/'raw/PAIR_MOMENT_ABLATION_HIGHREF8.json'))
lo=json.load(open(OUT/'raw/LOWER_STRUCTURE_RESULTS_HIGHREF8.json'))
R=raw['records']; S=raw['summary']

def arm(r,p,bc,m,a): return r['probe_results'][str(p)]['basis_results'][str(bc)]['methods'][m]['alphas'][str(a)]
def agg(p,bc,m,a): return S['candidates'][f'p{p}_b{bc}_{m}_a{a}']
def unbiased(p,bc,m,a):
    return sum(arm(r,p,bc,m,a)['unbiased_mse'] for r in R)/sum(r['baseline_unbiased_mse'] for r in R)
def bootstrap(p,bc,m,a,B=20000):
    rng=np.random.default_rng(20260729+p+bc+int(100*a))
    bs=np.array([r['baseline_mse'] for r in R]); ms=np.array([arm(r,p,bc,m,a)['mse'] for r in R]); vals=[]
    for _ in range(B):
        ix=rng.integers(0,len(R),len(R)); vals.append(ms[ix].sum()/bs[ix].sum())
    return [float(x) for x in np.quantile(vals,[.025,.975])]
def delta_boot(p,bc,a,B=20000):
    rng=np.random.default_rng(991+p+bc+int(100*a)); bs=np.array([r['baseline_mse'] for r in R])
    c=np.array([arm(r,p,bc,'companion_pairs',a)['mse'] for r in R]); s=np.array([arm(r,p,bc,'primary_pairs',a)['mse'] for r in R]); vals=[]
    for _ in range(B):
        ix=rng.integers(0,len(R),len(R)); vals.append((c[ix].sum()-s[ix].sum())/bs[ix].sum())
    return float((c.sum()-s.sum())/bs.sum()),[float(x) for x in np.quantile(vals,[.025,.975])]

def pair_stats(p,bc):
    ds=[r['probe_results'][str(p)]['basis_results'][str(bc)]['decomposition'] for r in R]
    out={}
    for k in ds[0]:
        a=np.array([d[k] for d in ds]); out[k]={'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max()),'mean':float(a.mean())}
    return out

arms=[]
for p,bc,a in [(32,129,.5),(32,16,.1),(128,129,.5),(128,129,.2),(128,16,.1)]:
 for m in ['companion_pairs','primary_pairs','companion_diag_primary_row','primary_diag_companion_row']:
    g=agg(p,bc,m,a)
    arms.append({'probes':p,'companion_bases':bc,'shrinkage':a,'pair_source':m,'raw_ratio':g['pooled_ratio'],'raw_ci95':bootstrap(p,bc,m,a),'noise_corrected_ratio':unbiased(p,bc,m,a),'wins':g['wins'],'n':24,'median':g['median'],'p90':g['p90'],'worst':g['worst']})
summary={
 'status':'development reanalysis on an exposed 24-network prospective companion-validation block plus an exposed high-reference 8-network oracle pair substitution audit',
 'primary_decision':'close independent pair-moment estimation; retain fused same-cloud selected contractions',
 'high_reference_pair_substitution':{
   'exact_pairs_ratio':hi['summary']['exact']['pooled_ratio'],
   'primary_sample_pairs_ratio':hi['summary']['sample_both']['pooled_ratio'],
   'absolute_ratio_difference':hi['summary']['sample_both']['pooled_ratio']-hi['summary']['exact']['pooled_ratio'],
   'exact_diag_sample_row_ratio':hi['summary']['exact_q_sample_t']['pooled_ratio'],
   'sample_diag_exact_row_ratio':hi['summary']['sample_q_exact_t']['pooled_ratio'],
 },
 'new_24_network_ablation':{
   'arms':arms,
   'pair_source_deltas':{
     'p128_b129_a0.5':dict(zip(['point','ci95'],delta_boot(128,129,.5))),
     'p128_b129_a0.2':dict(zip(['point','ci95'],delta_boot(128,129,.2))),
     'p128_b16_a0.1':dict(zip(['point','ci95'],delta_boot(128,16,.1))),
   },
   'pair_increment_geometry':{f'p{p}_b{bc}':pair_stats(p,bc) for p in [32,128] for bc in [16,129]},
 },
 'rank_audit':{
   'pooled_shared_r90':lo['training_universal_spectrum']['r90'],
   'pooled_shared_r95':lo['training_universal_spectrum']['r95'],
   'pooled_shared_r99':lo['training_universal_spectrum']['r99'],
   'local_rank2_median_anchor_relative_error':lo['summary']['local_rank2']['median_anchor_rel_error'],
   'local_rank2_pooled_ratio':lo['summary']['local_rank2']['pooled_ratio'],
   'local_exact_pooled_ratio':lo['summary']['exact']['pooled_ratio'],
   'local_rank2_energy_median':float(np.median([x['local_energy']['2'] for x in lo['records']])),
 },
 'gate_assessment':{
   'development_ratio_below_0.75':'pass only for full 129-basis companion center at 128 probes and alpha 0.5 (post-hoc extension)',
   'promotion_ratio_at_most_0.595':'fail',
   'wins_at_least_75_percent':'pass for several arms',
   'worst_at_most_1.10_1.15':'fails for the raw-best 0.5 arm; safe 0.2 arm fails the ratio gate',
   'positive_alignment':'pair-source substitution alignment is effectively identical (median cosine >0.999999 at full companion)',
   'added_compute_below_14B':'pair accumulator passes; independent companion center does not',
 },
 'limitations':['No new immutable cohort was opened. The 24-network block was already exposed by the frozen 32-probe companion validation before this 128-probe/pair-source extension.','The high-reference 8-network pair substitution bundle does not ship scalar contraction arrays, so exact scalar RMSE cannot be recomputed; final-output metrics are authoritative.','Measured subprocess FlopScope was not available; the pair-specific arithmetic is counted exactly, while companion propagation uses the prior measured/proxy report.']
}
json.dump(summary,open(OUT/'LEGAL_ESTIMATOR_RESULTS.json','w'),indent=2)

# Rows.csv
cols=['seed','baseline_mse','baseline_unbiased_mse','p128_b129_a05_companion_ratio','p128_b129_a05_primary_ratio','p128_b129_a02_companion_ratio','p128_b129_a02_primary_ratio','p128_b16_a01_companion_ratio','p128_b16_a01_primary_ratio','p128_b129_pair_increment_over_full','p128_b129_full_primary_cosine','p128_b16_pair_increment_over_full','p128_b16_full_primary_cosine','runtime_seconds']
with open(OUT/'ROWS.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for r in R:
  row={'seed':r['seed'],'baseline_mse':r['baseline_mse'],'baseline_unbiased_mse':r['baseline_unbiased_mse'],'runtime_seconds':r['runtime_seconds']}
  for bc,a,tag in [(129,.5,'b129_a05'),(129,.2,'b129_a02'),(16,.1,'b16_a01')]:
   row[f'p128_{tag}_companion_ratio']=arm(r,128,bc,'companion_pairs',a)['ratio']
   row[f'p128_{tag}_primary_ratio']=arm(r,128,bc,'primary_pairs',a)['ratio']
  for bc in [129,16]:
   d=r['probe_results']['128']['basis_results'][str(bc)]['decomposition']
   row[f'p128_b{bc}_pair_increment_over_full']=d['independent_pair_increment_over_full_norm']
   row[f'p128_b{bc}_full_primary_cosine']=d['full_vs_primary_pair_cosine']
  w.writerow(row)

N=66048;D=256;P=128;Nc=16*512
cost={
 'dimensions':{'rows':N,'dimension':D,'probes':P,'companion_16_rows':Nc},
 'primary_same_cloud_fused':{
   'incremental_flops':4*N*P,
   'incremental_billions':4*N*P/1e9,
   'assumption':'H@V.T and selected H[:,i] already exist for frozen radial features; count two multiply-reductions (s and t)',
   'extra_propagation_flops':0,
 },
 'standalone_direct_without_projection_reuse':{
   'flops':N*P*(2*D-1)+2*P*(2*N-1),
   'billions':(N*P*(2*D-1)+2*P*(2*N-1))/1e9,
 },
 'full_second_matrix_internal_diagnostic':{'flops':D*D*(2*N-1),'billions':D*D*(2*N-1)/1e9,'not_recommended':True},
 'companion_16_pair_accumulation_after_projection':{'incremental_flops':4*Nc*P,'billions':4*Nc*P/1e9},
 'memory_streaming':{'recommended_block_rows':512,'projection_buffer_bytes_float64':512*P*8,'selected_buffer_bytes_float64':512*P*8,'total_megabytes':2*512*P*8/1e6},
 'assets':{'new_static_assets_bytes':0,'runtime_outputs_float64_bytes':2*P*8},
 'companion_context':{'full_companion_heavy_cost_ratio':1.935484870967742,'full_companion_adjusted_proxy_32probe':1.4958027457088725,'companion_16_heavy_cost_ratio':1.116029,'note':'These costs belong to the center estimator, not the retained pair accumulator.'}
}
json.dump(cost,open(OUT/'COST_MODEL.json','w'),indent=2)

registry={
 'schema_version':1,'probe_count':128,'dimension':256,
 'probe_contract':{'left':'u_p=e_{i_p}','right':'v_p=normalized selected observable Q row','selection':'top 128 Q-row norms, no replacement'},
 'exact_counts_per_network':{'unique_selected_indices':128,'marginal_second_slots':128,'row_direction_slots':128,'exact_pair_slot_deduplications':0},
 'observable_pair_slots':[{'name':'s_p','formula':'(D/rho^2) mean(h_i_p^2)','count':128},{'name':'t_p','formula':'(D/rho^2) mean(h_i_p (v_p^T h))','count':128}],
 'external_center_slots':[{'name':'d_i_p','formula':'mu_i_p-m_i_p','count':128},{'name':'a_p','formula':'v_p^T(mu-m)','count':128}],
 'derived_without_extra_estimation':['mu_i=m_i+d_i','z_p=v_p^T mu=v_p^T m+a_p'],
 'downstream_map':'c_output = lower_defect @ beta, beta shape (128,256)',
 'smallest_defensible_runtime_representation':'256 primary-cloud pair scalars + 256 independent center contractions; no full covariance and no independent pair cloud'
}
json.dump(registry,open(OUT/'CONTRACTION_REGISTRY.json','w'),indent=2)

(OUT/'PAIR_TARGET_DERIVATION.md').write_text(r'''# Pair-target derivation

For frozen probe $p$, let $u_p=e_{i_p}$, right direction $v_p$, sample pointwise center $m$, Gaussian mean $\mu$, raw second moment $M=\mathbb E[hh^T]$, and connected cubic contraction $c_p=e_{i_p}^T C_3 v_p$. Define

\[
d=\mu-m,\quad a_p=v_p^Td,\quad z_p=v_p^T\mu,\quad s_p=M_{i_pi_p},\quad t_p=M_{i_p,:}v_p.
\]

The exact pointwise-centered radial-Hermite anchor contraction is

\[
A_p(m)=\frac{c_p+s_pa_p+2d_{i_p}t_p+2(m_{i_p}^2-\mu_{i_p}^2)z_p}{D+1}.
\]

Dropping the empirically neutral connected defect leaves the lower recentering target

\[
\ell_p=\frac{s_pa_p+2d_{i_p}t_p+2(m_{i_p}^2-\mu_{i_p}^2)z_p}{D+1}.
\]

## Required decomposition

- **Mean-projection term:** $s_pa_p/(D+1)$, requiring $a_p=v_p^T(\mu-m)$.
- **Marginal-second-moment term:** the same product viewed through $s_p=M_{ii}$.
- **Row-direction pair term:** $2d_it_p/(D+1)$.
- **Center-induced linear term:** $2d_it_p/(D+1)-4m_id_iz_p/(D+1)$ after expanding $m_i^2-\mu_i^2=-2m_id_i-d_i^2$.
- **Center-induced diagonal-quadratic term:** $-2d_i^2z_p/(D+1)$.
- **Optional connected cubic:** $c_p/(D+1)$; excluded from the retained estimator.

The local scalar-error coefficients are

\[
\partial_{a_p}\ell_p=\frac{s_p}{D+1},\quad
\partial_{s_p}\ell_p=\frac{a_p}{D+1},\quad
\partial_{t_p}\ell_p=\frac{2d_i}{D+1},
\]
\[
\partial_{d_i}\ell_p=\frac{2t_p-4\mu_i z_p}{D+1},\quad
\partial_{z_p}\ell_p=\frac{2(m_i^2-\mu_i^2)}{D+1}.
\]

For frozen cross-fit coefficient row $\beta_p\in\mathbb R^{256}$, every scalar anchor error $e_p$ maps exactly to

\[
c_{\rm out}=\sum_p e_p\beta_p=e^T\beta.
\]

The authoritative metric is therefore output-space quadratic loss. For baseline error $r$,

\[
\|r+c_{\rm out}\|^2=\|r\|^2+2\langle r,c_{\rm out}\rangle+\|c_{\rm out}\|^2.
\]

## Consequence

Pair moments do not create an absolute correction by themselves: their coefficients $a_p$ and $d_i$ vanish when the center defect is unavailable. They modulate the center-driven correction. This is why pair terms are algebraically necessary but need not be independently estimated.
''')

(OUT/'RANK_AND_SHARING_AUDIT.md').write_text(f'''# Rank and sharing audit

## Exact registry

The frozen selector chooses 128 different rows from 256 without replacement. Consequently each network has exactly 128 distinct diagonal slots `M[i_p,i_p]` and 128 distinct row-direction slots `M[i_p,:] @ v_p`; there is no exact slot deduplication. They share one projection matrix `H @ V.T`, so all row-direction moments can be accumulated in one GEMM/reduction rather than 128 separate adjoints.

## Downstream-weighted rank

The exposed lower-structure corpus gives pooled shared ranks **29/36/44** for 90%/95%/99% energy. A universal rank-16 representation is therefore too small. Per network, however, the downstream-weighted lower matrix has median rank-2 energy **{summary['rank_audit']['local_rank2_energy_median']:.8f}**, and oracle local rank 2 has median anchor relative error **{summary['rank_audit']['local_rank2_median_anchor_relative_error']:.6f}**.

This local rank is mechanism evidence, not a legal compression: its right space is network-specific and contains the unknown center-defect direction. A shared representation still needs roughly 30 modes.

## Precision audit

On the high-reference 8-network pair substitution, exact Gaussian pairs scored **{hi['summary']['exact']['pooled_ratio']:.6f}** and primary Kerdock pairs scored **{hi['summary']['sample_both']['pooled_ratio']:.6f}**, a ratio difference of only **{hi['summary']['sample_both']['pooled_ratio']-hi['summary']['exact']['pooled_ratio']:.6f}**.

On the independent-center 24-network ablation, the 128-probe/full-companion arm changes from **{agg(128,129,'primary_pairs',.5)['pooled_ratio']:.6f}** with primary pairs to **{agg(128,129,'companion_pairs',.5)['pooled_ratio']:.6f}** with independent pairs. The median pair-source increment is only **{pair_stats(128,129)['independent_pair_increment_over_full_norm']['median']*100:.3f}%** of correction norm, with median correction cosine **{pair_stats(128,129)['full_vs_primary_pair_cosine']['median']:.9f}**.

Thus pair precision is already far beyond what is needed for the 0.595 target. The unresolved precision requirement belongs to the center contractions, not to `s_p` or `t_p`.
''')

(OUT/'EXPERIMENT_SPEC.md').write_text('''# Experiment specification

Date: 2026-07-29

## Frozen question

Does an independently estimated selected pair-moment defect improve the lower radial-Hermite control enough to justify a deployable implementation, relative to directly reusing selected pair contractions from the primary Kerdock cloud?

## Cohorts

1. Frozen M109/M110 high-reference mechanism evidence (24 networks), used only as prior oracle context.
2. Exposed high-reference pair substitution (8 networks): exact center fixed; exact versus primary-cloud pair moments.
3. Exposed prospective companion validation block (24 networks): primary rotation 3, companion rotation 97, two independent 524,288-node final references. The original frozen arm used 32 probes. This experiment adds a post-hoc 128-probe extension and pair-source swaps; it is development evidence, not immutable validation.

## Arms

At 32 and 128 probes, companion basis counts 16 and 129, compare:

- companion diagonal + companion row moments;
- primary diagonal + primary row moments;
- companion diagonal + primary row moments;
- primary diagonal + companion row moments.

The companion mean is held fixed within each comparison. Shrinkages are 0.05, 0.10, 0.20, and 0.50. Final-output MSE, wins, tails, pair-increment output norm, and correction cosine are authoritative.

## Stop rule

Close independent pair estimation if primary-pair substitution is output-equivalent to independent pairs and the pair increment is negligible compared with center-estimation error. Retain only the direct fused accumulator if its exact incremental arithmetic fits the budget.
''')

(OUT/'DECISION.md').write_text(f'''# Decision

## Verdict: retain one module; close the standalone family

**Close independent selected pair-moment estimation as a standalone Path 1 branch.**

Retain only the fused primary-cloud accumulator for

- `s_p = M_s[i_p,i_p]`;
- `t_p = M_s[i_p,:] @ v_p`.

These are algebraically necessary, but an independent estimate does not materially change the final correction. The center defect supplies the sign and amplitude.

## Decisive results

- High-reference exact center: exact pairs **{hi['summary']['exact']['pooled_ratio']:.6f}** versus primary pairs **{hi['summary']['sample_both']['pooled_ratio']:.6f}**.
- New 24-network, 128-probe, full-companion-center ablation at alpha 0.50:
  - independent pairs: **{agg(128,129,'companion_pairs',.5)['pooled_ratio']:.6f}**, {agg(128,129,'companion_pairs',.5)['wins']}/24 wins, worst {agg(128,129,'companion_pairs',.5)['worst']:.3f};
  - primary pairs: **{agg(128,129,'primary_pairs',.5)['pooled_ratio']:.6f},** {agg(128,129,'primary_pairs',.5)['wins']}/24 wins, worst {agg(128,129,'primary_pairs',.5)['worst']:.3f};
  - paired ratio difference: **{delta_boot(128,129,.5)[0]:+.6f}**, bootstrap 95% interval [{delta_boot(128,129,.5)[1][0]:+.6f}, {delta_boot(128,129,.5)[1][1]:+.6f}].
- Median independent-pair increment: **{pair_stats(128,129)['independent_pair_increment_over_full_norm']['median']*100:.3f}%** of full correction norm; median full-versus-primary-pair cosine **{pair_stats(128,129)['full_vs_primary_pair_cosine']['median']:.9f}**.
- The safer alpha 0.20 arm reaches {agg(128,129,'companion_pairs',.2)['wins']}/24 wins and worst {agg(128,129,'companion_pairs',.2)['worst']:.3f}, but raw ratio **{agg(128,129,'companion_pairs',.2)['pooled_ratio']:.3f}** misses the 0.75 development gate.

The post-hoc 128-probe alpha 0.50 arm passes the development ratio at **{agg(128,129,'companion_pairs',.5)['pooled_ratio']:.3f}**, but fails the promotion ratio, tail, and compute gates. It is center-estimator mechanism evidence only.

## Smallest defensible representation

At runtime keep 256 observable primary-cloud pair scalars (`128 s_p + 128 t_p`) and estimate only 256 external center contractions (`128 d_i + 128 v_p^T d`). `mu_i` and `v_p^T mu` follow algebraically. Do not construct a full covariance and do not propagate an independent cloud for pair moments.

## Next branch

Redirect all statistical work to the selected center contractions. Reuse this pair module unchanged in analytic centered-defect, independent micro-cubature, or shared-arithmetic center estimators.
''')

# README
(OUT/'README.md').write_text('''# Direct selected pair-moment experiment

This bundle completes the bounded Experiment 2 audit. Start with `DECISION.md`, then `PAIR_TARGET_DERIVATION.md` and `RANK_AND_SHARING_AUDIT.md`.

Run `python test_direct_pair_estimator.py` for algebraic validation. `raw/PAIR_COMPANION_ABLATION_24_COMBINED.json` contains the full new development ablation.
''')

# Manifest last
files=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name not in {'MANIFEST.json'}:
  h=hashlib.sha256(p.read_bytes()).hexdigest();files.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':h})
manifest={'experiment':'Direct Estimation of the Selected Pair-Moment Defect','date':'2026-07-29','decision':'retain fused primary-cloud pair accumulator; close independent pair estimator','files':files,'source_cohorts':{'highref_oracle':'exposed','highref_pair8':'exposed','companion_validation24':'already exposed before this extension'},'tests':'python test_direct_pair_estimator.py'}
json.dump(manifest,open(OUT/'MANIFEST.json','w'),indent=2)
print('built',len(files),'files')

from pathlib import Path
import json,csv,hashlib,shutil,os,math,statistics,tarfile,ast,zipfile
import numpy as np
ROOT=Path('/mnt/data/T0_official_grader_instrumentation_20260729')
PKG=ROOT/'packages'; ROOT.mkdir(parents=True,exist_ok=True)
for s in ['T0.1_basis_curve','T0.2_compute_calibration','T0.3_A42_A43_grade','evidence','scripts']:(ROOT/s).mkdir(exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write_csv(p,rows,fields=None):
 if not rows:return
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields:fields.append(k)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def loadj(p):return json.load(open(p))
# Evidence copies
src_evidence={
 'strassen_sparse_basis_frontier.json':'/mnt/data/t0_work/arc_code/arc_whitebox/results/strassen_sparse_basis_frontier.json',
 'sparse_kerdock_frontier_selection.json':'/mnt/data/t0_work/arc_code/arc_whitebox/results/sparse_kerdock_frontier_selection.json',
 'kerdock_mub5_official_full100.json':'/mnt/data/t0_work/arc_code/arc_whitebox/results/kerdock_mub5_official_full100.json',
 'kerdock_mub5_winograd_official_row0.json':'/mnt/data/t0_work/arc_code/arc_whitebox/results/kerdock_mub5_winograd_official_row0.json',
 'A43_RESULTS_SUMMARY.json':'/mnt/data/t0_work/a43_bundle/results/RESULTS_SUMMARY.json',
 'A43_arm_measurements.csv':'/mnt/data/t0_work/a43_bundle/results/arm_measurements.csv',
 'A43_validation_suite.json':'/mnt/data/t0_work/a43_bundle/results/validation_suite.json',
}
for n,p in src_evidence.items():shutil.copy2(p,ROOT/'evidence'/n)
# local results copies
local_files=[]
for p in sorted(Path('/mnt/data').glob('t0_*.json'))+sorted(Path('/mnt/data').glob('t0_*.npy')):
 dest=ROOT/'evidence'/p.name;shutil.copy2(p,dest);local_files.append(dest.name)
# Constants
B=272_000_000_000
current={'adjusted_score':1.550e-7,'raw_mse':2.416e-7,'effective_compute':174.5e9,'budget':B}
current['score_multiplier']=current['adjusted_score']/current['raw_mse']
current['compute_multiplier']=current['effective_compute']/B
A43_TRACK=170_382_691_584
PROD_TRACK=170_906_815_488
A43_SAVE=PROD_TRACK-A43_TRACK
# Parse prior frontier
front=loadj(ROOT/'evidence'/'strassen_sparse_basis_frontier.json')
front_by={x['bases']:x for x in front['frontier']}
sparse=loadj(ROOT/'evidence'/'sparse_kerdock_frontier_selection.json')
# fit exposed scaling for seed3 methods
power={}
for method in ['greedy','greedy_swaps']:
 xs=[x for x in sparse['summaries'] if x['family']=='seed3_only' and x['method']==method]
 k=np.array([x['total_bases'] for x in xs],float);m=np.array([x['nested_fivefold_raw_mse'] for x in xs])
 power[method]=float(-np.polyfit(np.log(k),np.log(m),1)[0])
# fold bootstrap ratios
fold_stats={}
rng=np.random.default_rng(20260729)
for method in ['greedy','greedy_swaps']:
 base={r['fold']:r['test_mse'] for r in sparse['records'] if r['family']=='seed3_only' and r['method']==method and r['total_bases']==129}
 for k in [32,64,96,129]:
  vals={r['fold']:r['test_mse'] for r in sparse['records'] if r['family']=='seed3_only' and r['method']==method and r['total_bases']==k}
  ratios=np.array([(vals[i]*front_by[k]['projected_effective_compute'])/(base[i]*front_by[129]['projected_effective_compute']) for i in sorted(base)])
  idx=rng.integers(0,len(ratios),(200000,len(ratios)));boot=ratios[idx].mean(1)
  fold_stats[f'{method}_{k}']={'ratios':ratios.tolist(),'mean':float(ratios.mean()),'median':float(np.median(ratios)),'min':float(ratios.min()),'max':float(ratios.max()),'bootstrap_95':[float(x) for x in np.quantile(boot,[.025,.975])]}
# shape tracer calibration, local source trace
trace={32:44_432_054_528,64:88_765_901_056,96:133_099_747_584,129:178_834_820_992}
scale=A43_TRACK/trace[129]
shape_cost={k:trace[k]*scale for k in trace}
prop_cost={k:A43_TRACK*k/129 for k in [16,20,32,64,96,129]}
# local timing records
local={}
for seed in [51000,51001]:
 for k in [32,64,96,129]:
  local[(seed,k)]=loadj(Path(f'/mnt/data/t0_panel_seed{seed}_k{k}.json'))
# time linearity
ks=[];ts=[]
for (seed,k),r in local.items():ks.append(k);ts.append(r['run_s'])
coef=np.polyfit(np.array(ks,float),np.array(ts,float),1);pred=np.polyval(coef,ks);r2=1-float(np.sum((np.array(ts)-pred)**2)/np.sum((np.array(ts)-np.mean(ts))**2))
# full-disagreement proxy
proxy={}
for seed in [51000,51001]:
 full=np.load(f'/mnt/data/t0_panel_seed{seed}_k129.npy')
 for k in [32,64,96,129]:
  v=np.load(f'/mnt/data/t0_panel_seed{seed}_k{k}.npy');d=v-full
  proxy[(seed,k)]={'mse_vs_129':float(np.mean(d*d)),'rms_vs_129':float(np.sqrt(np.mean(d*d))),'cosine_vs_129':float(np.dot(v,full)/(np.linalg.norm(v)*np.linalg.norm(full)))}
# package audit
package_audit=[]
for tarp in sorted(PKG.glob('*.tar.gz')):
 with tarfile.open(tarp,'r:gz') as tf:
  names=tf.getnames();bad=[n for n in names if n.startswith('/') or '..' in Path(n).parts]
  members={Path(n).name:n for n in names if Path(n).name in ['manifest.json','estimator.py','fast_matmul.py','kerdock_mub5_seed3.npz']}
  rec={'package':tarp.name,'sha256':sha(tarp),'bytes':tarp.stat().st_size,'safe_paths':not bad,'files':sorted(names)}
  if 'manifest.json' in members:
   m=json.loads(tf.extractfile(members['manifest.json']).read())
   checks={}
   for x in m['files']:
    b=tf.extractfile(members[x['name']]).read();checks[x['name']]=hashlib.sha256(b).hexdigest()==x['sha256']
   rec['manifest_hashes_ok']=all(checks.values());rec['manifest_file_checks']=checks;rec['description']=m.get('description')
  package_audit.append(rec)
dump(ROOT/'PACKAGE_AUDIT.json',package_audit)
# T0.1 rows
rows=[]
for k in [129,96,64,32]:
 pkg=PKG/f'A43_basis{k:03d}.tar.gz'
 times=[local[(s,k)]['run_s'] for s in [51000,51001]]
 peaks=[local[(s,k)]['peak_rss_mib'] for s in [51000,51001]]
 prox=[proxy[(s,k)]['mse_vs_129'] for s in [51000,51001]]
 f=front_by[k]
 ratio=f['projected_adjusted_score']/front_by[129]['projected_adjusted_score']
 fs=fold_stats[f'greedy_swaps_{k}']
 residual_equiv=max(0.0,current['effective_compute']-A43_TRACK)
 c_const=prop_cost[k]+residual_equiv
 rows.append({
  'arm':f'basis_{k}','basis_count':k,'row_count':k*512,'basis_rule':'literal prefix of original basis order; k=129 includes coordinate basis','package':pkg.name,'package_sha256':sha(pkg),
  'local_time_mean_s':statistics.mean(times),'local_time_min_s':min(times),'local_time_max_s':max(times),'local_peak_rss_max_mib':max(peaks),
  'local_mse_vs_129_mean_NOT_TARGET':statistics.mean(prox),'local_mse_vs_129_max_NOT_TARGET':max(prox),
  'shape_traced_scaled_tracked_flops':shape_cost[k],'proportional_tracked_flops':prop_cost[k],
  'constant_residual_scenario_effective_compute':c_const,'constant_residual_scenario_multiplier':max(.1,c_const/B),'floor_reached_constant_residual':c_const<=.1*B,
  'archived_exposed_raw_mse':f['raw_mse'],'archived_exposed_projected_effective_compute':f['projected_effective_compute'],'archived_exposed_projected_adjusted_score':f['projected_adjusted_score'],'archived_adjusted_ratio_vs_129':ratio,
  'required_raw_gain_to_tie_129_under_archived_curve':ratio,
  'seed3_greedy_swaps_fold_adjusted_ratio_mean':fs['mean'],'seed3_greedy_swaps_fold_ratio_boot_lo':fs['bootstrap_95'][0],'seed3_greedy_swaps_fold_ratio_boot_hi':fs['bootstrap_95'][1],
  'official_raw_mse':'PENDING','official_adjusted_score':'PENDING','official_effective_compute':'PENDING','official_residual_wall_s':'PENDING','status':'FROZEN_PACKAGE_OFFICIAL_BLOCKED'
 })
write_csv(ROOT/'T0.1_basis_curve'/'ROWS.csv',rows)
# T0.1 results/cost/decision
r1={'status':'BLOCKED_OFFICIAL_DEPENDENCY','current_position':current,'basis_arms':rows,'local_timing_fit':{'seconds_intercept':float(coef[1]),'seconds_per_basis':float(coef[0]),'r_squared':r2},'exposed_scaling_exponent_mse_proportional_to_k_minus_p':power,'interpretation':{'flat_adjusted_curve_threshold_p':1.0,'exposed_p_above_one':all(x>1 for x in power.values()),'official_floor_not_reached_by_any_preregistered_arm':True},'limits':['Archived basis frontier used official exposed IDs 0-49 and basis selection; it is a prior, not the requested official frozen-prefix curve.','MSE versus the 129-basis output is a disagreement diagnostic, not Gaussian truth.','No official hidden/protected result was opened.']}
dump(ROOT/'T0.1_basis_curve'/'RESULTS.json',r1)
cost1={'budget':B,'score_formula':'raw_mse * max(0.1, effective_compute / budget)','effective_compute_formula':'tracked_flops + 1e11 * residual_wall_seconds','a43_full_tracked_flops':A43_TRACK,'basis_cost_models':{'proportional':prop_cost,'shape_trace_scaled_to_full':shape_cost,'spread_note':'The two independent projections differ by less than about 0.25% at preregistered counts; official FlopScope remains authoritative.'},'floor_compute':.1*B,'floor_basis_bracket':{'constant_residual_model':'between 16 and 20 bases','row_scaled_residual_model':'approximately 20.6 bases','preregistered_minimum_bases':32}}
dump(ROOT/'T0.1_basis_curve'/'COST_MODEL.json',cost1)
(ROOT/'T0.1_basis_curve'/'NODE_SPEC.md').write_text('''# T0.1 Node Spec\n\nFreeze one estimator family and change only the count of complete 512-row bases: 129, 96, 64, and 32. The partial arms are literal prefixes of the original basis order. No correction, amplitude, basis weighting, rotation, or implementation choice may change between arms. Measure official raw MSE, adjusted score, tracked FLOPs, residual wall time, effective compute, and tails.\n''')
(ROOT/'T0.1_basis_curve'/'DECISION.md').write_text(f'''# Decision — BLOCKED\n\nThe local/proxy branch is exhausted. All four packages are frozen and validated. Historical exposed data strongly favor 129 bases: the archived projected adjusted ratios are {rows[1]['archived_adjusted_ratio_vs_129']:.3f} at 96, {rows[2]['archived_adjusted_ratio_vs_129']:.3f} at 64, and {rows[3]['archived_adjusted_ratio_vs_129']:.3f} at 32. The five-fold exploratory intervals also remain mostly above one.\n\nThis does **not** close the official question. The exact missing dependency is the official WhestBench 0.13.0 + FlopScope 0.9.1 subprocess and official cohort/submission interface. No further local basis ordering, thresholding, or timing is justified. Run the four frozen arms once; then either preserve 129 or notify T2/T3 with the measured control gain required.\n''')
(ROOT/'T0.1_basis_curve'/'CHILDREN.md').write_text('''# Children\n\nNo local child remains. Dependency fulfillment is the official four-arm run. Only if the official curve is flat/improving may T2/T3 inherit the measured operating point. The 16- and 20-basis packages are local-only floor diagnostics and must not consume official submissions without coordinator allocation.\n''')
# T0.2
pairs=[]
for seed,cp,dp in [(51000,'/mnt/data/t0_pkg_k129.json','/mnt/data/t0_pkg_delta64.json'),(51001,'/mnt/data/t0_pair_clean_seed51001.json','/mnt/data/t0_pair_delta_seed51001.json')]:
 c=loadj(cp);d=loadj(dp);pairs.append({'seed':seed,'clean_time_s':c['seconds'],'delta_time_s':d['seconds'],'local_time_delta_s':d['seconds']-c['seconds'],'clean_digest':c['final_digest'],'delta_digest':d['final_digest'],'output_identical':c['final_digest']==d['final_digest']})
write_csv(ROOT/'T0.2_compute_calibration'/'ROWS.csv',pairs)
delta=2_147_483_648
r2d={'status':'BLOCKED_OFFICIAL_DEPENDENCY','clean_package':'A43.tar.gz','delta_package':'A43_delta64.tar.gz','expected_tracked_delta_flops':delta,'local_pairs':pairs,'all_outputs_identical':all(x['output_identical'] for x in pairs),'predicted_no_residual_score_delta_at_current_raw':current['raw_mse']*delta/B,'predicted_multiplier_delta':delta/B,'official_estimands':{'observed_tracked_delta':'F_delta-F_clean','observed_residual_delta_seconds':'(C_delta-C_clean-observed_tracked_delta)/1e11','score_sensitivity_per_effective_flop':'(S_delta-S_clean)/(C_delta-C_clean)','raw_mse_parity':'must be exact except deterministic numerical reporting'} }
dump(ROOT/'T0.2_compute_calibration'/'RESULTS.json',r2d)
# repricing
items=[
 ('translation_reuse_final_replay',0.0847,'validated replacement cost'),('A43_tracked_saving',0.524123904,'saving; negative cost in deployment'),('one_billion',1.0,'reference'),('full_second_final_replay',5.549,'historical full replay'),('ordinary_pilot_1024_plus_2048',8.17,'pilot total'),('cross_scale_pilot_low',10.2,'cross-scale low'),('adjoint_compression_measured',10.34,'measured FlopScope'),('cross_scale_pilot_high',12.2,'cross-scale high'),('historical_gate_ceiling',14.0,'old <14B gate'),('companion16_propagation_only',19.811940882,'A43 16 bases depth30 projection'),('companion16_plus_inherited_control',22.049940882,'+2.238B control accounting'),('companion16_plus_literal_control',39.486940882,'+19.675B literal control accounting')]
repr_rows=[]
for name,bill,note in items:
 repr_rows.append({'item':name,'added_effective_compute_B':bill,'required_raw_gain_factor':1+bill/(current['effective_compute']/1e9),'required_raw_reduction_fraction':1-1/(1+bill/(current['effective_compute']/1e9)),'fixed_raw_score_penalty':current['raw_mse']*bill/272,'note':note})
write_csv(ROOT/'T0.2_compute_calibration'/'REPRICING.csv',repr_rows)
cost2={'budget':B,'current_position':current,'delta_pair':{'tracked_delta':delta,'expected_no_residual_score_delta':current['raw_mse']*delta/B},'marginal_rates_at_current_raw':{'score_per_1B_effective':current['raw_mse']*1e9/B,'score_per_1ms_residual':current['raw_mse']*1e8/B,'raw_gain_required_per_1B_factor':1+1/(current['effective_compute']/1e9)},'repricing':repr_rows}
dump(ROOT/'T0.2_compute_calibration'/'COST_MODEL.json',cost2)
(ROOT/'T0.2_compute_calibration'/'NODE_SPEC.md').write_text('''# T0.2 Node Spec\n\nCompare clean A43 with an output-independent delta arm containing exactly 64 eager tracked float32 256×256 matrix multiplications. The expected tracked difference is 2,147,483,648 operations. Use identical official subprocess conditions and confirm raw output parity. Infer residual-wall charging from any effective-compute difference beyond the tracked delta.\n''')
(ROOT/'T0.2_compute_calibration'/'DECISION.md').write_text('''# Decision — BLOCKED\n\nThe calibration pair is constructed, hash-frozen, and output-identical on two complete depth-32, 66,048-row runs. Local wall differences are deliberately not interpreted as official residual time because local NumPy cannot separate FlopScope backend time from residual Python time. The official paired subprocess is the sole remaining measurement.\n\nAll historical cost gates have been repriced in `REPRICING.csv`; no further local no-op design is useful.\n''')
(ROOT/'T0.2_compute_calibration'/'CHILDREN.md').write_text('''# Children\n\nNo new hypothesis child. After the paired official run, replace all historical `<14B` heuristics with the measured exchange rate and propagate the canonical cost model to T1–T4.\n''')
# T0.3
summary=loadj(ROOT/'evidence'/'A43_RESULTS_SUMMARY.json')
# normalized arm rows from known archive
arms=[
 {'arm':'production_baseline','package':'production_baseline.tar.gz','tracked_flops':PROD_TRACK,'archived_local_full_wall_s':63.995,'archived_peak_rss_mib':2172.0,'numerical_drift_rms_vs_production':0.0,'official_status':'PENDING'},
 {'arm':'A42','package':'A42.tar.gz','tracked_flops':PROD_TRACK,'archived_local_full_wall_s':28.011,'archived_peak_rss_mib':374.7,'numerical_drift_rms_vs_production':3.418e-12,'official_status':'PENDING'},
 {'arm':'A43','package':'A43.tar.gz','tracked_flops':A43_TRACK,'archived_local_full_wall_s':25.702,'archived_peak_rss_mib':374.8,'numerical_drift_rms_vs_production':3.418e-12,'official_status':'PENDING'}]
# new common k8 panel
for r in arms:
 m={'production_baseline':'baseline','A42':'a42','A43':'a43'}[r['arm']];j=loadj(f'/mnt/data/t0_t03_{m}_seed51002_k8.json');r.update({'new_k8_seed':51002,'new_k8_wall_s':j['run_s'],'new_k8_digest':j['output_digest'],'package_sha256':sha(PKG/r['package'])})
write_csv(ROOT/'T0.3_A42_A43_grade'/'ROWS.csv',arms)
r3={'status':'BLOCKED_OFFICIAL_DEPENDENCY','arms':arms,'exact_relations':{'A42_vs_A43_local_bit_identical':True,'A43_tracked_saving':A43_SAVE,'A43_break_even_extra_residual_seconds':A43_SAVE/1e11,'memory_reduction_fraction_vs_production':1-374.8/2172.0},'promotion_gate':['measured official adjusted score improves','raw MSE and tails parity','no numerical failure','effective compute improves'],'retention_rule':'Preserve A42/A43 streaming memory reduction even if adjusted score is neutral.'}
dump(ROOT/'T0.3_A42_A43_grade'/'RESULTS.json',r3)
cost3={'production_tracked_flops':PROD_TRACK,'A42_tracked_flops':PROD_TRACK,'A43_tracked_flops':A43_TRACK,'A43_tracked_saving':A43_SAVE,'A43_break_even_residual_overhead_seconds':A43_SAVE/1e11,'score_improvement_if_raw_and_residual_unchanged_fraction':A43_SAVE/current['effective_compute']}
dump(ROOT/'T0.3_A42_A43_grade'/'COST_MODEL.json',cost3)
(ROOT/'T0.3_A42_A43_grade'/'NODE_SPEC.md').write_text('''# T0.3 Node Spec\n\nRun production baseline, A42, and A43 in identical official subprocess conditions. Promote only on measured adjusted improvement with numerical, raw-MSE, and tail parity. Retain the streaming memory reduction even if score-neutral.\n''')
(ROOT/'T0.3_A42_A43_grade'/'DECISION.md').write_text(f'''# Decision — BLOCKED / RETAIN MODULE\n\nA42 and A43 are bit-identical locally in both the archived full-shape run and a new common 8-basis panel. A43 saves exactly {A43_SAVE:,} projected tracked operations and may tolerate {A43_SAVE/1e11*1000:.3f} ms more official residual time than production. Peak memory falls by about {(1-374.8/2172)*100:.1f}%.\n\nPromotion remains blocked on the same official subprocess comparison. Independently of score, retain the streaming implementation as a memory/iteration module unless the official API fails. No neighboring chunk, layout, mixed-precision, or compiler tuning remains open.\n''')
(ROOT/'T0.3_A42_A43_grade'/'CHILDREN.md').write_text('''# Children\n\nNo local implementation child. Official result chooses production, A42 fallback, or A43 primary. Then freeze that kernel as the baseline for every statistical tree.\n''')
# common node manifests, proposed ledger
for sub in ['T0.1_basis_curve','T0.2_compute_calibration','T0.3_A42_A43_grade']:
 d=ROOT/sub
 files=[p for p in d.rglob('*') if p.is_file()]
 man={'node':sub,'status':'BLOCKED_OFFICIAL_DEPENDENCY','files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]}
 dump(d/'MANIFEST.json',man)
ledger=[
 {'ID':'T0.1','Branch':'Official instrumentation','Experiment':'Frozen A43 basis-count curve 129/96/64/32','Environment':'Local exact-width packages; official dependencies absent','Canonical result':'Four packages frozen; historical exposed proxy favors 129; official curve not measured','Status':'BLOCKED — exact official subprocess/cohort dependency','Next action':'Run seven unique official packages per OFFICIAL_RUNBOOK.md'},
 {'ID':'T0.2','Branch':'Official instrumentation','Experiment':'2.147483648B tracked-operation calibration pair','Environment':'Two full local depth-32 pairs','Canonical result':'Outputs digest-identical on both seeds; official residual exchange pending','Status':'BLOCKED — official paired measurement','Next action':'Run A43 and A43_delta64 under identical subprocess conditions'},
 {'ID':'T0.3','Branch':'Implementation','Experiment':'Production vs A42 vs A43 official grade','Environment':'Archived full-shape plus new common k8 panel','Canonical result':'A42/A43 exact local parity; A43 -0.524123904B tracked; ~82.7% memory reduction','Status':'BLOCKED promotion / RETAIN MODULE','Next action':'Paired official three-arm grade; retain streaming memory path'}]
write_csv(ROOT/'PROPOSED_LEDGER_ADDITIONS.csv',ledger)
# runbook + main report
(ROOT/'OFFICIAL_RUNBOOK.md').write_text('''# Official execution runbook\n\n## Unique submissions: seven\n\nReuse `A43.tar.gz` for T0.1/129, T0.2/clean, and T0.3/A43. Run, in one unchanged grader window:\n\n1. `production_baseline.tar.gz`\n2. `A42.tar.gz`\n3. `A43.tar.gz`\n4. `A43_delta64.tar.gz`\n5. `A43_basis096.tar.gz`\n6. `A43_basis064.tar.gz`\n7. `A43_basis032.tar.gz`\n\nRecord per network: raw MSE, adjusted score, tracked FLOPs, residual wall time, effective compute, total wall, failures, median, p90, worst, and package hash. Do not tune or replace an arm after seeing any result. The 16/20-basis packages are local-only floor diagnostics and are not authorized submissions.\n\n## Immediate calculations\n\n- T0.2 residual delta: `(C_delta - C_clean - 2_147_483_648) / 1e11`.\n- T0.3 A43 passes compute if `residual_A43 - residual_prod < 0.00524123904 s`, subject to raw/tail parity.\n- T0.1 required control gain at each partial count: `(adjusted_partial / adjusted_129)` before adding control cost; then include exact control cost using T0.2's canonical exchange model.\n''')
(ROOT/'README.md').write_text(f'''# Tree T0 — Official Grader and Operating-Point Instrumentation\n\n## Executive result\n\nT0 is locally exhausted but **not officially closed**. The exact official grader stack, official cohort, and submission interface are absent. All remaining measurements have therefore been converted into seven immutable packages and a deterministic runbook rather than mislabeled projections.\n\n### What is closed locally\n\n- Four basis-count packages (129/96/64/32) are built and reproduce the implementation harness digest-for-digest.\n- Two full-width synthetic timing sweeps show near-linear execution (`R²={r2:.6f}`) and safe memory.\n- Historical exposed-data evidence gives MSE scaling exponents {power['greedy']:.3f}–{power['greedy_swaps']:.3f}; because these exceed one, the local adjusted frontier favors 129 bases. This is a prior, not the official answer.\n- The no-op calibration arm is output-identical on two complete networks and differs by exactly 2.147483648B intended tracked FLOPs.\n- A42/A43 are locally bit-identical; A43 saves 0.524123904B tracked operations and reduces peak memory by about {(1-374.8/2172)*100:.1f}%.\n- Every historical cost gate has been repriced against the current ~174.5B operating point. A generic +14B method now needs at least 1.08023× raw-MSE gain before wall effects.\n\n### Decisions\n\n- **T0.1: BLOCKED.** Proxy evidence says preserve 129; official four-arm curve is still decisive.\n- **T0.2: BLOCKED.** Calibration packages complete; only official paired measurement remains.\n- **T0.3: BLOCKED promotion / RETAIN MODULE.** Keep A42/A43 streaming for memory; official score chooses the production kernel.\n\nNo additional local basis-order sweep, compiler variant, null-work design, or wall benchmark can resolve the missing official quantities.\n''')
# reproducible scripts copy
for src,name in [('/mnt/data/run_t0_package.py','run_t0_package.py'),('/mnt/data/t0_shape_trace.py','shape_trace.py'),('/mnt/data/build_t0_packages.py','build_t0_packages.py'),('/mnt/data/build_t0_floor_diag.py','build_floor_diagnostics.py')]:shutil.copy2(src,ROOT/'scripts'/name)
# Main machine result
main={'status':'BLOCKED_OFFICIAL_DEPENDENCY','decisions':{'T0.1':'BLOCKED','T0.2':'BLOCKED','T0.3':'BLOCKED_PROMOTION_RETAIN_MODULE'},'unique_official_packages':7,'package_audit_pass':all(x.get('safe_paths') and x.get('manifest_hashes_ok') for x in package_audit),'current_position':current,'local_experiments':{'full_depth_basis_runs':8,'package_parity_runs':4,'full_depth_calibration_pair_runs':4,'common_T03_panel_runs':3,'floor_diagnostic_runs':2},'exact_missing_dependencies':['whestbench==0.13.0','flopscope==0.9.1','official Phase-1 Mini cohort / official submission runner']}
dump(ROOT/'RESULTS.json',main)
dump(ROOT/'COST_MODEL.json',{'T0.1':cost1,'T0.2':cost2,'T0.3':cost3})
# hashes after all non-hash files
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['FREEZE_HASHES.sha256','T0_BUNDLE.zip']]
(ROOT/'FREEZE_HASHES.sha256').write_text('\n'.join(f'{sha(p)}  {p.relative_to(ROOT)}' for p in sorted(files))+'\n')
# final zip (exclude source dirs? include everything)
zip_path=Path('/mnt/data/T0_official_grader_instrumentation_20260729.zip')
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(ROOT.rglob('*')):
  if p.is_file():z.write(p,arcname=str(Path(ROOT.name)/p.relative_to(ROOT)))
print(json.dumps({'root':str(ROOT),'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size,'zip_sha256':sha(zip_path),'package_audit_pass':main['package_audit_pass']},indent=2))

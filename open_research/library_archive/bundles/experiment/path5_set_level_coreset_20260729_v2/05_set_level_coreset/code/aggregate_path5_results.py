from pathlib import Path
import json,glob,math
import numpy as np
R=Path('/mnt/data/path5_work')

def stat(v):
 v=np.asarray(v,dtype=float)
 return {'n':len(v),'mean':float(v.mean()),'median':float(np.median(v)),'worst':float(v.max()),'passes_1.1e-8':int((v<=1.1e-8).sum()),'passes_2.2e-8':int((v<=2.2e-8).sum()),'values':v.tolist()}
# full 128 support library on dev8
full_dev=[]; pass_counts=[]
for s in range(64000,64008):
 with np.load(R/f'dataset_q128_c64/setlevel_seed{s}_q128_c64.npz',allow_pickle=False) as z:lab=z['labels'].copy()
 full_dev.append(float(lab.min()));pass_counts.append({'seed':s,'pass11':int((lab<=1.1e-8).sum()),'pass22':int((lab<=2.2e-8).sum()),'best':float(lab.min()),'median':float(np.median(lab))})
# top8 dev
D=json.load(open(R/'top8_sketch_dev8.json'))
# train32 top8 coverage
train_best=[];train_pass=[]
for p in sorted((R/'top8_rank_train32').glob('top8_rank_seed*.npz')):
 with np.load(p,allow_pickle=False) as z:lab=z['labels'].copy()
 seed=int(p.stem.split('seed')[1]);train_best.append(float(lab.min()));train_pass.append({'seed':seed,'pass11':int((lab<=1.1e-8).sum()),'pass22':int((lab<=2.2e-8).sum()),'best':float(lab.min())})
# old partially exposed validation merge
A=json.load(open(R/'frozen_support_validation_64100_64115.json'))['records']
B=json.load(open(R/'frozen_support_validation_64111_64115.json'))['records']
records=A+B
cand81=[r['values']['dev_selected_candidate81']['mse'] for r in records]
top8old=[r['values']['oracle_best_of_dev_top8']['mse'] for r in records]
# selector results
allrank=json.load(open(R/'direct_sketch_ranker_dev8_aggregate.json'))['ranked_summary']
top8sum=D['summary']
ml=json.load(open(R/'top8_ranker_light_results.json'))
# portfolio misses full sweep
miss=[]
for s in [64320,64331]:miss.append(json.load(open(R/f'oracle_sweep_{s}_q128_c64.json'))['summary'])
N=66048;d=256;mrows=8192;pilot=2064
full_final=2*N*d*d
cost={}
for q in [32,128]:
 sketch=2*N*d*q;pilot_cost=2*pilot*d*d;selected=2*mrows*d*d;score=8*2*4096*q*q
 total=sketch+pilot_cost+selected+score
 cost[str(q)]={'full_final_layer_flops':full_final,'pilot_coordinate_flops_approx':pilot_cost,'all_row_sketch_flops':sketch,'selected_full_output_flops':selected,'eight_support_score_ops_approx':score,'total_approx':total,'net_saved_approx':full_final-total,'fraction_of_dense_final':total/full_final,'fraction_of_175.5B_baseline_saved':(full_final-total)/175.5e9}
out={
 'status':'pause_tested_selector_families_preserve_fixed_support_library',
 'gates':{'primary_same_support_oracle_mse':1.1e-8,'secondary_tail':2.2e-8,'relative_weight_bounds':[0.05,4.0],'ess_floor':0.8},
 'support_library':{'size':128,'families':{'fixed_balanced_random':64,'affine_stratified':64},'pairs_per_support':4096,'rows_per_support':8192,'basis_count':129,'top8_candidate_ids':[81,111,88,35,91,78,51,38]},
 'portfolio_existence':{
  'smoke_63998':json.load(open(R/'oracle_sweep_63998_q128_c64.json'))['summary'],
  'development_64000_64007_full128_best':stat(full_dev),
  'development_64000_64007_candidate_counts':pass_counts,
  'development_top8_oracle_best':top8sum['oracle_best_top8'],
  'new_training_64300_64331_top8_oracle_best':stat(train_best),
  'new_training_top8_candidate_counts':train_pass,
  'full128_recovery_on_top8_misses':miss,
  'previously_exposed_validation_64100_64115_top8_oracle_best':stat(top8old),
 },
 'fixed_support_generalization':{'candidate81_previous_validation':stat(cand81)},
 'selectors':{
  'full128_direct_sketch_best_development_rule':allrank[0],
  'top8_direct_sketch_q128_r1e-4':top8sum['q128_r0.0001'],
  'top8_direct_sketch_q32_r1':top8sum['q32_r1'],
  'learned_ranker_frozen':ml['best'],
 },
 'compute':cost,
 'split_hygiene':{'development_seeds':[63998,*range(64000,64008)],'new_ranker_training_seeds':list(range(64300,64332)),'previously_exposed_validation_seeds':list(range(64100,64116)),'new_untouched_test_opened':False,'protected_official_or_mini_opened':False},
 'conclusion':{'support_portfolio_gate':'passes strongly for full 128-support library on every evaluated network with full sweep','selector_gate':'fails','runtime_gate':'q32/top8 arithmetic passes but statistical gate fails; q128/full128 scan is not affordable','next_action':'preserve library and labels; pause nearby sketch/ridge/tree sweeps; reopen only with a qualitatively new earlier-layer set predictor or much stronger direct support-error model'}
}
(R/'PATH5_SET_LEVEL_CORESET_RESULTS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'full_dev':out['portfolio_existence']['development_64000_64007_full128_best'],'train32':out['portfolio_existence']['new_training_64300_64331_top8_oracle_best'],'oldval':out['portfolio_existence']['previously_exposed_validation_64100_64115_top8_oracle_best'],'candidate81':out['fixed_support_generalization']['candidate81_previous_validation'],'selectors':out['selectors'],'compute':out['compute']},indent=2))

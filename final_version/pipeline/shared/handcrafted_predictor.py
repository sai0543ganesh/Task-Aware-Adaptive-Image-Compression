"""Frozen28-handcrafted-input MLP with the report quality rule; no CNN features."""
import gc,json,math,pathlib,time
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits
import simple_grouped_core as c
import simple_grouped_pipeline as features
from dense_heavy_paired_study import apply_rule
SEEDS=[42]
DECISION_RULE='Checkpoint 1 geometry floor, +1 and 0.27 promotion'
CFG=dict(hidden_layers=[128,64],activation='relu',alpha=.02,batch_size=256,learning_rate=.001,max_iter=80,early_stopping=False,selected_on_development_job=6260,decision_rule=DECISION_RULE)
APPEARANCE=['source_colorfulness','original_laplacian_variance','original_texture_mean','cheap_v3_color_corr_y1_7','original_jpeg_vertical_block_diff','original_saturation_high_fraction','source_jpeg_horizontal_block_diff','original_tenengrad_mean','source_gradient_kurtosis','source_gradient_skewness','cheap_v3_full_zlib_ratio','original_edge_density','source_laplacian_abs_mean','source_gradient_p90']

def geometry(x,names,sides):
 h=x[:,names.index('original_height')];w=x[:,names.index('original_width')];rows=[]
 for hh,ww in zip(h,w):
  row=[]
  for q in c.TIERS:
   nh=max(1,round(float(hh)*math.sqrt(q/100)));nw=max(1,round(float(ww)*math.sqrt(q/100)))
   choices=[s for s in sides if s*s<=nh*nw];row.append(math.log(max(choices) if choices else min(sides)))
  rows.append(row+[math.log(max(hh,ww)/max(1,min(hh,ww))),math.log(max(1,hh*ww)),math.log(max(1,min(hh,ww)))])
 names_extra=['mapped_log_side_'+str(i) for i in range(11)]+['source_log_aspect','source_log_area','source_log_short']
 return np.asarray(rows,np.float32),names_extra

def matrix(frame,sides):
 extra,en=geometry(frame.values,list(frame.columns),sides);names=list(frame.columns)+en
 chosen=APPEARANCE+en;keep=[names.index(name) for name in chosen]
 assert len(chosen)==28 and not any(name.startswith(('cheap_v3_mobile_','extra_cam_')) for name in chosen)
 return np.column_stack([frame.values,extra])[:,keep],chosen

def make_model():
 return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',QuantileTransformer(n_quantiles=128,output_distribution='normal',random_state=42)),('model',MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',alpha=.02,batch_size=256,learning_rate_init=.001,max_iter=80,early_stopping=False,n_iter_no_change=10,random_state=42))])

def probabilities(model,x,floor):
 p=np.zeros((len(x),11),np.float64);p[:,model.named_steps['model'].classes_.astype(int)]=model.predict_proba(x)
 p[np.arange(11)[None,:]<floor[:,None]]=0;empty=p.sum(1)==0;p[empty,10]=1
 return p/p.sum(1,keepdims=True)

def applied(p,floor):return apply_rule(p,floor)[2]

def fit_mlp(frame,lab,policy,directory,teacher_path):
 directory.mkdir();start=time.perf_counter();x,names=matrix(frame,policy['sides'])
 fit=np.asarray(policy['fit_indices']);hold=np.asarray(policy['holdout_indices']);fit=fit[lab['eligible'][fit]];ev=hold[lab['eligible'][hold]]
 with threadpool_limits(limits=4):
  evaluation=make_model();evaluation.fit(x[fit],lab['y'][fit]);p=probabilities(evaluation,x[ev],lab['floor'][ev])
 raw=p.argmax(1);actual=applied(p,lab['floor'][ev]);truth=lab['y'][ev]
 diag=dict(raw=dict(images=len(ev),exact_percent=100*float((raw==truth).mean()),within_one_percent=100*float((np.abs(raw-truth)<=1).mean()),unsafe_at_least_two_tiers_percent=100*float((raw<=truth-2).mean())),
  resolved_holdout=len(ev),holdout_total=len(hold),holdout_unresolved=len(hold)-len(ev),coverage_percent=100*len(ev)/len(hold),fit_rows=len(fit),
  applied_unsafe_percent=100*float((actual<=truth-2).mean()),actual_teacher_failure_at_applied_percent=100*float((~lab['response'][ev,actual]).mean()),
  evaluated_feature_count=28,evaluation_features=names,holdout_not_used_for_model_or_threshold_selection=True,diagnostic_identities_previously_reused=True,
  fixed_rule='Checkpoint 1 geometry floor, +1 and 0.27 promotion',evaluation_fit_and_scoring_seconds=time.perf_counter()-start)
 np.savez_compressed(directory/'holdout_predictions.npz',indices=ev,p=p,y=truth,raw=raw,floor=lab['floor'][ev],applied=actual)
 c.dump(directory/'MLP_EVALUATION.json',diag)
 refit=time.perf_counter();full=np.flatnonzero(lab['eligible'])
 with threadpool_limits(limits=4):deployment=make_model();deployment.fit(x[full],lab['y'][full])
 ck=dict(predictor=deployment,feature_columns=names,config=CFG,epochs=int(deployment.named_steps['model'].n_iter_),full_refit_rows=len(full),tiers=c.TIERS.tolist(),parent_teacher=str(teacher_path),parent_teacher_sha256=c.sha(teacher_path))
 torch.save(ck,directory/'deployed_seed42.pt')
 timing=dict(total_seconds=time.perf_counter()-start,refit_seconds=time.perf_counter()-refit,features=28,full_refit_rows=len(full),new_teacher_new_labels_new_weights=True)
 c.dump(directory/'results.json',dict(diagnostic=diag,timing=timing));return [ck],diag,timing

def predict80(source,checkpoints,policy,columns,directory):
 directory.mkdir();start=time.perf_counter();assert len(checkpoints)==1
 frame,_,ft=features.extract(source,columns,False,directory,'source80k');x,names=matrix(frame,policy['sides']);ck=checkpoints[0]
 assert names==ck['feature_columns'];floor=c.floor_indices(source.shapes);tick=time.perf_counter()
 with threadpool_limits(limits=4):p=probabilities(ck['predictor'],x,floor)
 raw=p.argmax(1);chosen=applied(p,floor)
 np.savez_compressed(directory/'predictions80k.npz',p=p,raw=raw,floor=floor,applied=chosen)
 report=dict(images=len(source),feature_time=ft,inference_and_export_seconds=time.perf_counter()-tick,complete_prediction_seconds=time.perf_counter()-start,
  selected_feature_count=28,source_MobileNet_passes_per_image=0,ResNet_feature_passes_on80k=0,feature_loader_batch=128,feature_loader_workers=8,
  raw_counts=np.bincount(raw,minlength=11).tolist(),applied_counts=np.bincount(chosen,minlength=11).tolist(),decision_rule=DECISION_RULE,no80k_teacher_ground_truth=True)
 c.dump(directory/'prediction_results.json',report);return chosen,report

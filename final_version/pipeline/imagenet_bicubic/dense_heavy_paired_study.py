"""Fresh matched baseline/teacher/MLP/80k comparisons for non-ResNet heavy classifiers.

No inherited ResNet teacher labels or MLP weights. One allocation, fixed source
identities/grid/splits/recipe; report complete measurements including failures.
"""
import time
PROGRAM_START=time.perf_counter()
import csv,gc,hashlib,json,math,os,platform,subprocess,sys,tarfile,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import PIL,torch,torchvision,sklearn
from torch import nn
from torchvision import models
import simple_grouped_core as c
import augmented_policy as mapping
import shared_resnet_policy as rp
import search_definition as mlp
import simple_grouped_pipeline as features
import cross_backbone_execution as execution
import saved5327_encoded_ram as encoded
import bounded_source_cache as source_cache
from parallel_5327_mapping import map_selected_images
from resource_monitor import Monitor

ARCHITECTURES=['densenet121']
ROOT=Path(os.environ['MTP_RESEARCH_OUTPUT_ROOT'])/('job_'+os.environ['SLURM_JOB_ID'])
CFG=dict(width=128,depth=1,drop=.3,lr=.0006,wd=.02,noise=.015,aux=.1,near=.25,
         ordinal=.1,k=0,transform='quantile',periodic=False,name='fixed_candidate6')
SEEDS=[42,43,44]
THRESHOLD=.27

def status(stage,**values):
    c.dump(ROOT/'status.json',dict(stage=stage,**values));c.say(stage,**values)

def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def fresh_architecture(name,classes,device='cuda'):
    if name not in ARCHITECTURES:raise ValueError(name)
    c.seed()
    net=getattr(models,name)(weights=None,num_classes=len(classes)).to(device)
    assert net.__class__.__name__ in ['DenseNet','ConvNeXt'],'Unexpected task backbone'
    net._backbone_architecture=name
    net._native_bn_spec=[(m.eps,m.momentum) for m in net.modules() if isinstance(m,nn.BatchNorm2d)]
    return net

def unpack():
    start=time.perf_counter();saved=ROOT/'inputs';saved.mkdir()
    with tarfile.open(Path(os.environ['MTP_RESEARCH_INPUT_DIR'])/'cross_backbone_inputs.tar.gz') as bank:
        for m in bank.getmembers():
            assert m.isfile() and (saved/m.name).resolve().is_relative_to(saved.resolve()),m.name
        bank.extractall(saved)
    meta=json.loads((saved/'ASSET_MANIFEST.json').read_text())
    for name,digest in meta['sha256'].items():assert c.sha(saved/name)==digest,name
    tables={n:read(saved/n) for n in ['sources.csv','validation.csv','test.csv']}
    classes=json.loads((saved/'protocol.json').read_text())['classes']
    policy=json.loads((saved/'grid_split_policy.json').read_text())
    assert [len(tables[n]) for n in ['sources.csv','validation.csv','test.csv']]==[100000,1000,5000]
    assert len(classes)==100 and policy['sides']==[32,40,48,56,72,96,120,136,160,168,224]
    sets=[{r.get('source_sha256',r.get('sha256')) for r in tables[n]} for n in ['sources.csv','validation.csv','test.csv']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    def verify(row):
        raw=Path(row['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row.get('source_sha256',row.get('sha256')),row['path']
        assert classes[int(row['label'])]==row['class_name']
        return len(raw)
    total=0
    for records in tables.values():
        with ThreadPoolExecutor(max_workers=16) as pool:total+=sum(pool.map(verify,records))
    initial=[r for r in tables['sources.csv'] if r['part']=='teacher']
    rest=[r for r in tables['sources.csv'] if r['part']=='remaining']
    assert len(initial)==20000 and len(rest)==80000
    fit=np.asarray(policy['fit_indices']);hold=np.asarray(policy['holdout_indices'])
    assert len(fit)==16000 and len(hold)==4000
    assert sorted(np.r_[fit,hold].tolist())==list(range(20000))
    groups=np.array([r['source_sha256'] for r in initial])
    assert not set(groups[fit])&set(groups[hold])
    columns=[r['feature'] for r in read(saved/'optimized_v28_features.csv')]
    assert len(columns)==215
    common=time.perf_counter()-PROGRAM_START
    c.dump(ROOT/'input_checks.json',dict(hashes_verified=106000,bytes_read=total,source_hash_disjoint=True,
        MLP_nominal_fit=16000,MLP_nominal_holdout=4000,MLP_group_disjoint=True,
        common_setup_seconds=common,source_images_read_only=True,old_teacher_labels_or_weights_imported=False))
    return tables,classes,policy,columns,common

def label_teacher(net,source,policy,directory):
    directory.mkdir();started=time.perf_counter();floor=c.floor_indices(source.shapes)
    allp=np.empty((len(source),11,100),np.float32);stages=[];forwards=0
    # Only floor-clamped requests are reused. Same-sized but differently
    # prefiltered nominal tiers are NOT presumed to contain equal pixels.
    for tier in reversed(range(11)):
        active=np.flatnonzero(floor<=tier);reuse=np.flatnonzero(floor>tier)
        if len(active):
            before=time.perf_counter();subset=c.SubsetRAM(source,active)
            data=map_selected_images(subset,np.full(len(active),tier,dtype=np.int64),policy['sides'],floor[active],4,c.RAMImages)
            mapped=time.perf_counter()-before
            metric,p=c.evaluate(net,data,probabilities=True);allp[active,tier]=p;forwards+=len(active)
            stages.append(dict(tier=tier,images=len(active),mapping_seconds=mapped,evaluation=metric))
            del data,subset,p;gc.collect()
        if len(reuse):allp[reuse,tier]=allp[reuse,floor[reuse]]
        status('TEACHER_LABEL_TIER',architecture=net._backbone_architecture,tier=tier,actual_forward_images=forwards)
    assert np.isfinite(allp).all() and np.allclose(allp.sum(2),1,atol=1e-4)
    truth=np.array([int(r['label']) for r in source.rows]);response=allp.argmax(2)==truth[:,None]
    eligible=response[:,-1];suffix=np.logical_and.accumulate(response[:,::-1],axis=1)[:,::-1]
    allowed=suffix&(np.arange(11)[None,:]>=floor[:,None]);y=np.where(eligible,allowed.argmax(1),10).astype(np.int64)
    proxy=np.empty((len(source),36),np.float32)
    for j,tier in enumerate([0,2,3,4,7,8]):
        conf=allp[:,tier].max(1);good=response[:,tier].astype(np.float32)
        agree=(allp[:,tier].argmax(1)==allp[:,-1].argmax(1)).astype(np.float32)
        proxy[:,6*j:6*j+6]=np.stack([conf,good,agree,conf*good,conf*agree,
          np.full(len(source),math.log1p(c.TIERS[tier])/math.log1p(100))],1)
    lab=dict(y=y,response=response,eligible=eligible,floor=floor,soft=allp[:,-1].copy(),proxy=proxy)
    np.savez_compressed(directory/'teacher_labels.npz',**lab,identities=np.array([r['id'] for r in source.rows]))
    info=dict(seconds=time.perf_counter()-started,resolved=int(eligible.sum()),unresolved=int((~eligible).sum()),
      actual_forward_images=forwards,nominal_requests=len(source)*11,stages=stages,
      target='minimum feasible all-correct suffix; exclude full-quality-incorrect source examples')
    c.dump(directory/'results.json',info);del allp;return lab,info

def apply_rule(p,floor):
    raw=p.argmax(1);base=np.maximum(np.minimum(raw+1,10),floor)
    risk=(p*(np.arange(11)[None,:]>=base[:,None]+2)).sum(1)
    return raw,base,np.where(risk>THRESHOLD,np.minimum(base+1,10),base)

def fit_mlp(frame,lab,policy,directory,teacher_path):
    directory.mkdir();started=time.perf_counter()
    extra,extra_names=mlp.geometry_features(frame.values,list(frame.columns),policy['sides'])
    x=np.column_stack([frame.values,extra]);names=list(frame.columns)+extra_names
    assert x.shape==(20000,229) and np.isfinite(x).all()
    fit=np.asarray(policy['fit_indices']);hold=np.asarray(policy['holdout_indices'])
    fit=fit[lab['eligible'][fit]];ev=hold[lab['eligible'][hold]]
    if len(fit)<1000 or len(ev)<100:raise RuntimeError('Insufficient resolved teacher labels for meaningful MLP evaluation')
    z,state=mlp.preprocess(x,lab['y'],fit,CFG,names);evaluation_names=state['feature_columns']
    predictions=[]
    for seed in SEEDS:
        net,epochs,_=mlp.fit(z,lab['y'],lab['floor'],lab['soft'],lab['proxy'],fit,ev[:1],CFG,seed,fixed=3)
        predictions.append(mlp.probabilities(net,torch.tensor(z[ev],device='cuda'),torch.tensor(lab['floor'][ev],device='cuda')))
        torch.save(dict(model=net.state_dict(),preprocessing=state,config=CFG,epochs=epochs,
            parent_teacher_sha256=c.sha(teacher_path)),directory/('evaluation_seed'+str(seed)+'.pt'))
        del net;gc.collect();torch.cuda.empty_cache()
    p=np.mean(predictions,axis=0);raw,base,applied=apply_rule(p,lab['floor'][ev])
    diagnostic=dict(raw=mlp.metrics(lab['y'][ev],raw),resolved_holdout=len(ev),holdout_total=len(hold),
        holdout_unresolved=len(hold)-len(ev),coverage_percent=100*len(ev)/len(hold),fit_rows=len(fit),
        plus1_unsafe_percent=100*float((base<=lab['y'][ev]-2).mean()),
        applied_unsafe_percent=100*float((applied<=lab['y'][ev]-2).mean()),
        actual_teacher_failure_at_applied_percent=100*float((~lab['response'][ev,applied]).mean()),
        evaluated_feature_count=z.shape[1],evaluation_features=evaluation_names,
        holdout_not_used_for_model_or_threshold_selection=True,diagnostic_identities_previously_reused=True)
    np.savez_compressed(directory/'holdout_predictions.npz',indices=ev,p=p,y=lab['y'][ev],raw=raw,
        floor=lab['floor'][ev],base=base,applied=applied)
    diagnostic['evaluation_fit_and_scoring_seconds']=time.perf_counter()-started
    c.dump(directory/'MLP_EVALUATION.json',diagnostic)
    full=np.flatnonzero(lab['eligible']);refit_start=time.perf_counter()
    z,state=mlp.preprocess(x,lab['y'],full,CFG,names);checkpoints=[]
    if state['feature_columns']!=evaluation_names:raise RuntimeError('Evaluation/refit feature identity changed')
    # On the same source images the recipe normally retains222 features. Record
    # actual count instead of assigning a historical count if constants differ.
    for seed in SEEDS:
        net,epochs,_=mlp.fit(z,lab['y'],lab['floor'],lab['soft'],lab['proxy'],full,full[:1],CFG,seed,fixed=3)
        ck=dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},preprocessing=state,
            config=CFG,epochs=epochs,full_refit_rows=len(full),tiers=c.TIERS.tolist(),
            parent_teacher_sha256=c.sha(teacher_path),parent_teacher=str(teacher_path))
        torch.save(ck,directory/('deployed_seed'+str(seed)+'.pt'));checkpoints.append(ck)
        del net;gc.collect();torch.cuda.empty_cache()
    with (directory/'selected_features.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['input_index','feature']);w.writerows(enumerate(state['feature_columns']))
    timing=dict(total_seconds=time.perf_counter()-started,refit_seconds=time.perf_counter()-refit_start,
        full_refit_rows=len(full),features=len(state['feature_columns']),same_recipe_as_5509=True,
        new_teacher_new_labels_new_weights=True)
    c.dump(directory/'results.json',dict(diagnostic=diagnostic,timing=timing))
    return checkpoints,diagnostic,timing

def proposed(name,tables,classes,policy,columns,directory,common,monitor):
    started=time.perf_counter();directory.mkdir();execution.OUT=directory
    monitor.set_phase('proposed_teacher_source_cache');status('INITIAL20K_TEACHER',architecture=name)
    initial=[r for r in tables['sources.csv'] if r['part']=='teacher']
    source,teacher_cache=source_cache.cache(initial,directory,'initial20k',workers=16)
    val=execution.fixed(tables['validation.csv'],directory,'teacher_validation')
    monitor.set_phase('proposed_teacher_training');net=c.fresh_model(classes)
    teacher=execution.train(net,source,val,classes,directory/'teacher','teacher')
    teacher['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    monitor.set_phase('proposed_teacher_labels');status('NEW_TEACHER_LABELS',architecture=name)
    lab,label_time=label_teacher(net,source,policy,directory/'labels')
    del net,val;gc.collect();torch.cuda.empty_cache()
    monitor.set_phase('proposed_initial_features');status('INITIAL20K_FEATURES',architecture=name)
    frame,_,feature_time=features.extract(source,columns,True,directory,'initial20k')
    del source;gc.collect();torch.cuda.empty_cache()
    monitor.set_phase('proposed_MLP_training');status('NEW_MLP_FIT',architecture=name)
    checkpoints,diagnostic,mlp_time=fit_mlp(frame,lab,policy,directory/'mlp',directory/'teacher/best.pth')
    del frame,lab;gc.collect();torch.cuda.empty_cache()
    monitor.set_phase('proposed_remaining_cache');remaining=[r for r in tables['sources.csv'] if r['part']=='remaining']
    source,cache=source_cache.cache(remaining,directory,'remaining80k_source',workers=16)
    monitor.set_phase('proposed_80k_features_prediction');status('REMAINING80K_PREDICTION',architecture=name)
    applied,prediction=execution.predict80(source,checkpoints,policy,columns,directory/'prediction')
    # predict80's old report label is replaced with the actual feature count.
    prediction['selected_feature_count']=len(checkpoints[0]['preprocessing']['feature_columns'])
    c.dump(directory/'prediction/prediction_results.json',prediction)
    del checkpoints;gc.collect();torch.cuda.empty_cache()
    monitor.set_phase('proposed_mapping');tick=time.perf_counter()
    data=map_selected_images(source,applied,policy['sides'],c.floor_indices(source.shapes),4,c.RAMImages)
    storage=dict(mapping_seconds=time.perf_counter()-tick,source_encoded_image_payload_bytes=cache['source_payload_bytes'],
        source_decoded_RGB_bytes=sum(h*w*3 for h,w in source.shapes),mapped_decoded_RGB_bytes=data.bytes,
        mapped_geometry=c.geometry(data),persisted_compressed_image_payload_bytes=0)
    with (directory/'applied80k_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','source_h','source_w','tier','percent','mapped_h','mapped_w'])
        for i,r in enumerate(source.rows):w.writerow([r['id'],*source.shapes[i],int(applied[i]),float(c.TIERS[applied[i]]),*data.shapes[i]])
    c.dump(directory/'storage.json',storage);del source;gc.collect();torch.cuda.empty_cache()
    val=execution.fixed(tables['validation.csv'],directory,'student_validation')
    ck=torch.load(directory/'teacher/best.pth',map_location='cpu',weights_only=True)
    assert ck['classes']==classes and ck['backbone']==name
    net=c.fresh_model(classes);net.load_state_dict(ck['model_state_dict']);del ck
    monitor.set_phase('proposed_continuation_training');status('CONTINUATION80K_TRAINING',architecture=name)
    training=execution.train(net,data,val,classes,directory/'training','proposed',guard=None)
    monitor.set_phase('proposed_test');test=execution.fixed(tables['test.csv'],directory,'test')
    test_metric=c.evaluate(net,test,directory/'test_predictions.npz');monitor.sample()
    own=time.perf_counter()-started
    result=dict(architecture=name,teacher=teacher,teacher_cache=teacher_cache,labeling=label_time,
        initial_feature_time=feature_time,MLP=diagnostic,MLP_training=mlp_time,prediction=prediction,
        storage=storage,remaining_source_cache=cache,training=training,test=test_metric,
        fresh_pipeline_wall_seconds=own,common_setup_seconds=common,complete_accounted_seconds=own+common,
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        timing='Fresh teacher-to-student single-allocation wall time plus equally charged common setup; no reused model creation timings')
    c.dump(directory/'results.json',result)
    del net,data,val,test;gc.collect();torch.cuda.empty_cache();return result

def gpu_capacity_preflight(name,classes,directory):
    """One disposable real64x224 optimizer step before any task training."""
    started=time.perf_counter();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    net=fresh_architecture(name,classes);net.train()
    optimizer=execution.task_optimizer(net,'ordinary')
    x=torch.randn(64,3,224,224,device='cuda');y=torch.arange(64,device='cuda')%len(classes)
    loss=nn.functional.cross_entropy(net(x),y);assert torch.isfinite(loss)
    loss.backward();optimizer.step();c.sync()
    value=dict(passed=True,synthetic_only=True,batch=64,shape=[64,3,224,224],
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-started,
        native_BN=net._native_bn_spec,architecture=name,no_task_weights_or_data_reused=True)
    c.dump(directory/'GPU_CAPACITY_PREFLIGHT.json',value)
    del loss,optimizer,x,y,net;gc.collect();torch.cuda.empty_cache();c.seed();return value


def analyze_result(row,directory):
    base=row['baseline'];prop=row['proposed'];notes=[]
    if row['time_saved_percent']<=0:notes.append('No complete-pipeline speed saving observed. Inspect measured teacher,label,feature and training costs before changing the method.')
    if row['top1_drop_percentage_points']>5:notes.append('Top1 loss exceeds the reference5point tolerance. Inspect validation curves,teacher correctness,MLP coverage and final resolution distributions; test score is not a tuning target.')
    diag=prop['MLP']
    if diag['applied_unsafe_percent']>=5:notes.append('Applied diagnostic ordinal unsafe exceeds5%; future risk calibration must use a development subset, not this holdout or final classification test.')
    if diag['coverage_percent']<80:notes.append('Low teacher-resolved coverage; conditional MLP percentages are not all-image accuracy. Inspect teacher correctness before adding features.')
    h=json.loads((directory/'proposed/training/history.json').read_text())
    val=[(r['epoch'],r['validation']['top1_percent']) for r in h if 'validation' in r]
    peak=max(val,key=lambda v:v[1]);last=val[-1]
    if peak[1]-last[1]>3:notes.append('Validation fell over3points from its recorded peak. This is evidence for investigating schedule/regularization, not proof of one cause.')
    report=dict(time_saved_percent=row['time_saved_percent'],top1_drop_points=row['top1_drop_percentage_points'],
        baseline_seconds=base['accounted_complete_seconds'],proposed_seconds=prop['complete_accounted_seconds'],
        MLP_exact=diag['raw']['exact_percent'],MLP_within_one=diag['raw']['within_one_percent'],
        applied_unsafe=diag['applied_unsafe_percent'],resolved_coverage=diag['coverage_percent'],
        validation_peak=peak,validation_final=last,observations=notes,
        no_test_based_hyperparameter_changes=True,next_step='Review per-stage evidence; only bounded development-validated followups, preserving this comparison')
    c.dump(directory/'analysis.json',report)


def cache_preflight():
    import contextlib,io
    directory=ROOT.parent/'source_cache_checks';directory.mkdir(exist_ok=False)
    rng=np.random.default_rng(42);rows=[]
    for i,(h,w) in enumerate([(96,128),(224,224),(128,320)]):
        p=directory/('source'+str(i)+'.jpg');Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8)).save(p)
        rows.append(dict(path=str(p),source_sha256=c.sha(p),bytes=str(p.stat().st_size),label='0',id=str(i)))
    original=source_cache.DECODED_CACHE_LIMIT_BYTES;outputs={};modes={}
    try:
        for limit,name in [(0,'encoded'),(16*2**30,'decoded')]:
            source_cache.DECODED_CACHE_LIMIT_BYTES=limit
            with contextlib.redirect_stdout(io.StringIO()):data,info=source_cache.cache(rows,directory,name,workers=4)
            outputs[name]=[data.rgb(i).copy() for i in range(len(rows))];modes[name]=info['source_cache_mode']
            del data;gc.collect()
        assert all(np.array_equal(a,b) for a,b in zip(outputs['encoded'],outputs['decoded']))
    finally:source_cache.DECODED_CACHE_LIMIT_BYTES=original
    return dict(both_cache_paths_passed=True,decoded_pixels_identical=True,images=len(rows),modes=modes,synthetic_only=True)


def cpu_preflight():
    """No source images or GPU required; verify installed constructors and native BN."""
    torch.set_num_threads(4);out={}
    for name in ARCHITECTURES:
        net=fresh_architecture(name,list(range(100)),'cpu').eval();shapes={}
        with torch.inference_mode():
            for side in sorted(set([32,40,48,56,72,96,120,136,160,168,224]+[32,48,64,80,96,128,144,160,176,224])):
                result=net(torch.zeros(2,3,side,side));assert result.shape==(2,100)
                assert torch.isfinite(result).all();shapes[str(side)]=list(result.shape)
        net.train();loss=net(torch.randn(2,3,32,32)).square().mean();loss.backward()
        assert [(m.eps,m.momentum) for m in net.modules() if isinstance(m,nn.BatchNorm2d)]==net._native_bn_spec
        out[name]=dict(parameter_count=sum(p.numel() for p in net.parameters()),native_BN=net._native_bn_spec,output_shapes=shapes,CPU_backward_passed=True)
        del net
    print(json.dumps(dict(versions=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,Pillow=PIL.__version__,sklearn=sklearn.__version__),checks=out,cache_checks=cache_preflight())))

def run():
    ROOT.mkdir(parents=True,exist_ok=False);torch.multiprocessing.set_start_method('fork',force=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');c.seed()
    assert torch.cuda.is_available() and any(kind in torch.cuda.get_device_name(0) for kind in ['A100','P100'])
    tables,classes,policy,columns,common=unpack();mapping.install_floor(policy['sides']);rp.install(c,'teacher')
    versions=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
        Pillow=PIL.__version__,numpy=np.__version__,sklearn=sklearn.__version__,CUDA=torch.version.cuda,
        cuDNN=torch.backends.cudnn.version(),node=os.environ.get('SLURMD_NODENAME'),GPU=torch.cuda.get_device_name(0))
    try:versions['nvidia_driver']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
    except Exception as e:versions['driver_query_error']=str(e)
    c.dump(ROOT/'protocol.json',dict(architectures=ARCHITECTURES,versions=versions,batch=64,train_workers=4,
        eval_workers=4,precision='FP32',CPUs=int(os.environ.get('SLURM_CPUS_PER_TASK',16)),RAM_requested_MiB=int(os.environ['SLURM_MEM_PER_NODE']),epochs_each_stage=50,
        TF32_enabled=False,source_pixel_storage='Linux fork COW read-only buffers',source_cache_policy='same16GiB decoded threshold; encoded RAM otherwise; no input resize',
        tiers=c.TIERS.tolist(),mapped_sides=policy['sides'],MLP_recipe=CFG,MLP_seeds=SEEDS,
        task_optimizers=dict(densenet121='SGD momentum0.9 wd0.0001 lr0.1 ordinary/0.01 continuation/0.001 adaptation',convnext_family='AdamW wd0.05 lr0.0005 ordinary/0.00005 continuation/0.00001 adaptation; no decay on1D/bias'),
        native_normalization=dict(densenet121='BatchNorm defaults retained',convnext_tiny='LayerNorm defaults retained',convnext_small='LayerNorm defaults retained'),
        safety_threshold=THRESHOLD,all_models_new_weights=True,each_pair_same_allocation=True,
        baseline_first=True,no_test_based_parameter_adjustment=True,
        test_and_MLP_diagnostic_identities_previously_reused=True,
        logical_training_presentations_each_method=5000000,
        proposed='20k teacher50 -> new labels -> same feature/MLP recipe -> selective+1 -> 80k48+2',
        time_guard='No speed-target early rejection; measure negative results too. Slurm walltime is the resource limit.',
        source_code_sha256={p.name:c.sha(p) for p in Path(__file__).parent.glob('*.py')}))
    execution.reference.stream_cache=lambda rows:source_cache.cache(rows,execution.reference.OUT,'baseline100k_source',workers=16)
    results={};failures={}
    for name in ARCHITECTURES:
        model_start=time.perf_counter();directory=ROOT/name;directory.mkdir();execution.OUT=directory
        c.fresh_model=lambda classes,arch=name:fresh_architecture(arch,classes)
        monitor=Monitor(directory);monitor.thread.start()
        try:
            status('ARCHITECTURE_INTERFACE_CHECK',architecture=name)
            tick=time.perf_counter();capacity=gpu_capacity_preflight(name,classes,directory);execution.synthetic_interfaces(classes)
            pair_common=common+time.perf_counter()-tick
            status('BASELINE100K',architecture=name)
            base=execution.baseline(tables,classes,monitor)
            base['accounted_complete_seconds']=base['complete_arm_seconds']+pair_common
            c.dump(directory/'baseline/results.json',base)
            torch.cuda.reset_peak_memory_stats()
            prop=proposed(name,tables,classes,policy,columns,directory/'proposed',pair_common,monitor)
            saving=100*(1-prop['complete_accounted_seconds']/base['accounted_complete_seconds'])
            loss=base['test']['top1_percent']-prop['test']['top1_percent']
            row=dict(architecture=name,baseline=base,proposed=prop,time_saved_percent=saving,
                top1_drop_percentage_points=loss,observed_saving=saving>0,
                reference_18percent_5point_gate=saving>18 and loss<=5,
                model_program_seconds=time.perf_counter()-model_start)
            c.dump(directory/'comparison.json',row);results[name]=row
            analyze_result(row,directory)
            status('ARCHITECTURE_COMPLETE',architecture=name,saving_percent=saving,top1_drop_points=loss)
        except Exception:
            failures[name]=traceback.format_exc();c.dump(directory/'failure.json',dict(traceback=failures[name],outputs_preserved=True))
            status('ARCHITECTURE_FAILED',architecture=name)
        finally:
            monitor.stop.set();monitor.thread.join(timeout=10);gc.collect();torch.cuda.empty_cache()
            c.dump(ROOT/'results.json',dict(completed=results,failed=failures,partial=len(results)<len(ARCHITECTURES),
                total_program_seconds=time.perf_counter()-PROGRAM_START))
    status('COMPLETE' if not failures else 'COMPLETED_WITH_FAILURES',completed=list(results),failed=list(failures))
    if failures:raise RuntimeError('At least one architecture failed; inspect preserved per-architecture failure.json')

if __name__=='__main__':
    source_cache.install_fork_storage()
    if '--cache-preflight' in sys.argv:print(json.dumps(cache_preflight()))
    elif '--cpu-preflight' in sys.argv:cpu_preflight()
    else:
        try:run()
        except Exception:
            ROOT.mkdir(parents=True,exist_ok=True);c.dump(ROOT/'failure.json',dict(traceback=traceback.format_exc()));raise




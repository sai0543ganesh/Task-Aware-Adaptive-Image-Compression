"""5509 execution helpers: preserve each new backbone native BN, add architecture metadata.
The saved-ResNet run/predictor-setup functions are not called by the new study.
"""
"""One matched baseline/proposed confirmation, with frozen 5327/5364 parents.

No teacher retraining. Complete proposed accounting is an explicitly identified
sum of measured reused creation stages and the newly measured 80k arm. Current
baseline and continuation use the same GPU allocation, 64/4/FP32/standard BN.
"""
import time
PROCESS_START=time.perf_counter()
import csv,gc,hashlib,json,math,os,tarfile,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import simple_grouped_core as c
import augmented_policy as mapping
import shared_resnet_policy as rp
import head_adaptation as adaptation
import search_definition as predictor
import simple_grouped_pipeline as features
import baseline5345_reference as reference
import saved5327_encoded_ram as encoded
from parallel_5327_mapping import map_selected_images
from resource_monitor import Monitor

OUT=Path(os.environ['MTP_RESEARCH_OUTPUT_ROOT'])/('job_'+os.environ['SLURM_JOB_ID'])
SAFETY_THRESHOLD=.27
VALIDATE={1,2,3,4,5,10,15,20,25,30,35,40,45,49,50}


def task_learning_rate(net,stage):
    if net._backbone_architecture.startswith('convnext_'):return {'ordinary':.0005,'continuation':.00005,'adapt':.00001}[stage]
    return {'ordinary':.1,'continuation':.01,'adapt':.001}[stage]


def task_optimizer(net,stage):
    lr=task_learning_rate(net,stage)
    if net._backbone_architecture.startswith('convnext_'):
        decay=[];no_decay=[]
        for name,p in net.named_parameters():
            if p.requires_grad:(no_decay if p.ndim<=1 or name.endswith('.bias') else decay).append(p)
        return torch.optim.AdamW([{'params':decay,'weight_decay':.05},{'params':no_decay,'weight_decay':0.}],lr=lr)
    return torch.optim.SGD(net.parameters(),lr=lr,momentum=.9,weight_decay=1e-4)


def status(stage,**kw):
    c.dump(OUT/'status.json',dict(stage=stage,**kw));c.say(stage,**kw)


def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def unpack():
    dest=OUT/'saved_inputs';dest.mkdir()
    with tarfile.open(Path(os.environ['MTP_RESEARCH_INPUT_DIR'])/'saved5327_inputs.tar.gz') as bank:
        for m in bank.getmembers():
            assert m.isfile() and (dest/m.name).resolve().is_relative_to(dest.resolve()),m.name
        bank.extractall(dest)
    manifest=json.loads((dest/'ASSET_MANIFEST.json').read_text())
    for name,digest in manifest['sha256'].items():assert c.sha(dest/name)==digest,name
    return dest,manifest


def fixed(records,directory,name):
    source,info=c.cache(records,directory,name+'_source',workers=16)
    arrays=[np.array(source.image(i).resize((224,224),Image.Resampling.BILINEAR),dtype=np.uint8,copy=True) for i in range(len(source))]
    result=c.RAMImages(records,arrays);del source;gc.collect();return result


class TimeGuard(Exception):pass


def train(net,data,val,classes,directory,kind,guard=None,epochs=50,adaptation_epochs=2):
    """Shared execution loop; method-specific resolution/regularization declared."""
    directory.mkdir(parents=True,exist_ok=False)
    student=kind in ['proposed','interface_student']
    assert epochs>=1 and (not student or 0<adaptation_epochs<epochs)
    first_adaptation=epochs-adaptation_epochs+1 if student else epochs+1
    view=mapping.AugmentedView(data if student else mapping.CanonicalTrainingCanvas(data))
    stream,sampler=rp.common_loader(c,view,True)
    opt=task_optimizer(net,'continuation' if student else 'ordinary')
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs-adaptation_epochs if student else epochs)
    assert [(m.eps,m.momentum) for m in net.modules() if isinstance(m,nn.BatchNorm2d)]==net._native_bn_spec
    history=[];best=-1.;best5=-1.;bestepoch=0;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        if student and epoch==first_adaptation:
            del stream;gc.collect()
            view=adaptation.Fixed224(data);stream,sampler=rp.common_loader(c,view,True)
            opt=task_optimizer(net,'adapt')
            schedule=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.,total_iters=adaptation_epochs)
        sampler.epoch=epoch;net.train();c.sync();tick=time.perf_counter()
        total=steps=extra=0;loss_sum=torch.zeros((),device='cuda');correct=torch.zeros((),device='cuda')
        iterator_wait=0.;mark=time.perf_counter();lr=opt.param_groups[0]['lr']
        for x,y,_ in stream:
            iterator_wait+=time.perf_counter()-mark;count=len(y)
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            if count==1:
                x=x.repeat(2,1,1,1);y=y.repeat(2);extra+=1
            opt.zero_grad(set_to_none=True);logits=net(x)
            loss=nn.functional.cross_entropy(logits,y,label_smoothing=.05 if student and epoch<first_adaptation else 0.)
            loss.backward()
            if net._backbone_architecture.startswith('convnext_'):nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            loss_sum+=loss.detach()*count;correct+=(logits[:count].argmax(1)==y[:count]).sum()
            total+=count;steps+=1
            if guard and guard['fixed_seconds']+time.perf_counter()-guard['arm_start']>=guard['ceiling']:
                raise TimeGuard('Actual complete proposed time budget exhausted')
            mark=time.perf_counter()
        c.sync();seconds=time.perf_counter()-tick;schedule.step();assert total==len(data)
        item=dict(epoch=epoch,images=total,train_top1_percent=100*float(correct)/total,
                  loss=float(loss_sum)/total,training_seconds=seconds,data_iterator_wait_seconds=iterator_wait,
                  optimizer_steps=steps,maximum_batch_images=64,workers=4,precision='FP32',
                  extra_duplicate_forward_images=extra,actual_forward_images=total+extra,
                  learning_rate_applied=lr,label_smoothing=.05 if student and epoch<first_adaptation else 0.,
                  input_policy='direct224 adaptation' if student and epoch>=first_adaptation else 'assigned size RRC+flip' if student else 'source RRC224+flip')
        assert math.isfinite(item['loss']),'Nonfinite training loss'
        if epoch in VALIDATE or epoch==epochs:
            metric=c.evaluate(net,val);item['validation']=metric
            checkpoint=dict(backbone=net._backbone_architecture,optimizer_name=type(opt).__name__,model_state_dict=net.state_dict(),optimizer_state_dict=opt.state_dict(),
                            scheduler_state_dict=schedule.state_dict(),epoch=epoch,classes=classes,
                            input_policy=item['input_policy'],batch=64,workers=4,precision='FP32')
            torch.save(checkpoint,directory/'latest.pth')
            if metric['top1_percent']>best:
                best=metric['top1_percent'];bestepoch=epoch;torch.save(checkpoint,directory/'best.pth')
            if metric['top5_percent']>best5:
                best5=metric['top5_percent'];torch.save(checkpoint,directory/'best_top5.pth')
        history.append(item);c.dump(directory/'history.json',history);c.say('MATCHED_EPOCH',arm=kind,**item)
        status(kind.upper()+'_TRAINING',epoch=epoch,best_validation_top1=best)
        if guard and epoch>=5:
            recent=float(np.mean([h['training_seconds'] for h in history[-3:]]))
            remaining_small=max(0,48-epoch);remaining_adapt=min(2,50-epoch)
            pending_validation=sum(e>epoch for e in VALIDATE)
            projected=(guard['fixed_seconds']+time.perf_counter()-guard['arm_start']+
                       recent*remaining_small+guard['adapt_epoch_seconds']*remaining_adapt+
                       pending_validation*guard['validation_seconds']+guard['tail_reserve_seconds'])
            c.dump(directory/'timing_projection.json',dict(epoch=epoch,estimate_only=True,
                   projected_accounted_seconds=projected,time_ceiling=guard['ceiling'],
                   remaining_small_epochs=remaining_small,remaining_adaptation_epochs=remaining_adapt))
            if projected>=guard['ceiling']:
                raise TimeGuard('Projected complete proposed time exceeds the18percent-saving ceiling')
    ck=torch.load(directory/'best.pth',map_location='cpu',weights_only=True);net.load_state_dict(ck['model_state_dict'])
    del ck,stream;gc.collect()
    return dict(epochs=epochs,images=len(data),logical_training_presentations=len(data)*epochs,
                selected_epoch=bestepoch,best_validation_top1=best,best_recorded_validation_top5=best5,
                seconds=time.perf_counter()-start,training_loop_seconds=sum(h['training_seconds'] for h in history),
                actual_forward_presentations=sum(h['actual_forward_images'] for h in history),
                history_path=str(directory/'history.json'),checkpoint_selection='validation top1 only')


def synthetic_interfaces(classes):
    """Small in-allocation checks of the real loop, before either task arm."""
    arrays=[np.full((side,side,3),i*30,np.uint8) for i,side in enumerate([32,48,64,64,96,96,96])]
    data=c.RAMImages([dict(id=i,label=i) for i in range(7)],arrays)
    val=c.RAMImages(data.rows,[np.array(data.image(i).resize((224,224),Image.Resampling.BILINEAR),copy=True) for i in range(7)])
    checks={}
    for name,kind,epochs in [('baseline','interface_baseline',1),('student','interface_student',3)]:
        net=c.fresh_model(classes)
        checks[name]=train(net,data,val,classes,OUT/'interfaces'/name,kind,epochs=epochs,adaptation_epochs=1)
        h=json.loads((OUT/'interfaces'/name/'history.json').read_text())
        assert len(h)==epochs and all(r['images']==7 for r in h)
        if name=='student':
            assert h[-1]['input_policy']=='direct224 adaptation' and h[0]['input_policy']=='assigned size RRC+flip'
            assert h[-1]['learning_rate_applied']==task_learning_rate(net,'adapt')
        # These are disposable synthetic test weights. Keep the checks/history,
        # and avoid exporting random model payloads as project checkpoints.
        for filename in ['latest.pth','best.pth','best_top5.pth']:
            target=(OUT/'interfaces'/name/filename).resolve()
            assert target.is_relative_to((OUT/'interfaces').resolve())
            if target.exists():target.unlink()
        del net;gc.collect();torch.cuda.empty_cache()
    c.dump(OUT/'interface_checks.json',dict(passed=True,synthetic_only=True,checks=checks,
            task_weights_not_reused=True,extra_setup_cost_charged_to_both_arms=True))
    del data,val;gc.collect();torch.cuda.empty_cache();c.seed()


def predictor_setup(saved):
    """Verify frozen models and account for previously unseparated preprocessing."""
    tick=time.perf_counter()
    policy=json.loads((saved/'mapping_policy.json').read_text());lab=np.load(saved/'teacher_labels.npz')
    columns=[r['feature'] for r in read(saved/'optimized_v28_features.csv')]
    raw=np.load(saved/'initial20k_features.npz')['x']
    extra,extra_names=predictor.geometry_features(raw,columns,policy['sides'])
    raw=np.column_stack([raw,extra]);names=columns+extra_names
    checkpoints=[torch.load(saved/'mlp'/('deployed_seed'+str(seed)+'.pt'),map_location='cpu',weights_only=False) for seed in [42,43,44]]
    for ck in checkpoints:
        assert ck['parent_teacher'].endswith('/original_full_v1/job_5327/teacher/best.pth')
        assert ck['label_sha256']==c.sha(saved/'teacher_labels.npz')
        assert ck['epochs']==3 and ck['full_refit_rows']==17821
        assert ck['sides']==policy['sides'] and np.array_equal(ck['tiers'],c.TIERS)
        assert len(ck['preprocessing']['feature_columns'])==222
        assert ck['preprocessing']['input_columns']==names
    dev=np.array(policy['fit_indices']);dev=dev[lab['eligible'][dev]]
    # Repeat the selected preprocessing only; do not repeat research search or
    # invent a zero cost for preparing the selected predictor.
    predictor.preprocess(raw,lab['y'],dev,checkpoints[0]['config'],names)
    z,state=predictor.preprocess(raw,lab['y'],np.flatnonzero(lab['eligible']),checkpoints[0]['config'],names)
    assert state['feature_columns']==checkpoints[0]['preprocessing']['feature_columns']
    savedstate=checkpoints[0]['preprocessing']
    check=np.clip(savedstate['scaler'].transform(savedstate['imputer'].transform(raw)[:,savedstate['keep']]),-8,8).astype('float32')
    assert np.array_equal(z,check),'Frozen preprocessing did not reproduce'
    diagnostic=json.loads((saved/'FULL222_FIXED_RULE_DIAGNOSTIC.json').read_text())
    assert diagnostic['raw_exact_numeric_target_met'] and diagnostic['applied_unsafe_numeric_target_met']
    assert diagnostic['threshold']==SAFETY_THRESHOLD and diagnostic['resolved_images']==3580
    teacher=json.loads((saved/'teacher/results.json').read_text())
    selected=json.loads((saved/'mlp/results.json').read_text())
    replay=json.loads((saved/'teacher_profile_timing.json').read_text())
    assert replay['parity']['response_bits']==replay['parity']['eligible_rows']==replay['parity']['tier_labels']==0
    preprocessing=time.perf_counter()-tick
    components=dict(teacher_creation=teacher['complete_process_seconds'],
                    optimized_label_creation=replay['phase_plus_source_cache_seconds'],
                    initial_feature_creation=json.loads((saved/'initial20k_feature_timing.json').read_text())['wall_seconds_including_table_export'],
                    selected_MLP_evaluation_fit=selected['evaluation_fit_seconds'],
                    selected_MLP_deployment_refit=selected['deployment_refit_seconds'],
                    verified_preprocessing_and_predictor_setup=preprocessing)
    c.dump(OUT/'upstream_accounting.json',dict(measured_components_seconds=components,total_seconds=sum(components.values()),
        basis='sum of actual reused creation-stage measurements plus repeated preprocessing; not a fresh uninterrupted teacher-to-student run',
        teacher_original_node='gpu-P100-01',optimized_label_replay_node='gpu-P100-02',
        same_P100_GPU_architecture=True,upstream_not_same_allocation=True,
        original_label_creation_seconds=644.7349595725536,
        label_replay_soft_probability_difference=replay['parity'],
        frozen_original_teacher_auxiliary_arrays_and_MLP_weights_preserved=True,
        research_search_seconds_excluded_and_reported_separately=selected['total_search_process_seconds']))
    return checkpoints,policy,columns,diagnostic,sum(components.values())


def predict80(source,checkpoints,policy,columns,directory):
    directory.mkdir();tick=time.perf_counter()
    frame,_,ft=features.extract(source,columns,True,directory,'source80k')
    extra,extra_names=predictor.geometry_features(frame.values,columns,policy['sides'])
    raw=np.column_stack([frame.values,extra]);floor=c.floor_indices(source.shapes)
    ptick=time.perf_counter();probabilities=[]
    for ck in checkpoints:
        state=ck['preprocessing']
        x=np.clip(state['scaler'].transform(state['imputer'].transform(raw)[:,state['keep']]),-8,8).astype('float32')
        net=predictor.Net(x.shape[1],ck['config']).cuda();net.load_state_dict(ck['model'])
        probabilities.append(predictor.probabilities(net,torch.tensor(x,device='cuda'),torch.tensor(floor,device='cuda')))
        del net,x;gc.collect();torch.cuda.empty_cache()
    p=np.mean(probabilities,axis=0);rawpred=p.argmax(1)
    base=np.maximum(np.minimum(rawpred+1,10),floor)
    risk=(p*(np.arange(11)[None,:]>=base[:,None]+2)).sum(1)
    applied=np.where(risk>SAFETY_THRESHOLD,np.minimum(base+1,10),base)
    np.savez_compressed(directory/'predictions80k.npz',p=p,raw=rawpred,floor=floor,applied=applied)
    info=dict(images=len(source),feature_time=ft,inference_and_export_seconds=time.perf_counter()-ptick,
        complete_prediction_seconds=time.perf_counter()-tick,selected_feature_count=222,
        source_MobileNet_passes_per_image=1,ResNet_feature_passes_on80k=0,
        feature_loader_batch=128,feature_loader_workers=8,
        raw_counts=np.bincount(rawpred,minlength=11).tolist(),applied_counts=np.bincount(applied,minlength=11).tolist(),
        extra_safety_promotions=int((applied!=base).sum()),no80k_teacher_ground_truth=True)
    c.dump(directory/'prediction_results.json',info);return applied,info


def baseline(tables,classes,monitor):
    directory=OUT/'baseline';directory.mkdir();reference.OUT=directory
    monitor.set_phase('baseline100k_cache');torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    status('BASELINE100K_CACHE');source,cache=reference.stream_cache(tables['sources.csv'])
    val=fixed(tables['validation.csv'],directory,'validation');c.seed();net=c.fresh_model(classes)
    monitor.set_phase('baseline100k_training')
    result=train(net,source,val,classes,directory/'training','baseline')
    tail=time.perf_counter();test=fixed(tables['test.csv'],directory,'test')
    metric=c.evaluate(net,test,directory/'test_predictions.npz');monitor.sample()
    value=dict(training=result,test=metric,source_cache=cache,
        complete_arm_seconds=time.perf_counter()-start,test_setup_and_evaluation_seconds=time.perf_counter()-tail,
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    c.dump(directory/'results.json',value)
    del net,source,val,test;gc.collect();torch.cuda.empty_cache();return value


def proposed(tables,classes,saved,checkpoints,policy,diagnostic,upstream,common,base,monitor):
    directory=OUT/'proposed';directory.mkdir();start=time.perf_counter()
    baseline_seconds=base['complete_arm_seconds']+common;ceiling=.82*baseline_seconds
    fixed_seconds=upstream+common
    guard=dict(fixed_seconds=fixed_seconds,arm_start=start,ceiling=ceiling,
        adapt_epoch_seconds=base['training']['training_loop_seconds']/50*.8,
        validation_seconds=5.,tail_reserve_seconds=max(40.,base['test_setup_and_evaluation_seconds']+10.))
    monitor.set_phase('proposed_source80k_cache');torch.cuda.reset_peak_memory_stats()
    status('PROPOSED80K_CACHE',time_ceiling=ceiling)
    remaining=[r for r in tables['sources.csv'] if r['part']=='remaining'];assert len(remaining)==80000
    source,cache=encoded.cache(remaining,directory,'remaining80k_source',workers=16)
    mapping.install_floor(policy['sides'])
    monitor.set_phase('proposed_features_and_inference');status('PROPOSED80K_FEATURES')
    applied,prediction=predict80(source,checkpoints,policy,columns=[r['feature'] for r in read(saved/'optimized_v28_features.csv')],directory=directory/'prediction')
    tick=time.perf_counter();monitor.set_phase('proposed_mapping');status('PROPOSED80K_MAPPING')
    data=map_selected_images(source,applied,policy['sides'],c.floor_indices(source.shapes),workers=4,ram_factory=c.RAMImages)
    mapping_seconds=time.perf_counter()-tick
    storage=dict(source_encoded_image_payload_bytes=cache['source_payload_bytes'],
        source_decoded_RGB_bytes=sum(h*w*3 for h,w in source.shapes),mapped_decoded_RGB_bytes=data.bytes,
        mapped_geometry=c.geometry(data),mapping_seconds=mapping_seconds,
        persisted_compressed_image_payload_bytes=0,
        note='Mapped images remain decodedRAM; no physical JPEG/PNG storage saving is claimed')
    with (directory/'applied80k_manifest.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','source_h','source_w','applied_tier','retained_area_percent','mapped_h','mapped_w'])
        for i,r in enumerate(source.rows):writer.writerow([r['id'],*source.shapes[i],int(applied[i]),float(c.TIERS[applied[i]]),*data.shapes[i]])
    c.dump(directory/'storage.json',storage);del source;gc.collect();torch.cuda.empty_cache()
    val=fixed(tables['validation.csv'],directory,'validation')
    ck=torch.load(saved/'teacher/best.pth',map_location='cpu',weights_only=True)
    assert ck['classes']==classes
    net=c.fresh_model(classes);net.load_state_dict(ck['model_state_dict']);del ck
    monitor.set_phase('proposed80k_training')
    training=train(net,data,val,classes,directory/'training','proposed',guard)
    monitor.set_phase('proposed_test');test=fixed(tables['test.csv'],directory,'test')
    metric=c.evaluate(net,test,directory/'test_predictions.npz');monitor.sample()
    actual=time.perf_counter()-start;accounted=fixed_seconds+actual
    result=dict(training=training,test=metric,new80k_arm_seconds=actual,accounted_complete_pipeline_seconds=accounted,
        upstream_measured_seconds=upstream,common_setup_seconds_charged=common,
        timing_basis='measured reused upstream stages plus measured new80k arm plus common setup; not fresh uninterrupted pipeline',
        MLP=diagnostic,prediction=prediction,storage=storage,source_cache=cache,
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    c.dump(directory/'results.json',result);return result


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.multiprocessing.set_start_method('fork',force=True);torch.set_num_threads(4);c.seed()
    assert torch.cuda.is_available() and 'P100' in torch.cuda.get_device_name(0)
    rp.install(c,'teacher');saved,assets=unpack()
    tables={n:read(saved/n) for n in ['sources.csv','validation.csv','test.csv']}
    classes=json.loads((saved/'protocol.json').read_text())['classes']
    assert [len(tables[n]) for n in ['sources.csv','validation.csv','test.csv']]==[100000,1000,5000]
    sets=[{r.get('source_sha256',r.get('sha256')) for r in tables[n]} for n in ['sources.csv','validation.csv','test.csv']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    warmup_started=time.perf_counter();warmup_bytes=0
    def verify_and_warm(row):
        raw=Path(row['path']).read_bytes()
        expected=row.get('source_sha256',row.get('sha256'))
        assert hashlib.sha256(raw).hexdigest()==expected,'Changed input: '+row['path']
        return len(raw)
    for records in tables.values():
        assert all(classes[int(r['label'])]==r['class_name'] for r in records)
        with ThreadPoolExecutor(max_workers=16) as pool:warmup_bytes+=sum(pool.map(verify_and_warm,records))
    warmup_seconds=time.perf_counter()-warmup_started
    c.dump(OUT/'input_checks.json',dict(all106000_content_hashes_verified=True,source_disjoint_splits=True,
        shared_input_read_warmup_bytes=warmup_bytes,shared_input_read_warmup_seconds=warmup_seconds,
        common_cost_charged_to_each_arm=True,order='baseline then proposed; no privileged OS cache flushing',
        OS_page_cache_retention_during_long_run_not_guaranteed=True))
    assert c.sha(saved/'teacher/best.pth')=='b731a056cde30415fa55a9c279ef5733cdf7ffa40747dc35ac41e24bbd20e378'
    assert (Path(os.environ['TORCH_HOME'])/'hub/checkpoints/mobilenet_v3_small-047dcff4.pth').is_file()
    # Warm only the shared model interface, then reset before each actual arm.
    probe=c.fresh_model(classes).eval()
    with torch.inference_mode():assert probe(torch.zeros(2,3,224,224,device='cuda')).shape==(2,100)
    del probe;gc.collect();torch.cuda.empty_cache()
    synthetic_interfaces(classes)
    common=time.perf_counter()-PROCESS_START
    checkpoints,policy,columns,diagnostic,upstream=predictor_setup(saved)
    c.dump(OUT/'protocol.json',dict(classes=classes,input_assets=assets,GPU=torch.cuda.get_device_name(0),node=os.environ.get('SLURMD_NODENAME'),
        both_current_arms_same_allocation=True,batch=64,training_workers=4,evaluation_workers=4,precision='FP32',standard_BatchNorm=True,
        baseline_cache_workers=dict(header=16,decode=8),proposed_cache_workers=16,
        proposed_feature_loader=dict(batch=128,workers=8),mapping_threads=4,
        feature_loader_is_not_ResNet_training_loader=True,CPU_and_RAM_requests_equal_between_arms=True,
        baseline='100k original,50epochs,scratch,SGD.1 cosine50,RRC224+flip,CE',
        proposed='saved50epoch20k teacher; frozen222featureMLP; +1 plus selective extra safety promotion;48assigned-sizeRRC+flip+LS.05;2direct224adaptationSGD.001',
        no_CutMix=True,no_manual_BN_calibration=True,source_encoder='one frozenMobileNetV3Small pass per image',
        tiers=c.TIERS.tolist(),safety_threshold=SAFETY_THRESHOLD,
        ResNet_presentations_each_method=5000000,teacher_retrained=False,MLP_retrained=False,
        common_setup_seconds_charged_to_each_arm=common,
        input_content_checks_and_warmup_seconds=warmup_seconds,
        reusable_upstream_not_same_allocation=True,upstream_all_GPU_models_P100=True,
        evaluation='same frozen original1000 validation and5000test;directBILINEAR224;bestvalidationtop1checkpoint',
        no_test_based_tuning=True,validation_epochs=sorted(VALIDATE),
        acceptance='time<.82*matchedbaseline;testtop1drop<=5points;MLPexact>=50;appliedunsafe<5'))
    monitor=Monitor(OUT);monitor.thread.start()
    try:
        base=baseline(tables,classes,monitor)
        base['accounted_complete_seconds']=base['complete_arm_seconds']+common
        c.dump(OUT/'baseline/results.json',base);status('BASELINE_COMPLETE',test=base['test'],seconds=base['accounted_complete_seconds'])
        prop=proposed(tables,classes,saved,checkpoints,policy,diagnostic,upstream,common,base,monitor)
        saving=100*(1-prop['accounted_complete_pipeline_seconds']/base['accounted_complete_seconds'])
        loss=base['test']['top1_percent']-prop['test']['top1_percent']
        accepted=(saving>18 and loss<=5 and diagnostic['raw_exact_percent']>=50 and diagnostic['applied_unsafe_percent']<5)
        result=dict(baseline=base,proposed=prop,time_saved_percent=saving,top1_drop_points=loss,accepted=accepted,
                    full_objective_met_numerically=accepted,diagnostic_holdout_and_classification_test_previously_reused=True,
                    allocation_program_seconds=time.perf_counter()-PROCESS_START,
                    memory=json.loads((OUT/'memory_progress.json').read_text()))
        c.dump(OUT/'results.json',result);status('COMPLETE',accepted=accepted,time_saved_percent=saving,top1_drop_points=loss)
    finally:
        monitor.stop.set();monitor.thread.join(timeout=10)


if __name__=='__main__':
    raise SystemExit('Imported helpers only; run cross_backbone_paired_study.py instead')



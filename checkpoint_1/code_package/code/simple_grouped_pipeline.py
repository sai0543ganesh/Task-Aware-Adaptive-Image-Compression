"""Fresh randomized-source experiment; one allocation, explicit RAM pixels, native ResNets."""
import argparse,csv,gc,json,math,os,resource,shutil,subprocess,sys,tarfile,time,traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
import simple_grouped_core as c
import optimized_paired_mtp as v

ROOT=Path('/userhome/mtech/s.chatharasupalli/imagenet100_mtp')
INPUT=ROOT/'results/optimized_paired_v1/job_5269'

def read(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def status(out,stage,**kw):c.dump(Path(out)/'status.json',dict(stage=stage,unix_time=time.time(),**kw));c.say(stage,**kw)

def memory():
    return dict(main_process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                RSS_values_are_not_job_aggregate=True)

def gpu_interfaces():
    # One bounded interface check in the real allocation, not an extra training job.
    columns=pd.read_csv(Path(__file__).with_name('optimized_v28_features.csv'))['feature'].tolist()
    mobile=models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).cuda().eval()
    with torch.inference_mode():
        value=v.enrich_features(mobile,torch.zeros(2,3,224,224,device='cuda'),np.zeros((2,215),np.float32),torch.tensor([96,128]),columns)
        assert value.shape==(2,215) and np.isfinite(value).all()
        net=c.fresh_model(list(range(100))).eval()
        assert net(torch.zeros(2,3,96,128,device='cuda')).shape==(2,100)
    del mobile,net;gc.collect();torch.cuda.empty_cache();c.seed()
    # Exercise the real training loop once on seven synthetic images, including
    # the tiny-singleton boundary. Dispose these weights; no task data are used.
    arrays=[np.full((h,w,3),i*20,np.uint8) for i,(h,w) in enumerate([(11,11),(64,80),(64,80),(96,128),(96,128),(129,160),(213,320)])]
    data=c.RAMImages([dict(label=i) for i in range(7)],arrays)
    checknet=c.fresh_model(list(range(100)))
    checkpath=Path(os.environ['TMPDIR'])/'simple_loop_check'
    verified=c.train(checknet,data,data,list(range(100)),1,.1,checkpath,'simple_loop_check')
    h=json.loads((checkpath/'history.json').read_text())[0]
    assert h['images']==7 and h['extra_duplicate_forward_images']==1
    assert h['optimizer_steps']==len(c.Buckets(data,True)) and h['gradient_accumulation'] is False
    assert all(m.momentum==.1 for m in checknet.modules() if isinstance(m,torch.nn.BatchNorm2d))
    del checknet,data;gc.collect();torch.cuda.empty_cache();c.seed();c.say('GPU_AND_STANDARD_TRAINING_LOOP_CHECKS_PASSED')

def audit(out):
    before=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    old=json.loads((INPUT/'protocol.json').read_text());classes=old['classes']
    for name in ['randomized_sources.csv','validation.csv','test.csv']:
        shutil.copyfile(INPUT/name,out/name)
    source=read(out/'randomized_sources.csv');val=read(out/'validation.csv');test=read(out/'test.csv')
    assert len(classes)==100 and len(set(classes))==100
    for rows,n in [(source,100000),(val,1000),(test,5000)]:
        assert len(rows)==n and len({r.get('id',r['path']) for r in rows})==n
        for r in rows:
            assert classes[int(r['label'])]==r['class_name']
        with ThreadPoolExecutor(max_workers=16) as pool:
            present=list(pool.map(lambda r:Path(r['path']).is_file(),rows))
        if not all(present):raise RuntimeError('Missing input image: '+rows[present.index(False)]['path'])
    for part,n in [('teacher',200),('remaining',800)]:
        counts=Counter(r['class_name'] for r in source if r['part']==part)
        assert len(counts)==100 and set(counts.values())=={n},(part,counts)
    groups=[{r['source_sha256'] for r in source},{r['sha256'] for r in val},{r['sha256'] for r in test}]
    assert all(not (groups[i]&groups[j]) for i in range(3) for j in range(i)),'Source leakage'
    assert Counter(r['class_name'] for r in val)==Counter({k:10 for k in classes})
    assert Counter(r['class_name'] for r in test)==Counter({k:50 for k in classes})
    columns=pd.read_csv(Path(__file__).with_name('optimized_v28_features.csv'))['feature'].tolist()
    assert len(columns)==215 and len(set(columns))==215
    assert not any('class_id' in s for s in columns)
    assert (Path.home()/'.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth').is_file()
    protocol=dict(classes=classes,tiers=c.TIERS.tolist(),source_images=100000,teacher_images=20000,
        mlp_source_images=20000,remaining_images=80000,validation_images=1000,test_images=5000,
        source_sha256_lineage_disjoint_from_eval=True,
        manifest_sha256={n:c.sha(out/n) for n in ['randomized_sources.csv','validation.csv','test.csv']},
        epochs=dict(baseline=50,teacher=50,plus1=50,plus2=50),
        presentations_per_comparison=dict(baseline=5000000,proposed=5000000),
        architecture='torchvision ResNet18, scratch task weights; same architecture in every phase',
        input='Unchanged stored RGB pixels, normalized; right/bottom zero padding to next16 per dimension. No training/evaluation crop/resize/flip.',
        batching='16pixel same-canvas buckets, maximum64images,2M pixel target; shuffled batches; one meanCE SGD update per actual batch; no accumulation',
        batchnorm='Unchanged standard BatchNorm momentum0.1, no GroupNorm or custom running-stat updates',
        singleton_boundary='Only when N*finalH*finalW==1 duplicate image within forward; retain one logical sample and report extra forward work',
        optimizer='Established SGD lr0.1 momentum0.9 weight_decay0.0001 cosine50; continuation lr0.01; no additional warmup',
        feature_encoder='Frozen ImageNet MobileNetV3-Small; one source view,224 transform only in feature branch; no ResNet feature calls on80k',
        feature_candidates=215,class_id_input=False,task_checkpoints_reused=False,old_predictions_reused=False,
        geometry_guard='Label and deploy minimum short side min(source short side,96); never upscale source. Changed target constraint from historical v28, so new MLP scores are a separate benchmark.',
        oracle='Native BOX-resized pixels. Minimum all-correct suffix among geometry-allowed11 tiers. Source-incorrect rows unresolved/excluded from MLP supervised fitting.',
        codec='BOX spatial reduction followed by losslessPNG;100% unchanged source; no JPEG quantization mismatch',
        evaluation='Frozen independent validation for model selection, frozen independent test once per completed arm',
        teacher_quality_gate=dict(source_correct_minimum_percent=90,independent_validation_minimum_percent=40),
        randomized_preparation_reused_not_retimed=True,
        randomized_initial_counts20k=old['randomized_initial_retention_counts_20k'],
        randomized_initial_counts80k=old['randomized_initial_retention_counts_80k'],
        accepted_max_top1_drop_points=3.,required_complete_time_faster=True,
        timing='Fresh input RAM loading charged separately in each process. Proposed common upstream costs shared between independently initialized+1/+2 arms.',
        inventory_audit_seconds=time.perf_counter()-before,job=os.environ.get('SLURM_JOB_ID'),
        node=os.environ.get('SLURMD_NODENAME'),code_sha256={p.name:c.sha(p) for p in Path(__file__).parent.glob('*.py')})
    c.dump(out/'protocol.json',protocol);return protocol

def load_inputs(run,out):
    rows=read(run/'randomized_sources.csv');source,sourceinfo=c.cache(rows,out,'source100k')
    val,_=c.cache(read(run/'validation.csv'),out,'validation1000')
    test,_=c.cache(read(run/'test.csv'),out,'test5000')
    c.dump(out/'source_geometry.json',c.geometry(source))
    return source,val,test,sourceinfo

class Features(Dataset):
    def __init__(self,data,columns,encoder):self.data=data;self.columns=columns;self.encoder=encoder;self.transform=v.eval_transform()
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        before=time.perf_counter();im=self.data.image(i);values,bounds=v.source_numeric(im)
        raw=np.zeros(len(self.columns),np.float32)
        for j,key in enumerate(self.columns):
            if not key.startswith(('cheap_v3_mobile_','extra_cam_')):
                if key not in values:raise KeyError('Missing feature implementation: '+key)
                raw[j]=values[key]
        image=self.transform(im) if self.encoder else torch.empty(0)
        return image,raw,bounds,min(im.size),i,time.perf_counter()-before

def extract(data,columns,encoder,out,name):
    start=time.perf_counter();mobile=models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).cuda().eval() if encoder else None
    stream=DataLoader(Features(data,columns,encoder),batch_size=128,num_workers=8,pin_memory=True,persistent_workers=True)
    x=np.empty((len(data),len(columns)),np.float32);bounds=np.empty((len(data),10),np.float32);cpu=enc=0.
    with torch.inference_mode():
        for images,raw,b,short,index,seconds in stream:
            a=raw.numpy()
            if encoder:
                c.sync();t=time.perf_counter();a=v.enrich_features(mobile,images.cuda(non_blocking=True),a,short,columns);c.sync();enc+=time.perf_counter()-t
            ix=index.numpy();x[ix]=a;bounds[ix]=b.numpy();cpu+=float(seconds.sum())
            if int(ix[-1])%12800==12799:c.say('FEATURE_PROGRESS',split=name,images=int(ix[-1])+1)
    del stream,mobile;gc.collect();torch.cuda.empty_cache()
    timing=dict(images=len(data),wall_seconds=time.perf_counter()-start,worker_CPU_seconds_sum=cpu,
        encoder_GPU_wall_seconds=enc,source_encoder_passes_per_image=int(encoder),remaining80k_ResNet_feature_passes=0,
        raw_feature_candidates=len(columns),CPU_worker_sum_is_not_wall_seconds=True)
    np.savez_compressed(out/(name+'_features.npz'),x=x,bounds=bounds)
    timing['wall_seconds_including_table_export']=time.perf_counter()-start;c.dump(out/(name+'_feature_timing.json'),timing)
    return pd.DataFrame(x,columns=columns),bounds,timing

def label_teacher(net,data,out,source_evaluation=None):
    start=time.perf_counter();allp=np.empty((len(data),11,100),np.float32);ix=np.arange(len(data))
    for tier in reversed(range(11)):
        current=data if tier==10 else c.reduced(data,ix,np.full(len(data),tier))
        metric,p=source_evaluation if tier==10 and source_evaluation is not None else c.evaluate(net,current,probabilities=True);allp[:,tier]=p
        c.say('TEACHER_TIER_COMPLETE',tier=tier,pixel_percent=c.TIERS[tier],**metric)
        if tier!=10:del current;gc.collect()
    truth=np.array([int(r['label']) for r in data.rows]);response=allp.argmax(2)==truth[:,None]
    floor=c.floor_indices(data.shapes);suffix=np.logical_and.accumulate(response[:,::-1],axis=1)[:,::-1]
    eligible=response[:,-1];allowed=suffix&(np.arange(11)[None,:]>=floor[:,None])
    y=np.where(eligible,allowed.argmax(1),10).astype(np.int64);soft=allp[:,-1]
    proxy=np.empty((len(data),36),np.float32)
    for j,tier in enumerate([0,2,3,4,7,8]):
        conf=allp[:,tier].max(1);good=response[:,tier].astype(np.float32)
        agree=(allp[:,tier].argmax(1)==allp[:,-1].argmax(1)).astype(np.float32)
        proxy[:,6*j:6*j+6]=np.stack([conf,good,agree,conf*good,conf*agree,np.full(len(data),math.log1p(c.TIERS[tier])/math.log1p(100))],1)
    np.savez_compressed(out/'teacher_labels.npz',y=y,response=response,soft=soft,proxy=proxy,floor=floor,eligible=eligible,
        identities=np.array([r['id'] for r in data.rows]))
    info=dict(seconds=time.perf_counter()-start,rows=len(data),resolved=int(eligible.sum()),unresolved=int((~eligible).sum()),
        source_correct_percent=100*float(eligible.mean()),label_counts=np.bincount(y[eligible],minlength=11).tolist(),
        unresolved_assigned_source_only_for_storage_not_used_for_training=True,teacher_views_per_initial_image=11)
    c.dump(out/'label_timing.json',info);return y,response,soft,proxy,floor,eligible,info

def fit_predictor(frame,bounds,y,response,soft,proxy,floor,eligible,groups,out):
    start=time.perf_counter();allix=np.arange(len(y));dev,hold=v.split_indices(allix,y,4000,20261022,groups)
    dev=dev[eligible[dev]];evaluated_hold=hold[eligible[hold]]
    if len(dev)<1000 or len(evaluated_hold)<100:raise RuntimeError('Too few teacher-resolved examples for valid MLP fit/holdout')
    fitting,cal=v.split_indices(dev,y,max(100,len(dev)//5),20261023,groups)
    numeric=[key for key in frame.columns if not key.startswith(('cheap_v3_mobile_','extra_cam_'))];records=[]
    for family,columns in [('numeric',numeric),('encoder',list(frame.columns))]:
        for near in [.5,1.5]:
            cfg=dict(name='fresh_v28_native',near=near,aux=.2,residual=1.)
            net,state,epochs,p=v.fit_mlp(frame,y,bounds,soft,proxy,fitting,cal,columns,cfg)
            m=v.tier_metrics(y[cal],p.argmax(1));record=dict(family=family,config=cfg,epochs=epochs,metrics=m)
            records.append(record);c.say('MLP_CANDIDATE',**record);del net;torch.cuda.empty_cache()
    chosen=max(records,key=lambda r:r['metrics']['exact_percent']+.35*r['metrics']['within_one_percent'])
    cheap=[r for r in records if r['family']=='numeric' and r['metrics']['exact_percent']>=chosen['metrics']['exact_percent']-1 and r['metrics']['within_one_percent']>=chosen['metrics']['within_one_percent']-2]
    if cheap:chosen=max(cheap,key=lambda r:r['metrics']['exact_percent']+.35*r['metrics']['within_one_percent'])
    columns=numeric if chosen['family']=='numeric' else list(frame.columns)
    net,state,_,p=v.fit_mlp(frame,y,bounds,soft,proxy,dev,evaluated_hold,columns,chosen['config'],chosen['epochs'])
    raw=p.argmax(1);constrained=np.maximum(raw,floor[evaluated_hold]);metric=dict(argmax=v.tier_metrics(y[evaluated_hold],raw),
        geometry_constrained=v.tier_metrics(y[evaluated_hold],constrained))
    for promotion in [1,2]:
        applied=np.minimum(constrained+promotion,10);m=v.tier_metrics(y[evaluated_hold],applied)
        m['teacher_correct_at_applied_percent']=100*float(response[evaluated_hold,applied].mean());metric['plus'+str(promotion)]=m
    np.savez_compressed(out/'mlp_holdout.npz',all_holdout=hold,evaluated=evaluated_hold,y=y[evaluated_hold],p=p,raw=raw,constrained=constrained)
    torch.save(dict(model=net.state_dict(),preprocessing=state,selected=chosen),out/'mlp_evaluation.pt');del net
    supervised=allix[eligible]
    net,state,_,_=v.fit_mlp(frame,y,bounds,soft,proxy,supervised,supervised[:1],columns,chosen['config'],chosen['epochs'])
    torch.save(dict(model=net.state_dict(),preprocessing=state,selected=chosen,tiers=c.TIERS.tolist(),class_id_input=False),out/'mlp_deployed.pt')
    pd.DataFrame({'feature':state['feature_columns']}).to_csv(out/'selected_features.csv',index=False)
    result=dict(selected=chosen,candidates=records,metrics=metric,training_rows=len(dev),holdout_total=len(hold),
        holdout_resolved=len(evaluated_hold),holdout_unresolved=len(hold)-len(evaluated_hold),
        holdout_coverage_percent=100*len(evaluated_hold)/len(hold),full_refit_rows=len(supervised),
        candidate_features=len(frame.columns),selected_features=len(state['feature_columns']),class_id_input=False,
        holdout_used_for_selection=False,source_groups_disjoint=True,seconds=time.perf_counter()-start)
    c.dump(out/'mlp_results.json',result);return net,state,chosen,result

def encode_task(task):
    data,i,tier,destination=task;im=data.image(i);q=float(c.TIERS[tier]);w,h=im.size
    destination=Path(destination)
    if q<100:
        im=im.resize((max(1,round(w*math.sqrt(q/100))),max(1,round(h*math.sqrt(q/100)))),Image.Resampling.BOX)
        im.save(destination,format='PNG',compress_level=1);size=destination.stat().st_size;physical=size
    else:
        destination.symlink_to(data.rows[i]['path']);size=int(data.rows[i]['encoded_bytes']);physical=0
    a=np.array(im,dtype=np.uint8,copy=True);row=dict(data.rows[i]);row.update(path=str(destination),encoded_bytes=size,tier=tier)
    return a,row,size,physical

def materialize(data,tiers,out,name):
    start=time.perf_counter();directory=Path(os.environ['TMPDIR'])/(name+'_images');directory.mkdir(exist_ok=False)
    arrays=[];rows=[];payload=physical=0
    tasks=((data,i,int(tier),directory/(str(i)+('.png' if tier<10 else '.source'))) for i,tier in enumerate(tiers))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for a,row,size,n in pool.map(encode_task,tasks):arrays.append(a);rows.append(row);payload+=size;physical+=n
    encoded_seconds=time.perf_counter()-start;packed=c.RAMImages(rows,arrays)
    export=time.perf_counter()
    with tarfile.open(out/(name+'_images.tar'),'w',dereference=False) as tar:tar.add(directory,arcname=name)
    info=dict(seconds=time.perf_counter()-start,encoding_and_RAM_pack_seconds=export-start,
        encoding_seconds=encoded_seconds,archive_export_seconds=time.perf_counter()-export,
        effective_image_payload_bytes=payload,new_encoded_payload_bytes=physical,linked_source_payload_included=True,
        archive_bytes=(out/(name+'_images.tar')).stat().st_size,
        tier_counts=np.bincount(tiers,minlength=11).tolist(),geometry=c.geometry(packed),
        local_encoded_directory=str(directory),training_reads_explicit_RAM=True,decoded_RAM_bytes=packed.bytes)
    # Archive member paths are stable; source100% members are symlinks to permanent source files.
    with open(out/(name+'_image_manifest.csv'),'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','label','tier','archive_member','effective_bytes','source_path']);writer.writeheader()
        for i,row in enumerate(rows):writer.writerow(dict(id=row['id'],label=row['label'],tier=row['tier'],archive_member=name+'/'+Path(row['path']).name,effective_bytes=row['encoded_bytes'],source_path=data.rows[i]['path']))
    info['seconds']=time.perf_counter()-start;c.dump(out/(name+'_materialization.json'),info)
    return packed,info

def baseline(run,protocol):
    start=json.loads((run/'baseline_clock.json').read_text())['start_monotonic'];out=run/'baseline';out.mkdir();status(run,'BASELINE_RAM_LOAD')
    source,val,test,info=load_inputs(run,out);net=c.fresh_model(protocol['classes'])
    status(run,'BASELINE_TRAINING');training=c.train(net,source,val,protocol['classes'],50,.1,out/'training','baseline')
    evaluation=c.evaluate(net,test,out/'independent_test_predictions.npz')
    result=dict(training=training,test=evaluation,source_cache=info,memory=memory(),complete_process_seconds=time.perf_counter()-start+protocol['inventory_audit_seconds'],
        time_basis='Externally started process clock plus shared inventory/audit; includes interpreter startup, RAM loading, training, checkpoint exports and final test')
    c.dump(out/'results.json',result);status(run,'BASELINE_COMPLETE',**result)

def proposed(run,protocol):
    start=json.loads((run/'proposed_clock.json').read_text())['start_monotonic']-protocol['inventory_audit_seconds'];out=run/'proposed';out.mkdir();status(run,'PROPOSED_RAM_LOAD')
    source,val,test,info=load_inputs(run,out)
    baseline_path=run/'baseline/results.json'
    base=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    budget=base['complete_process_seconds'] if base else None
    if base and info['content_fingerprint']!=base['source_cache']['content_fingerprint']:raise RuntimeError('Baseline/proposed sources changed')
    teacher=c.SubsetRAM(source,[i for i,r in enumerate(source.rows) if r['part']=='teacher'])
    remaining=c.SubsetRAM(source,[i for i,r in enumerate(source.rows) if r['part']=='remaining'])
    c.dump(out/'source_inventory.json',dict(teacher_geometry=c.geometry(teacher),remaining_geometry=c.geometry(remaining),
        teacher_payload_bytes=sum(int(r['encoded_bytes']) for r in teacher.rows),remaining_payload_bytes=sum(int(r['encoded_bytes']) for r in remaining.rows)))
    net=c.fresh_model(protocol['classes']);status(run,'TEACHER_TRAINING')
    training=c.train(net,teacher,val,protocol['classes'],50,.1,out/'teacher','teacher',budget,time.perf_counter()-start)
    c.dump(out/'teacher_training_results.json',training)
    # Store fresh teacher once, reload identical weights for each independent branch.
    source_evaluation=c.evaluate(net,teacher,probabilities=True)
    c.dump(out/'teacher_quality_gate.json',dict(source=source_evaluation[0],validation=training['best_validation_percent'],
        required_source_percent=90,required_validation_percent=40,training=training))
    if source_evaluation[0]['top1_percent']<90 or training['best_validation_percent']<40:
        raise RuntimeError('Teacher quality gate failed: source correctness<90% or independent validation<40%. No80k training launched.')
    status(run,'NATIVE_TEACHER_LABELS');y,response,soft,proxy,floor,eligible,labels=label_teacher(net,teacher,out,source_evaluation)
    del net;gc.collect();torch.cuda.empty_cache()
    columns=pd.read_csv(Path(__file__).with_name('optimized_v28_features.csv'))['feature'].tolist()
    status(run,'FRESH_INITIAL_FEATURES');frame,bounds,ft=extract(teacher,columns,True,out,'initial20k')
    status(run,'FRESH_MLP_TRAINING');groups=np.array([r['source_sha256'] for r in teacher.rows])
    mlp,state,chosen,metrics=fit_predictor(frame,bounds,y,response,soft,proxy,floor,eligible,groups,out)
    status(run,'FRESH_80K_FEATURES');deploycolumns=state['input_columns'];use_encoder=any(k.startswith(('cheap_v3_mobile_','extra_cam_')) for k in deploycolumns)
    deployment,dbounds,dt=extract(remaining,deploycolumns,use_encoder,out,'remaining80k')
    t=time.perf_counter();transformed=v.helper.transform(deployment,np.arange(len(deployment)),state)
    p=v.mlp_probabilities(mlp,transformed,dbounds);raw=p.argmax(1);constrained=np.maximum(raw,c.floor_indices(remaining.shapes));c.sync()
    inference=time.perf_counter()-t;np.savez_compressed(out/'fresh80k_predictions.npz',p=p,raw=raw,constrained=constrained)
    c.dump(out/'deployment.json',dict(inference_seconds=inference,feature_timing=dt,
        raw_counts=np.bincount(raw,minlength=11).tolist(),constrained_counts=np.bincount(constrained,minlength=11).tolist(),
        geometry_guard_changed=int((raw!=constrained).sum()),teacher_passes_on80k=0))
    del mlp,frame,deployment;gc.collect();torch.cuda.empty_cache()
    shared=time.perf_counter()-start;results={};c.dump(out/'shared_upstream.json',dict(seconds=shared,teacher=training,labels=labels,MLP=metrics,initial_features=ft,remaining_features=dt,inference_seconds=inference,source_cache=info))
    for promotion in [1,2]:
        name='plus'+str(promotion);branch=time.perf_counter();status(run,name.upper()+'_MATERIALIZE')
        try:
            if budget is not None and shared>=budget:raise c.GuardRejected('Shared fresh upstream already exceeds complete baseline time')
            applied=np.minimum(constrained+promotion,10);current,storage=materialize(remaining,applied,out,name)
            np.save(out/(name+'_applied.npy'),applied)
            net=c.fresh_model(protocol['classes']);ck=torch.load(out/'teacher/best.pth',map_location='cpu',weights_only=True)
            assert ck['classes']==protocol['classes'];net.load_state_dict(ck['model_state_dict']);del ck
            status(run,name.upper()+'_TRAINING')
            trained=c.train(net,current,val,protocol['classes'],50,.01,out/name,name,budget,shared+time.perf_counter()-branch)
            testmetric=c.evaluate(net,test,out/(name+'_independent_test_predictions.npz'))
            elapsed=shared+time.perf_counter()-branch
            results[name]=dict(state='complete',training=trained,test=testmetric,storage=storage,
                shared_upstream_seconds=shared,branch_seconds=time.perf_counter()-branch,complete_accounted_seconds=elapsed,
                time_basis='Measured shared fresh upstream plus this branch; plus2 excludes plus1 work but includes all shared costs',
                accuracy_drop_points=base['test']['top1_percent']-testmetric['top1_percent'] if base else None,
                speedup=budget/elapsed if base else None,
                accepted=(elapsed<budget and base['test']['top1_percent']-testmetric['top1_percent']<=3) if base else None,
                comparison_pending=base is None)
            status(run,name.upper()+'_COMPLETE',**results[name])
        except c.GuardRejected as exc:
            results[name]=dict(state='time_guard_rejected',reason=str(exc),shared_upstream_seconds=shared,
                branch_seconds=time.perf_counter()-branch,complete_accounted_seconds=shared+time.perf_counter()-branch)
            status(run,name.upper()+'_TIME_GUARD_REJECTED',**results[name])
        finally:
            if 'current' in locals():del current
            if 'net' in locals():del net
            gc.collect();torch.cuda.empty_cache()
        c.dump(out/'results.json',dict(arms=results,shared_upstream_seconds=shared,memory=memory(),actual_process_elapsed_seconds=time.perf_counter()-start))
    c.dump(out/'results.json',dict(arms=results,shared_upstream_seconds=shared,memory=memory(),actual_process_elapsed_seconds=time.perf_counter()-start))

def finalize_comparison(run):
    base=json.loads((run/'baseline/results.json').read_text())
    shared=json.loads((run/'proposed/shared_upstream.json').read_text())
    result=json.loads((run/'proposed/results.json').read_text())
    if base['source_cache']['content_fingerprint']!=shared['source_cache']['content_fingerprint']:
        raise RuntimeError('Baseline/proposed source content changed between phases')
    for arm in result['arms'].values():
        if arm['state']=='complete':
            arm['accuracy_drop_points']=base['test']['top1_percent']-arm['test']['top1_percent']
            arm['speedup']=base['complete_process_seconds']/arm['complete_accounted_seconds']
            arm['accepted']=arm['complete_accounted_seconds']<base['complete_process_seconds'] and arm['accuracy_drop_points']<=3
            arm['comparison_pending']=False
    c.dump(run/'proposed/results.json',result)
    c.dump(run/'final_comparison.json',dict(baseline=base,proposed=result,
        time_guard_note='Proposed ran first by user request; no completed baseline budget was available during its training. Actual complete times compared after baseline.'))

def report(run):
    lines=['# Fresh native randomized experiment','', 'All completed task ResNets use native RGB pixels plus minimal16pixel padding. Baseline5M and each proposed path5M training presentations. Independent frozen5000 test images; no old task scores substituted.','']
    for name in ['baseline','proposed']:
        for filename in ['results.json','failure.json']:
            path=run/name/filename
            if path.exists():lines+=['## '+name+' '+filename,'','```json',path.read_text(),'```','']
    for name in ['teacher_quality_gate.json','source_inventory.json','shared_upstream.json','deployment.json']:
        path=run/'proposed'/name
        if path.exists():lines+=['## '+name,'','```json',path.read_text(),'```','']
    (run/'COMPARISON_REPORT.md').write_text('\n'.join(lines))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--stage',default='orchestrate');args=parser.parse_args()
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;c.seed()
    if args.stage=='orchestrate':
        start=time.perf_counter();protocol=audit(run);gpu_interfaces()
        protocol['inventory_audit_seconds']=time.perf_counter()-start
        protocol['common_setup_includes_GPU_interface_check']=True;c.dump(run/'protocol.json',protocol)
        protocol['execution_order']=['fresh_teacher20k_MLP_plus1_plus2','baseline100k']
        protocol['baseline_time_guards_available_during_proposed']=False
        c.dump(run/'protocol.json',protocol)
        c.dump(run/'allocation.json',dict(gpu=torch.cuda.get_device_name(0),node=os.environ.get('SLURMD_NODENAME'),job=os.environ.get('SLURM_JOB_ID')))
        for phase in ['proposed','baseline']:
            status(run,'START_'+phase.upper())
            c.dump(run/(phase+'_clock.json'),dict(start_monotonic=time.perf_counter(),start_unix=time.time()))
            with open(run/(phase+'.log'),'w',buffering=1) as log:
                child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-dir',str(run),'--stage',phase],stdout=log,stderr=subprocess.STDOUT)
            report(run)
            if child.returncode:
                status(run,'FAILED',phase=phase,exit_code=child.returncode);raise SystemExit(child.returncode)
        finalize_comparison(run)
        status(run,'FINISHED',allocation_experiment_wall_seconds=time.perf_counter()-start);report(run)
    else:
        try:
            protocol=json.loads((run/'protocol.json').read_text())
            (baseline if args.stage=='baseline' else proposed)(run,protocol)
        except BaseException as exc:
            directory=run/args.stage;directory.mkdir(exist_ok=True)
            c.dump(directory/'failure.json',dict(reason=str(exc),type=type(exc).__name__,traceback=traceback.format_exc(),memory=memory()))
            status(run,'FAILED',phase=args.stage,reason=str(exc));raise

if __name__=='__main__':main()



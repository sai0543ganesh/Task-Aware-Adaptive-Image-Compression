"""Fresh proposed-only ResNet pipeline; baseline5345 decoded-source cache unchanged.
No baseline fitting, reused teacher/MLP weights, adaptive cache, or speed-target abort.
"""
import time
START=time.perf_counter()
import csv,gc,hashlib,json,os,platform,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch,torchvision,PIL,sklearn
import simple_grouped_core as c
import augmented_policy as mapping
import shared_resnet_policy as rp
import baseline5345_reference as reference
import common_proposed_execution as execution
import handcrafted_predictor as hand
import dense_heavy_paired_study as fresh
from compression_lowram import trim_unused
import shared_data_policy as shared
from resource_monitor import Monitor

ARCHITECTURE=os.environ['MTP_TASK_ARCHITECTURE']
assert ARCHITECTURE in ['resnet18','densenet121']
JOBROOT=Path(os.environ['MTP_RESEARCH_OUTPUT_ROOT'])/('job_'+os.environ['SLURM_JOB_ID'])/ARCHITECTURE
ARM=os.environ.get('FINAL_RESNET_ARM','proposed')
ROOT=JOBROOT/ARM
RESUME=os.environ.get('MTP_RAM_PREP_RESUME')=='1'
PREFIX=json.loads((ROOT/'COMPRESSION_HANDOFF_STATE.json').read_text()) if RESUME else None
if RESUME:START=PREFIX['original_start_monotonic']
VALIDATE={1,2,3,4,5,10,15,20,25,30,35,40,45,50}
EXPECTED_CACHE={20000:'307984774185f62cba81b12c91b05729d0b8cf9f76d10546943abda496faaf30',80000:'8b93f49294ef5108889157d22b81a44e3db3cf900dbd76660aed6f9a092e68d1'}

def status(stage,**values):
    supplied_architecture=values.pop('architecture',ARCHITECTURE)
    assert supplied_architecture==ARCHITECTURE, supplied_architecture
    c.dump(ROOT/'status.json',dict(stage=stage,updated_unix=time.time(),**values))
    c.dump(JOBROOT/'status.json',dict(stage='MODEL_'+stage,arm=ARM,architecture=ARCHITECTURE,updated_unix=time.time(),**values));c.say(stage,architecture=ARCHITECTURE,**values)

def decoded_source(rows,directory,name):
    directory.mkdir(parents=True,exist_ok=True)
    encoded,info=shared.cache_source(rows,directory,name+'_source')
    tick=time.perf_counter();data=shared.materialize(encoded)
    info.update(materialization_seconds=time.perf_counter()-tick,decoded_shared_RAM_bytes=data.bytes,
        encoded_source_cache=False,encoded_preparation_cache_used=True,training_buffer_policy=shared.POLICY)
    info['seconds']+=info['materialization_seconds']
    del encoded;shared.release_unused()
    expected={**EXPECTED_CACHE,100000:'1f492ebd9454b0daad60a9827f6ca0489f90d374eef8c47d9474cef8fb799ad6'}
    assert info['content_fingerprint']==expected[len(rows)]
    c.dump(directory/(name+'_cache.json'),info)
    return data,info

def fixed(rows,directory,name):
    # Same validation/test preparation helper as historical baseline5345.
    tick=time.perf_counter()
    def prepare(row):
        pixels, _, _ = c.decode_one(row)
        return np.array(PIL.Image.fromarray(pixels).resize((224,224),PIL.Image.Resampling.BILINEAR),copy=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        arrays=list(pool.map(prepare,rows))
    data=c.RAMImages(rows,arrays);shared.release_unused()
    timings_path=ROOT/'fixed_cache_timings.json'
    previous=json.loads(timings_path.read_text()) if timings_path.exists() else []
    previous.append(dict(name=name,images=len(rows),seconds=time.perf_counter()-tick,implementation='hash verified single image decode then direct BILINEAR224; bounded native RGB staging'))
    c.dump(timings_path,previous);return data

def tag(net):
    net._backbone_architecture=ARCHITECTURE
    net._native_bn_spec=[(m.eps,m.momentum) for m in net.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    return net

def fresh_task_model(classes):
    c.seed()
    net=getattr(torchvision.models,ARCHITECTURE)(weights=None,num_classes=len(classes)).cuda()
    assert net.__class__.__name__==('ResNet' if ARCHITECTURE=='resnet18' else 'DenseNet')
    return tag(net)

def current_settings():
    return dict(batch=c.BATCH,pixel_batch_cap=c.PIXELS,training_workers=rp.CONFIG['training_workers'],
       evaluation_workers=rp.CONFIG['evaluation_workers'],pin_memory=rp.CONFIG['pin_memory'],
       persistent_workers=rp.CONFIG['persistent_workers'],prefetch_factor=rp.CONFIG['prefetch_factor'],
       torch_threads=torch.get_num_threads(),precision='FP32',gradient_accumulation=False,
       cudnn_benchmark=torch.backends.cudnn.benchmark,
       matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32)

def assert_settings():
    now=current_settings()
    assert now['batch']==64 and now['training_workers']==now['evaluation_workers']==4
    assert now['pixel_batch_cap']==10**12 and now['torch_threads']==4
    assert now['pin_memory'] and now['persistent_workers'] and now['prefetch_factor']==2
    assert not now['cudnn_benchmark'] and not torch.is_autocast_enabled()
    assert int(os.environ['SLURM_CPUS_PER_TASK'])==16 and int(os.environ['SLURM_MEM_PER_NODE'])==70656
    return now

def run():
    ROOT.mkdir(parents=True,exist_ok=RESUME)
    torch.multiprocessing.set_start_method('fork',force=True);torch.set_num_threads(4);c.seed()
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    assert bool(os.environ.get('SLURMD_NODENAME'))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rp.install(c,'teacher');execution.VALIDATE=set(VALIDATE)
    c.fresh_model=fresh_task_model
    fresh.ROOT=ROOT;fresh.PROGRAM_START=START;fresh.status=status
    # unpack loads manifests/grid only; its archive contains no teacher or MLP weights.
    if RESUME:
        assert PREFIX['job']==os.environ['SLURM_JOB_ID'] and PREFIX['architecture']==ARCHITECTURE and PREFIX['arm']==ARM
        saved=ROOT/'inputs';meta=json.loads((saved/'ASSET_MANIFEST.json').read_text())
        for name,sha in meta['sha256'].items():assert c.sha(saved/name)==sha,name
        tables={n:execution.read(saved/n) for n in ['sources.csv','validation.csv','test.csv']}
        classes=json.loads((saved/'protocol.json').read_text())['classes']
        policy=json.loads((saved/'grid_split_policy.json').read_text())
        columns=[v['feature'] for v in execution.read(saved/'optimized_v28_features.csv')]
        assert len(classes)==100 and len(columns)==215
    else:
        tables,classes,policy,columns,_=fresh.unpack()
    columns=[name for name in columns if not name.startswith(('cheap_v3_mobile_','extra_cam_'))]
    assert len(columns)==32 and set(hand.APPEARANCE) <= set(columns)
    execution.predict80=hand.predict80
    execution.OUT=ROOT;execution.status=status;mapping.install_floor(policy['sides'])
    if not RESUME:
        settings=assert_settings()
        preflight_started=time.perf_counter()
        execution.synthetic_interfaces(classes)
        preflight_seconds=time.perf_counter()-preflight_started
        versions=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
           Pillow=PIL.__version__,numpy=np.__version__,sklearn=sklearn.__version__,CUDA=torch.version.cuda,
           cuDNN=torch.backends.cudnn.version(),node=os.environ['SLURMD_NODENAME'],GPU=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
           GPU_UUID=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
           CPU_affinity=sorted(os.sched_getaffinity(0)))
        c.dump(ROOT/'protocol.json',dict(scope='fresh proposed route in final matched ResNet pair',
           task_architecture=ARCHITECTURE,
           real_training_loop_interface_preflight_seconds=preflight_seconds,
           versions=versions,settings=settings,CPUs=16,RAM_requested_GiB=69,
           source_cache=shared.POLICY,
           evaluation='same1k validation/5k test; directBILINEAR224; recorded validationtop1 selects checkpoint',
           validation_epochs=sorted(VALIDATE),teacher_epochs=50,continuation_epochs=50,
           teacher=ARCHITECTURE+' scratch;RRC224+flip;SGD.1 cosine50;CE;nativeBN',
           mapping='encoded preparation; unchanged BICUBIC/BILINEAR mapping and final decoded training cache',
           old_label_mapping_preserved=False,
           singleton_BN_guard='ResNet: degenerate spatial singleton duplicates; DenseNet: any singleton duplicates, as5561; actual extra forwards logged',
           continuation='new teacher checkpoint;48assigned-sizeRRC+flip SGD.01 cosine48 LS.05;2direct224 epochs SGD.001 CE',
           method_differences_from_baseline=['20k teacher then80k continuation','per-image compression/mapping and grouped sizes',
              'continuation starts atSGD.01 not scratchSGD.1','continuation label smoothing.05','two final direct224 adaptation epochs'],
           no_CutMix=True,no_manual_BN_recalibration=True,no_AMP=True,
           tiers=c.TIERS.tolist(),mapped_sides=policy['sides'],safety_threshold=.27,
           MLP_recipe=hand.CFG,MLP_seeds=hand.SEEDS,MLP_max_iterations=80,
           feature_loader=dict(batch=128,workers=8),mapping_threads=4,
           teacher_and_labels_and_features_and_MLP_fresh=True,task_weights_or_predictions_reused=False,
           frozen_MobileNet_pretrained_weights_reused=False,remaining80k_task_feature_probes=0,
           comparison_scope='proposed only with archived matched-policy baseline evidence',
           no_test_based_selection_or_threshold_tuning=True,development_and_test_identities_previously_reused=True,
           time_accounting='fresh entire pipeline stopwatch across one process-image restart; all creation/reload/cleanup costs included',
           shared_data_policy=shared.POLICY,source_files_read_only=True,no_speed_target_early_abort=True,training_unchanged=True,compression_preparation_process_restart=True,compression_benchmark_precedes_pipeline=False,benchmark_research_time_separately_reported=True,fresh_pair_order_baseline_then_proposed=True))
    else:
        settings=assert_settings();assert settings==PREFIX['settings']
        assert torch.cuda.get_device_name(0)==PREFIX['GPU']
        preflight_seconds=PREFIX['preflight_seconds']
        c.dump(ROOT/'COMPRESSION_HANDOFF_RESTART.json',dict(seconds=time.perf_counter()-PREFIX['restart_begin_monotonic'],same_training_and_feature_settings=True,old_host_allocations_released_by_exec=True,original_pipeline_start_retained=True))
    monitor=Monitor(ROOT)
    if RESUME:monitor.peaks=PREFIX['memory']['phases']
    execution.TRAIN_MONITOR=monitor
    monitor.thread.start();directory=ROOT/'proposed';directory.mkdir(exist_ok=RESUME)
    try:
        torch.cuda.reset_peak_memory_stats()
        if RESUME:
            teacher=PREFIX['teacher'];teacher_cache=PREFIX['teacher_cache'];label_time=PREFIX['labeling'];feature_time=PREFIX['initial_feature_time'];diagnostic=PREFIX['MLP'];mlp_time=PREFIX['MLP_training']
            checkpoints=[]
            for seed in hand.SEEDS:
                path=directory/'mlp'/('deployed_seed'+str(seed)+'.pt');assert c.sha(path)==PREFIX['checkpoint_sha256'][str(seed)]
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                assert Path(checkpoint['parent_teacher']).resolve()==(directory/'teacher/best.pth').resolve()
                assert checkpoint['parent_teacher_sha256']==c.sha(directory/'teacher/best.pth')
                checkpoints.append(checkpoint)
        else:
            monitor.set_phase('initial20k_decoded_cache');status('INITIAL20K_DECODED_CACHE')
            initial=[r for r in tables['sources.csv'] if r['part']=='teacher']
            source,teacher_cache=decoded_source(initial,directory,'initial20k')
            val=fixed(tables['validation.csv'],directory,'teacher_validation')
            monitor.set_phase('initial20k_teacher_training');status('INITIAL20K_TEACHER_TRAINING')
            assert_settings();net=tag(c.fresh_model(classes))
            teacher=execution.train(net,source,val,classes,directory/'teacher','teacher')
            # execution.train restores the best recorded validation-top1 checkpoint.
            monitor.set_phase('new_teacher_labels');status('NEW_TEACHER_LABELS')
            labels,label_time=fresh.label_teacher(net,source,policy,directory/'labels')
            del net,val;gc.collect();torch.cuda.empty_cache()
            monitor.set_phase('initial20k_features');status('INITIAL20K_FEATURES')
            frame,_,feature_time=execution.features.extract(source,columns,False,directory,'initial20k')
            del source;gc.collect();torch.cuda.empty_cache()
            monitor.set_phase('new_MLP_training');status('NEW_MLP_TRAINING')
            checkpoints,diagnostic,mlp_time=hand.fit_mlp(frame,labels,policy,directory/'mlp',directory/'teacher/best.pth')
            assert diagnostic['evaluated_feature_count']==28
            assert checkpoints[0]['feature_columns']==diagnostic['evaluation_features']
            del frame,labels;gc.collect();torch.cuda.empty_cache()
            monitor.sample();monitor.stop.set();monitor.thread.join(timeout=10)
            state=dict(job=os.environ['SLURM_JOB_ID'],architecture=ARCHITECTURE,arm=ARM,original_start_monotonic=START,restart_begin_monotonic=time.perf_counter(),settings=settings,GPU=torch.cuda.get_device_name(0),preflight_seconds=preflight_seconds,teacher=teacher,teacher_cache=teacher_cache,labeling=label_time,initial_feature_time=feature_time,MLP=diagnostic,MLP_training=mlp_time,memory=json.loads((ROOT/'memory_progress.json').read_text()),checkpoint_sha256={str(seed):c.sha(directory/'mlp'/('deployed_seed'+str(seed)+'.pt')) for seed in hand.SEEDS},GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            c.dump(ROOT/'COMPRESSION_HANDOFF_STATE.json',state)
            env=dict(os.environ,MTP_RAM_PREP_RESUME='1')
            os.execve(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve())],env)
            raise RuntimeError('execve unexpectedly returned')
        cleanup_started=time.perf_counter();trim_unused();c.dump(ROOT/'COMPRESSION_HANDOFF_CLEANUP.json',dict(seconds=time.perf_counter()-cleanup_started,unused_allocator_only=True,shared_source_policy=True,old_stage_allocations_released=True))
        monitor.set_phase('remaining80k_encoded_cache');status('REMAINING80K_ENCODED_CACHE')
        remaining=[r for r in tables['sources.csv'] if r['part']=='remaining']
        source,source_cache=shared.cache_source(remaining,directory,'remaining80k_source')
        monitor.set_phase('remaining80k_features_prediction');status('REMAINING80K_FEATURES_PREDICTION')
        applied,prediction=execution.predict80(source,checkpoints,policy,columns,directory/'prediction')
        del checkpoints;gc.collect();torch.cuda.empty_cache()
        monitor.set_phase('remaining80k_compression_mapping');status('REMAINING80K_COMPRESSION_MAPPING')
        tick=time.perf_counter();data=shared.materialize(source,applied,policy['sides'],c.floor_indices(source.shapes))
        storage=dict(source_decoded_RGB_bytes=source_cache['decoded_source_geometry_bytes'],source_encoded_file_bytes=source_cache['source_payload_bytes'],
          mapped_decoded_RGB_bytes=data.bytes,mapping_seconds=time.perf_counter()-tick,mapped_geometry=c.geometry(data),
          persisted_compressed_image_payload_bytes=0,source_files_unchanged=True,compression_preparation=data.source_info)
        with (directory/'applied80k_manifest.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['id','source_h','source_w','tier','retained_area_percent','mapped_h','mapped_w'])
            for i,row in enumerate(source.rows):writer.writerow([row['id'],*source.shapes[i],int(applied[i]),float(c.TIERS[applied[i]]),*data.shapes[i]])
        c.dump(directory/'storage.json',storage)
        del source;gc.collect();torch.cuda.empty_cache()
        # Baseline5345 prepares and retains both evaluation sets before task training.
        val=fixed(tables['validation.csv'],directory,'student_validation')
        ck=torch.load(directory/'teacher/best.pth',map_location='cpu',weights_only=True);assert ck['classes']==classes and ck['backbone']==ARCHITECTURE
        net=tag(c.fresh_model(classes));net.load_state_dict(ck['model_state_dict']);del ck
        monitor.set_phase('remaining80k_continuation');status('REMAINING80K_CONTINUATION');assert_settings()
        training=execution.train(net,data,val,classes,directory/'training','proposed',guard=None)
        monitor.set_phase('test_cache_preparation')
        del data,val;shared.release_unused()
        test=fixed(tables['test.csv'],directory,'test')
        monitor.set_phase('final_test');status('FINAL_TEST')
        result_test=c.evaluate(net,test,directory/'test_predictions.npz');monitor.set_phase('reporting')
        seconds=time.perf_counter()-START
        result=dict(teacher=teacher,teacher_cache=teacher_cache,labeling=label_time,initial_feature_time=feature_time,
          evaluation_cache_timings=json.loads((ROOT/'fixed_cache_timings.json').read_text()),
          real_loop_preflight_seconds=preflight_seconds,
          measured_stage_time_note='each stage is measured independently; residual setup is reported rather than silently omitted',
          MLP=diagnostic,MLP_training=mlp_time,remaining_source_cache=source_cache,prediction=prediction,
          storage=storage,training=training,test=result_test,complete_fresh_process_seconds=seconds,
          timing_basis='fresh whole pipeline wall including one compression-preparation process-image restart, all stages/reloads/cleanup; export follows',
          logical_classifier_training_presentations=5000000,
          task_architecture=ARCHITECTURE,arm=ARM,output_directory=str(ROOT),
          GPU_peak_allocated_bytes=max([torch.cuda.max_memory_allocated(),PREFIX['GPU_peak_allocated_bytes']]+[p.get('GPU_peak_allocated_bytes',0) for p in monitor.peaks.values()]),GPU_peak_reserved_bytes=max([torch.cuda.max_memory_reserved(),PREFIX['GPU_peak_reserved_bytes']]+[p.get('GPU_peak_reserved_bytes',0) for p in monitor.peaks.values()]),
          memory=json.loads((ROOT/'memory_progress.json').read_text()))
        c.dump(ROOT/'results.json',result);status('COMPLETE',seconds=seconds,test=result_test)
    finally:
        monitor.stop.set();monitor.thread.join(timeout=10)


def baseline_run():
    ROOT.mkdir(parents=True,exist_ok=False)
    torch.multiprocessing.set_start_method('fork',force=True);torch.set_num_threads(4);c.seed()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert 'A100' in torch.cuda.get_device_name(0) and bool(os.environ.get('SLURMD_NODENAME'))
    rp.install(c,'baseline');execution.VALIDATE=set(VALIDATE);c.fresh_model=fresh_task_model
    fresh.ROOT=ROOT;fresh.PROGRAM_START=START;fresh.status=status
    tables,classes,policy,columns,_=fresh.unpack()
    execution.OUT=ROOT;execution.status=status;mapping.install_floor(policy['sides'])
    settings=assert_settings();preflight_started=time.perf_counter();execution.synthetic_interfaces(classes)
    preflight_seconds=time.perf_counter()-preflight_started
    versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,CUDA=torch.version.cuda,
        node=os.environ['SLURMD_NODENAME'],GPU=torch.cuda.get_device_name(0),
        GPU_UUID=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    c.dump(ROOT/'protocol.json',dict(task_architecture=ARCHITECTURE,settings=settings,versions=versions,
        shared_data_policy=shared.POLICY,CPUs=16,RAM_requested_GiB=69,epochs=50,
        evaluation='same1k validation/5k test; directBILINEAR224; validationtop1 checkpoint selection',
        source_files_read_only=True,seed=42,training='native source RGB -> RRC224+flip;SGD.1 cosine50;CE;nativeBN',
        evaluation_identities_reused_during_development=True,one_classifier_seed=True,
        preparation_training_validation_test_peaks_measured_separately=True))
    monitor=Monitor(ROOT);execution.TRAIN_MONITOR=monitor;monitor.thread.start()
    try:
        directory=ROOT/'baseline';directory.mkdir()
        monitor.set_phase('baseline_source_preparation');status('BASELINE_SOURCE_PREPARATION')
        source,cache=decoded_source(tables['sources.csv'],directory,'original100k')
        monitor.set_phase('baseline_evaluation_preparation')
        val=fixed(tables['validation.csv'],directory,'validation')
        shared.release_unused();net=tag(c.fresh_model(classes));assert_settings()
        training=execution.train(net,source,val,classes,directory/'training','baseline')
        monitor.set_phase('test_cache_preparation')
        del source,val;shared.release_unused()
        test=fixed(tables['test.csv'],directory,'test')
        monitor.set_phase('final_test');status('FINAL_TEST')
        test_metric=c.evaluate(net,test,directory/'test_predictions.npz')
        monitor.set_phase('reporting')
        gpu=max([torch.cuda.max_memory_allocated()]+[p.get('GPU_peak_allocated_bytes',0) for p in monitor.peaks.values()])
        c.dump(ROOT/'results.json',dict(task_architecture=ARCHITECTURE,arm=ARM,output_directory=str(ROOT),training=training,test=test_metric,source_cache=cache,
            complete_fresh_process_seconds=time.perf_counter()-START,logical_classifier_training_presentations=5000000,
            GPU_peak_allocated_bytes=gpu,memory=json.loads((ROOT/'memory_progress.json').read_text()),
            real_loop_preflight_seconds=preflight_seconds,evaluation_cache_timings=json.loads((ROOT/'fixed_cache_timings.json').read_text())))
        status('COMPLETE',test=test_metric)
    finally:
        monitor.stop.set();monitor.thread.join(timeout=10)

if __name__=='__main__':
    try:
        if ARM=='baseline':baseline_run()
        else:run()
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True);c.dump(ROOT/'failure.json',dict(traceback=traceback.format_exc()));status('FAILED');raise


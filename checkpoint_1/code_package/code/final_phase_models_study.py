"""Fresh paired ResNet experiment on one physical GPU, with shared data policy."""
import hashlib,json,os,pathlib,subprocess,sys,time,traceback
START=time.perf_counter()
ROOT=pathlib.Path(os.environ['MTP_RESEARCH_OUTPUT_ROOT'])/('job_'+os.environ['SLURM_JOB_ID'])
BASE=ROOT
CODE=pathlib.Path(__file__).parent

def dump(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)

def peaks(value):
    return {k:v['sampled_peak_process_tree_PSS_bytes']/2**30 for k,v in value['memory']['phases'].items()}

def run_pair(architecture):
    pair_started=time.perf_counter()
    global ROOT
    ROOT=BASE/architecture
    ROOT.mkdir(parents=True,exist_ok=False)
    audit=json.loads((pathlib.Path(os.environ['MTP_RESEARCH_INPUT_DIR'])/'CODE_AUDIT.json').read_text())
    for name,sha in audit['source_sha256'].items():
        assert hashlib.sha256((CODE/name).read_bytes()).hexdigest()==sha,name
    dump('CODE_AUDIT.json',audit)
    tick=time.perf_counter()
    check=subprocess.run([sys.executable,'-u',str(CODE/'verify_data_policy.py')],capture_output=True,text=True)
    dump('COMMON_DATA_POLICY_CHECK.json',dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr))
    if check.returncode:raise RuntimeError('Common data-policy check failed before training')
    common_seconds=time.perf_counter()-tick
    completed={}
    for arm in ['baseline','proposed']:
        dump('status.json',dict(stage='STARTING',arm=arm,updated_unix=time.time()))
        env=dict(os.environ,MTP_TASK_ARCHITECTURE=architecture,FINAL_RESNET_ARM=arm)
        env.pop('MTP_RAM_PREP_RESUME',None)
        with (ROOT/(arm+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,'-u',str(CODE/'final_worker.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(arm+' worker failed; inspect preserved logs and failure.json')
        completed[arm]=json.loads((ROOT/arm/'results.json').read_text())
        assert completed[arm]['task_architecture']==architecture
        assert completed[arm]['arm']==arm
        assert pathlib.Path(completed[arm]['output_directory']).resolve()==(ROOT/arm).resolve()
        dump('partial_results.json',completed)
    bp=json.loads((ROOT/'baseline/protocol.json').read_text());pp=json.loads((ROOT/'proposed/protocol.json').read_text())
    assert bp['settings']==pp['settings']
    for key in ['node','GPU','GPU_UUID','CUDA_VISIBLE_DEVICES']:
        assert bp['versions'][key]==pp['versions'][key],key
    assert bp['shared_data_policy']==pp['shared_data_policy']
    for name in ['sources.csv','validation.csv','test.csv']:
        assert (ROOT/'baseline/inputs'/name).read_bytes()==(ROOT/'proposed/inputs'/name).read_bytes()
    for v in completed.values():assert v['logical_classifier_training_presentations']==5000000
    b,p=completed['baseline'],completed['proposed'];bm,pm=peaks(b),peaks(p)
    assert all(k in pm for k in ['teacher_training','proposed_training']) and 'baseline_training' in bm
    def audit_arm(arm,value):
        import numpy as np
        target=ROOT/arm/('baseline' if arm=='baseline' else 'proposed')
        pred=np.load(target/'test_predictions.npz');prob=pred['p'];labels=pred['labels']
        top1=100*float((prob.argmax(1)==labels).mean())
        top5=100*float((np.argsort(prob,axis=1)[:,-5:]==labels[:,None]).any(1).mean())
        assert abs(top1-value['test']['top1_percent'])<1e-9 and abs(top5-value['test']['top5_percent'])<1e-9
        return dict(images=len(labels),top1=top1,top5=top5)
    independent={a:audit_arm(a,v) for a,v in completed.items()}
    bt=b['complete_fresh_process_seconds'];pt=p['complete_fresh_process_seconds']
    bpeak=max(bm.values());ppeak=max(pm.values())
    result=dict(task_architecture=architecture,settings_match=True,same_physical_GPU=True,
        shared_data_policy=bp['shared_data_policy'],independent_test_metrics=independent,
        outside_route_data_check_seconds=common_seconds,no_shared_time_added_to_route_measurements=True,baseline_seconds=bt,proposed_seconds=pt,time_saved_percent=100*(1-pt/bt),
        baseline_whole_peak_PSS_GiB=bpeak,proposed_whole_peak_PSS_GiB=ppeak,
        whole_peak_RAM_saved_percent=100*(1-ppeak/bpeak),
        baseline_training_only_peak_PSS_GiB=bm['baseline_training'],
        proposed_teacher_training_only_peak_PSS_GiB=pm['teacher_training'],
        proposed_80k_training_only_peak_PSS_GiB=pm['proposed_training'],
        baseline_phase_peaks_GiB=bm,proposed_phase_peaks_GiB=pm,
        training_only_definition='epoch loops including iterator wait/forward/backward/update; excludes cache preparation, validation and checkpoint saving; retains the buffers required for training',
        top1_loss_points=b['test']['top1_percent']-p['test']['top1_percent'],
        top5_loss_points=b['test']['top5_percent']-p['test']['top5_percent'],
        proposed_MLP=p['MLP'],GPU_peaks_allocated_GiB={a:v['GPU_peak_allocated_bytes']/2**30 for a,v in completed.items()},
        total_pair_process_seconds=time.perf_counter()-pair_started,
        allocation_elapsed_seconds=time.perf_counter()-START,
        timing_boundary='Each fresh route includes imports, input verification, all loading/preparation, fitting, validation/checkpoints and test; archive transport follows',
        limitations=['one classifier seed','evaluation identities reused during earlier development','fixed execution order baseline then proposed; OS cache conditions can differ','sampled PSS can miss transients','whole method includes split training, mapping, regularization/adaptation and MLP overhead; not compression-only ablation'])
    dump('comparison.json',result);dump('results.json',completed)
    dump('status.json',dict(stage='COMPLETE',updated_unix=time.time()))
if __name__=='__main__':
    try:
        BASE.mkdir(parents=True,exist_ok=False)
        for architecture in ['resnet18','densenet121']:
            ROOT=BASE
            dump('status.json',dict(stage='STARTING_PAIR',architecture=architecture,updated_unix=time.time()))
            run_pair(architecture)
        ROOT=BASE
        dump('comparisons.json',{name:json.loads((BASE/name/'comparison.json').read_text()) for name in ['resnet18','densenet121']})
        dump('status.json',dict(stage='COMPLETE',updated_unix=time.time()))
    except Exception:
        failure=dict(traceback=traceback.format_exc(),failed_directory=str(ROOT))
        dump('failure.json',failure)
        ROOT=BASE
        dump('failure.json',failure)
        dump('status.json',dict(stage='FAILED',updated_unix=time.time(),failure=failure))
        raise

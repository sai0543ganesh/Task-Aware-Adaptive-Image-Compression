"""Original100k baseline, matching 5327 task recipe, with measured worker memory."""
import time
PROCESS_START=time.perf_counter()
import csv,gc,hashlib,json,os,resource,threading,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import simple_grouped_core as c
import augmented_policy as policy

ROOT=Path('/userhome/mtech/s.chatharasupalli/imagenet100_mtp')
SOURCE=ROOT/'results/original_full_v1/job_5327'
OUT=ROOT/'results/original_baseline_matched_v1'/('job_'+os.environ['SLURM_JOB_ID'])

def read(name):
    with (SOURCE/name).open(newline='') as f:return list(csv.DictReader(f))

class MemoryMonitor:
    def __init__(self):
        self.stop=threading.Event();self.peak_pss=0;self.peak_rss=0;self.samples=0;self.errors=0
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def snapshot(self):
        rss=pss=0;complete=True
        pending=[os.getpid()];seen=set()
        while pending:
            pid=pending.pop()
            if pid in seen:continue
            seen.add(pid)
            try:
                for task in (Path('/proc')/str(pid)/'task').iterdir():
                    pending.extend(map(int,(task/'children').read_text().split()))
                counters={}
                for line in (Path('/proc')/str(pid)/'smaps_rollup').read_text().splitlines():
                    if ':' in line:
                        key,value=line.split(':',1)
                        if key in ['Rss','Pss']:counters[key]=int(value.split()[0])*1024
                rss+=counters['Rss'];pss+=counters['Pss']
            except (OSError,KeyError,ValueError):complete=False
        self.peak_rss=max(self.peak_rss,rss)
        if complete:self.peak_pss=max(self.peak_pss,pss)
        else:self.errors+=1
        self.samples+=1
        return dict(sampled_process_tree_peak_PSS_bytes=self.peak_pss,
            sampled_process_tree_peak_summed_RSS_bytes=self.peak_rss,samples=self.samples,
            incomplete_samples=self.errors,sampling_interval_seconds=5,
            interpretation='PSS apportions shared pages across processes; summed RSS can double-count them; sampled peaks may miss transients',
            main_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    def loop(self):
        while not self.stop.is_set():
            try:c.dump(OUT/'memory_progress.json',self.snapshot())
            except Exception as e:c.say('MEMORY_SAMPLE_ERROR',error=str(e))
            self.stop.wait(5)

def header(row):
    p=Path(row['path'])
    with Image.open(p) as im:w,h=im.size
    return h,w,p.stat().st_size

def stream_cache(rows):
    """Same stored decodedRGB as 5327, packed incrementally to avoid two full copies."""
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:heads=list(pool.map(header,rows))
    total=sum(h*w*3 for h,w,_ in heads)
    if total>80*1024**3:raise RuntimeError('Decoded cache would leave insufficient headroom within96GiB allocation')
    data=c.RAMImages.__new__(c.RAMImages);data.rows=rows;data.shapes=[(h,w) for h,w,_ in heads]
    # Linux fork shares these read-only pages through copy-on-write. Do not
    # require a60GiB POSIX shared-memory allocation on a potentially64GiB tmpfs.
    data.bytes=total;data.offsets=[];data.source_info={};data.pixels=torch.empty(total,dtype=torch.uint8)
    cursor=0;fingerprint=hashlib.sha256()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for first in range(0,len(rows),128):
            for i,(a,d,n) in enumerate(pool.map(c.decode_one,rows[first:first+128]),first):
                assert a.shape[:2]==data.shapes[i]
                data.offsets.append(cursor);data.pixels[cursor:cursor+a.size].copy_(torch.from_numpy(a.reshape(-1)))
                cursor+=a.size;fingerprint.update((str(rows[i]['label'])+'|'+rows[i]['path']+'|'+d+'\n').encode())
            if first//128%40==0:c.say('SOURCE_RAM_PROGRESS',images=min(first+128,len(rows)),total=len(rows))
    assert cursor==total
    info=dict(images=len(rows),source_payload_bytes=sum(n for _,_,n in heads),decoded_shared_RAM_bytes=total,
        seconds=time.perf_counter()-start,content_fingerprint=fingerprint.hexdigest(),all_source_file_hashes_verified=True,
        repeated_image_file_reads_during_training=False,packing='bounded incremental decode; no duplicate whole100k decoded buffer',
        cache_sharing='Linux fork copy-on-write; decoded source tensor read-only after packing')
    data.source_info=info;c.dump(OUT/'source_cache.json',info);return data,info

def fixed_cache(rows,name):
    source,info=c.cache(rows,OUT,name+'_source')
    arrays=[np.array(source.image(i).resize((224,224),Image.Resampling.BILINEAR),dtype=np.uint8,copy=True) for i in range(len(source))]
    fixed=c.RAMImages(rows,arrays);return fixed

def run():
    OUT.mkdir(parents=True,exist_ok=False);monitor=MemoryMonitor();monitor.thread.start()
    try:
        torch.multiprocessing.set_start_method('fork',force=True)
        c.seed();torch.set_num_threads(4);policy.install_augmentation()
        assert torch.cuda.is_available() and 'P100' in torch.cuda.get_device_name(0)
        rows=read('sources.csv');vr=read('validation.csv');tr=read('test.csv')
        classes=json.loads((SOURCE/'protocol.json').read_text())['classes']
        assert len(rows)==100000 and len(vr)==1000 and len(tr)==5000 and len(classes)==100
        assert sum('/compression_200/' in r['path'] for r in rows)==20000
        assert sum('/train_800/' in r['path'] for r in rows)==80000
        for r in rows+vr+tr:assert classes[int(r['label'])]==r['class_name']
        trainhash={r['source_sha256'] for r in rows};vh={r['sha256'] for r in vr};th={r['sha256'] for r in tr}
        assert not trainhash&vh and not trainhash&th and not vh&th
        for name in ['sources.csv','validation.csv','test.csv']:(OUT/name).write_bytes((SOURCE/name).read_bytes())
        c.dump(OUT/'protocol.json',dict(source_job=5327,classes=classes,epochs=50,batch=64,seed=42,
            model='torchvision ResNet18 scratch; standard BatchNorm',
            train='stored originalRGB -> RandomResizedCrop224 + horizontalflip + ImageNet normalization',
            evaluation='same5327 independent1000validation/5000test; directBILINEAR224 + ImageNet normalization',
            optimizer='SGD lr0.1 momentum0.9 weight_decay0.0001 cosine50',
            comparison='5327 continuation lr0.01 and mapped inputs; baseline scratch lr0.1 and224 as5325 baseline',
            checkpoint_selection='validation top1 only; validation epochs1-5 then every5',
            preprocessing_code_source='frozen job5327 code; source/hash comparison saved before submission',
            presentations=5000000,source_cache='decodedRGB inRAM; one-time packing timed',node=os.environ.get('SLURMD_NODENAME')))
        c.dump(OUT/'status.json',dict(stage='CACHE_ORIGINAL100K'))
        data,info=stream_cache(rows);val=fixed_cache(vr,'validation');test=fixed_cache(tr,'test');gc.collect()
        canvas=policy.CanonicalTrainingCanvas(data);net=c.fresh_model(classes);torch.cuda.reset_peak_memory_stats()
        c.dump(OUT/'status.json',dict(stage='BASELINE_TRAINING'))
        training=c.train(net,canvas,val,classes,50,.1,OUT/'training','original100k_baseline')
        metric=c.evaluate(net,test,OUT/'test_predictions.npz');memory=monitor.snapshot()
        memory.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        seconds=time.perf_counter()-PROCESS_START
        proposed=json.loads((SOURCE/'downstream/results.json').read_text())['proposed'];comparison={}
        for name,a in proposed.items():
            pt=a['complete_accounted_seconds'];drop=metric['top1_percent']-a['test']['top1_percent']
            comparison[name]=dict(top1=a['test']['top1_percent'],top5=a['test']['top5_percent'],complete_accounted_seconds=pt,
                time_saved_seconds=seconds-pt,time_saved_percent=100*(1-pt/seconds),accuracy_drop_points=drop,
                accepted=pt<seconds and drop<=3,
                memory_comparison='5327 has no matching process-tree PSS; do not compute totalRAMsaved from unlike measurements')
        c.dump(OUT/'results.json',dict(training=training,test=metric,complete_process_seconds=seconds,
            source_cache=info,memory=memory,proposed5327_comparison=comparison,
            evaluation_manifest_hashes={k:c.sha(SOURCE/k) for k in ['validation.csv','test.csv']},
            GPU=torch.cuda.get_device_name(0),precision='FP32',teacher_or_MLP_retrained=False))
        c.dump(OUT/'status.json',dict(stage='COMPLETE',complete_process_seconds=seconds));c.say('BASELINE_COMPLETE',test=metric,seconds=seconds)
    finally:
        monitor.stop.set();monitor.thread.join(timeout=10)

if __name__=='__main__':
    try:run()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);c.dump(OUT/'failure.json',dict(traceback=traceback.format_exc()));c.dump(OUT/'status.json',dict(stage='FAILED'));raise



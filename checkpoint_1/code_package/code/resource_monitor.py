"""Sample process-tree PSS without counting shared pages once per worker."""
import os,resource,threading,time
from pathlib import Path
import simple_grouped_core as c

class Monitor:
    def __init__(self,directory):
        self.directory=Path(directory);self.stop=threading.Event();self.lock=threading.RLock();self.phase='setup';self.peaks={}
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def set_phase(self,name):
        import torch
        with self.lock:
            self.sample()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                old=self.peaks.setdefault(self.phase,{})
                old['GPU_peak_allocated_bytes']=max(old.get('GPU_peak_allocated_bytes',0),torch.cuda.max_memory_allocated())
                old['GPU_peak_reserved_bytes']=max(old.get('GPU_peak_reserved_bytes',0),torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()
            self.phase=name
            self.sample()
    def sample(self):
        with self.lock:
            pending=[os.getpid()];seen=set();pss=rss=0;complete=True
            while pending:
                pid=pending.pop()
                if pid in seen:continue
                seen.add(pid)
                try:
                    for task in (Path('/proc')/str(pid)/'task').iterdir():pending.extend(map(int,(task/'children').read_text().split()))
                    d={}
                    for line in (Path('/proc')/str(pid)/'smaps_rollup').read_text().splitlines():
                        if ':' in line:
                            k,v=line.split(':',1)
                            if k in ['Rss','Pss']:d[k]=int(v.split()[0])*1024
                    pss+=d['Pss'];rss+=d['Rss']
                except (OSError,KeyError,ValueError):complete=False
            with self.lock:
                p=self.peaks.setdefault(self.phase,dict(sampled_peak_process_tree_PSS_bytes=0,sampled_peak_summed_RSS_bytes=0,samples=0,incomplete_samples=0))
                p['samples']+=1;p['incomplete_samples']+=int(not complete)
                if complete and pss>p['sampled_peak_process_tree_PSS_bytes']:
                    p['sampled_peak_process_tree_PSS_bytes']=pss
                    p['peak_PSS_unix_time']=time.time()
                p['sampled_peak_summed_RSS_bytes']=max(p['sampled_peak_summed_RSS_bytes'],rss)
                output={k:dict(v) for k,v in self.peaks.items()}
                # Serialize the write with both background and explicit samples.
                # c.dump uses a common temporary filename for this output path.
                c.dump(self.directory/'memory_progress.json',dict(phases=output,interval_seconds=5,
                    main_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    note='Phase PSS samples apportion shared pages; RSS is cumulative/summed and is not unique physicalRAM. Sampling can miss peaks.'))
            return output
    def loop(self):
        while not self.stop.is_set():
            try:self.sample()
            except Exception as e:c.say('MEMORY_SAMPLE_ERROR',error=str(e))
            self.stop.wait(5)



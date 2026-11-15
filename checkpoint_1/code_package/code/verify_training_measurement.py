"""CPU regression for the actual phase-instrumented training loop, no task data."""
import os,pathlib,tempfile
ROOT=pathlib.Path(tempfile.mkdtemp(prefix='mtp-phase-regression-'))
os.environ['MTP_RESEARCH_OUTPUT_ROOT']=str(ROOT)
os.environ['SLURM_JOB_ID']='cpucheck'
import json,gc,torch,torchvision
import simple_grouped_core as c
import common_proposed_execution as execution
import shared_resnet_policy as rp
from resource_monitor import Monitor

def main():
    torch.set_num_threads(4);rp.install(c,'baseline');c.sync=lambda:None
    original_tensor_cuda=torch.Tensor.cuda;original_module_cuda=torch.nn.Module.cuda
    torch.Tensor.cuda=lambda self,*args,**kwargs:self
    torch.nn.Module.cuda=lambda self,*args,**kwargs:self
    original_zeros=torch.zeros
    def cpu_zeros(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_zeros(*args,**kwargs)
    torch.zeros=cpu_zeros
    reports={}
    try:
        for architecture in ['resnet18','densenet121']:
            def model(classes):
                c.seed();net=getattr(torchvision.models,architecture)(weights=None,num_classes=len(classes))
                net._backbone_architecture=architecture;return net
            c.fresh_model=model;out=ROOT/architecture;out.mkdir();execution.OUT=out
            monitor=Monitor(out);execution.TRAIN_MONITOR=monitor;monitor.thread.start()
            try:
                execution.synthetic_interfaces(list(range(100)))
                monitor.set_phase('finished');phases=json.loads((out/'memory_progress.json').read_text())['phases']
                for name in ['interface_baseline_training','interface_baseline_validation','interface_student_training','interface_student_validation']:
                    assert phases[name]['samples']>0 and phases[name]['sampled_peak_process_tree_PSS_bytes']>0,name
                reports[architecture]=dict(passed=True,training_and_validation_phases_separate=True,phase_names=list(phases))
            finally:monitor.stop.set();monitor.thread.join(timeout=10);execution.TRAIN_MONITOR=None
            gc.collect()
    finally:
        torch.Tensor.cuda=original_tensor_cuda;torch.nn.Module.cuda=original_module_cuda;torch.zeros=original_zeros
    print(json.dumps(dict(CPU_only=True,synthetic_images_only=True,reports=reports)))
if __name__=='__main__':main()

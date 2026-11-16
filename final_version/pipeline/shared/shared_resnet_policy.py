"""Prepared common policy for a later matched pair; never starts training itself."""
from torch.utils.data import DataLoader
import torch

CONFIG=dict(node='gpu-P100-01',GPU_count=1,CPUs=16,host_memory_GiB=96,
            maximum_batch_images=64,training_workers=4,evaluation_workers=4,
            pin_memory=True,persistent_workers=True,prefetch_factor=2,
            precision='FP32',cudnn_benchmark=False,torch_threads=4,
            pixel_batch_cap=10**12,
            gradient_accumulation=False,teacher_epochs=50,student_epochs=50,
            baseline_epochs=50,logical_presentations=5000000)

def common_loader(core,view,training):
    sampler=core.Buckets(view,training)
    stream=DataLoader(view,batch_sampler=sampler,collate_fn=core.collate,
        num_workers=CONFIG['training_workers'] if training else CONFIG['evaluation_workers'],
        pin_memory=CONFIG['pin_memory'],persistent_workers=CONFIG['persistent_workers'],
        prefetch_factor=CONFIG['prefetch_factor'])
    assert stream.num_workers==4 and core.BATCH==64 and core.PIXELS==CONFIG['pixel_batch_cap']
    assert stream.pin_memory and stream.persistent_workers and stream.prefetch_factor==2
    return stream,sampler

def install(core,kind,mapping=None,adaptation=None):
    assert kind in ['baseline','teacher','student']
    core.BATCH=64;core.PIXELS=CONFIG['pixel_batch_cap'];torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    def loader(data,training=False):
        view=mapping.AugmentedView(data) if kind=='student' and training else data
        return common_loader(core,view,training)
    core.loader=loader
    if adaptation is not None:
        adaptation.loader=lambda data:common_loader(core,adaptation.Fixed224(data),True)
    return dict(CONFIG,arm=kind)

def check_actual(stream,core):
    assert core.BATCH==CONFIG['maximum_batch_images']
    assert core.PIXELS==CONFIG['pixel_batch_cap']
    assert stream.num_workers==CONFIG['training_workers']
    assert stream.prefetch_factor==CONFIG['prefetch_factor']
    assert not torch.backends.cudnn.benchmark
    return dict(CONFIG)




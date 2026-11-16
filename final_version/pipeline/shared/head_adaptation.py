"""Low-LR full-network adaptation to the direct evaluation representation."""
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import simple_grouped_core as c
CONFIG=dict(epochs=5,learning_rate=.001,optimizer='SGD momentum0.9 weight_decay0.0001',
            trainable_parameters='all ResNet parameters',BN_running_updates='all standard BatchNorm layers during adaptation',
            initial_training_only_BN_recalibration_images=2048,
            input_policy='directPIL_BILINEAR224 from same compressed native pixels; no crop/flip',
            extra_logical_training_presentations=0)

class Fixed224:
    def __init__(self,data):self.data=data;self.rows=data.rows;self.shapes=[(224,224)]*len(data)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        im=self.data.image(i).resize((224,224),Image.Resampling.BILINEAR)
        x=torch.from_numpy(np.array(im,dtype=np.uint8,copy=True)).permute(2,0,1).float()/255
        return (x-c.MEAN)/c.STD,int(self.rows[i]['label']),i

def loader(data):
    view=Fixed224(data);sampler=c.Buckets(view,True)
    return DataLoader(view,batch_sampler=sampler,collate_fn=c.collate,num_workers=8,
                      pin_memory=True,persistent_workers=True,prefetch_factor=2),sampler

def mode(net):
    net.train()

def begin(net):
    for name,p in net.named_parameters():
        p.requires_grad_(True);p.grad=None
    mode(net)
    optimizer=torch.optim.SGD(net.parameters(),lr=CONFIG['learning_rate'],momentum=.9,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=5)
    names=[name for name,p in net.named_parameters() if p.requires_grad]
    assert len(names)>2 and all(p.requires_grad for p in net.parameters())
    return optimizer,schedule,dict(config=CONFIG,trainable_names=names,
        trainable_parameter_count=sum(p.numel() for p in net.parameters() if p.requires_grad))




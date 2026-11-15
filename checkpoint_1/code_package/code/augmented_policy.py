"""Shared oracle/deployment size floor and assigned-size augmentation."""
import math
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import simple_grouped_core as c

BASE_FLOOR=c.floor_indices
BASE_LOADER=c.loader
SIDES=None

def side_for(h,w,tier,sides):
    q=float(c.TIERS[int(tier)])
    hh=max(1,round(h*math.sqrt(q/100)));ww=max(1,round(w*math.sqrt(q/100)))
    valid=[s for s in sides if s*s<=hh*ww]
    if not valid:raise RuntimeError('Image below smallest frozen mapped canvas; refusing implicit upscale')
    return max(valid)

def postmapping_floor(shapes,minimum=96):
    if SIDES is None:raise RuntimeError('Mapping must be frozen before computing deployment floor')
    base=BASE_FLOOR(shapes,minimum);result=[]
    for (h,w),first in zip(shapes,base):
        source_max=side_for(h,w,10,SIDES)
        # Never invent recovered detail in an already tiny source. Preserve the
        # largest available canvas for such cases and report them separately.
        required=min(minimum,source_max)
        allowed=[]
        for j in range(int(first),11):
            q=float(c.TIERS[j])
            reduced_area=max(1,round(h*math.sqrt(q/100)))*max(1,round(w*math.sqrt(q/100)))
            if reduced_area<min(SIDES)**2:continue
            if side_for(h,w,j,SIDES)>=required:allowed.append(j)
        if not allowed:raise RuntimeError('No mapping satisfies final-resolution safety guard')
        result.append(allowed[0])
    return np.asarray(result,dtype=np.int64)

def install_floor(sides):
    global SIDES
    SIDES=sorted(map(int,sides));assert len(set(SIDES))==11
    c.floor_indices=postmapping_floor

class AugmentedView:
    def __init__(self,data):self.data=data;self.rows=data.rows;self.shapes=data.shapes
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        im=self.data.image(i);h,w=self.shapes[i]
        top,left,ch,cw=transforms.RandomResizedCrop.get_params(im,scale=(.08,1.),ratio=(.75,4/3))
        im=F.resized_crop(im,top,left,ch,cw,[h,w],interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)
        if float(torch.rand(()))<.5:im=F.hflip(im)
        x=F.to_tensor(im);return (x-c.MEAN)/c.STD,int(self.rows[i]['label']),i

class CanonicalTrainingCanvas:
    """Baseline: source RGB in RAM; augmentation outputs224, without a prior square warp."""
    def __init__(self,data):self.data=data;self.rows=data.rows;self.shapes=[(224,224)]*len(data)
    def __len__(self):return len(self.rows)
    def image(self,i):return self.data.image(i)

def augmented_loader(data,training=False):
    return BASE_LOADER(AugmentedView(data) if training else data,training)

def install_augmentation():
    c.PIXELS=10**12;c.loader=augmented_loader



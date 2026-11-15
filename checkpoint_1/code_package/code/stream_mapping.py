"""Pixel-identical mapping into one final RAM buffer, no whole-output array list.
Source representation, BOX/BILINEAR, tiers, floors, worker count and dataset API unchanged.
"""
import math,os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import simple_grouped_core as c

def map_selected_streaming(data,tiers,sides,floor,workers=4,ram_factory=None):
    if workers<1:raise ValueError('workers must be positive')
    if ram_factory is not c.RAMImages:raise ValueError('Use the unchanged RAMImages dataset class')
    tiers=np.asarray(tiers);floor=np.asarray(floor);n=len(data)
    assert tiers.shape==floor.shape==(n,) and np.issubdtype(tiers.dtype,np.integer) and np.issubdtype(floor.dtype,np.integer)
    assert np.all((tiers>=0)&(tiers<=10)&(floor>=0)&(floor<=10))
    sides=sorted(set(map(int,sides)));assert len(sides)==11 and sides[0]>0
    applied=np.maximum(tiers,floor);shapes=[];reduced=[];offsets=[];total=0
    for i,(h,w) in enumerate(data.shapes):
        q=float(c.TIERS[applied[i]]);hh=max(1,round(h*math.sqrt(q/100)));ww=max(1,round(w*math.sqrt(q/100)))
        allowed=[s for s in sides if s*s<=hh*ww]
        if not allowed:raise ValueError('Smallest frozen canvas exceeds pixel budget')
        side=max(allowed);offsets.append(total);total+=side*side*3;shapes.append((side,side));reduced.append((ww,hh))
    stat=os.statvfs('/dev/shm');free=stat.f_bavail*stat.f_frsize
    if total>free*.8:raise RuntimeError('Insufficient shared RAM for final mapped cache')
    output=ram_factory.__new__(ram_factory);output.rows=[dict(r) for r in data.rows]
    output.offsets=offsets;output.shapes=shapes;output.bytes=total;output.source_info={}
    output.pixels=torch.empty(total,dtype=torch.uint8).share_memory_()
    packed=output.pixels.numpy()
    def chunk(first):
        for i in range(first,min(first+16,n)):
            im=data.image(i)
            if im.mode!='RGB':raise ValueError('Expected RGB')
            if float(c.TIERS[applied[i]])<100:im=im.resize(reduced[i],Image.Resampling.BOX)
            side=shapes[i][0];im=im.resize((side,side),Image.Resampling.BILINEAR)
            value=np.asarray(im,dtype=np.uint8).reshape(-1);off=offsets[i]
            np.copyto(packed[off:off+value.size],value)
    if workers==1:
        for first in range(0,n,16):chunk(first)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(chunk,range(0,n,16)):pass
    return output


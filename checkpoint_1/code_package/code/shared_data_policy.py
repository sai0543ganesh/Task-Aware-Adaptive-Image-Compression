"""Identical source-loading and final RGB buffer policy for both ResNet routes."""
import gc, math, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import simple_grouped_core as c
import source_bytes_cache as encoded
from compression_lowram import trim_unused

POLICY=dict(source='unchanged encoded file bytes, read and hash-verified once per required stage',
    decoded_training_buffer='one private anonymous contiguous uint8 RGB tensor, read-only fork sharing',
    source_header_threads=16,materialization_threads=4,materialization_wave=256,
    release='release encoded source before classifier fitting; release stage buffers before loading next stage')

def cache_source(rows,directory,name):
    data,info=encoded.cache(rows,directory,name,workers=16)
    info['shared_policy']=POLICY
    c.dump(directory/(name+'_cache.json'),info)
    return data,info

def materialize(source,tiers=None,sides=None,floor=None):
    start=time.perf_counter();compressed=tiers is not None
    shapes=[];offsets=[];reduced=[];total=0
    if compressed:
        tiers=np.maximum(np.asarray(tiers),np.asarray(floor));sides=sorted(map(int,sides))
        assert len(tiers)==len(source) and len(sides)==11
    for i,(h,w) in enumerate(source.shapes):
        if compressed:
            q=float(c.TIERS[tiers[i]]);hh=max(1,round(h*math.sqrt(q/100)));ww=max(1,round(w*math.sqrt(q/100)))
            allowed=[s for s in sides if s*s<=hh*ww]
            if not allowed:raise ValueError('Frozen compression canvas exceeds retained pixel budget')
            side=max(allowed);shape=(side,side);reduced.append((ww,hh))
        else:shape=(h,w)
        offsets.append(total);shapes.append(shape);total+=math.prod(shape)*3
    output=c.RAMImages.__new__(c.RAMImages);output.rows=[dict(r) for r in source.rows]
    output.shapes=shapes;output.offsets=offsets;output.bytes=total
    output.pixels=torch.empty(total,dtype=torch.uint8)
    packed=output.pixels.numpy()
    def convert(i):
        im=source.image(i)
        if compressed:
            if float(c.TIERS[tiers[i]])<100:im=im.resize(reduced[i],Image.Resampling.BOX)
            side=shapes[i][0];im=im.resize((side,side),Image.Resampling.BILINEAR)
        a=np.asarray(im,dtype=np.uint8).reshape(-1)
        np.copyto(packed[offsets[i]:offsets[i]+a.size],a)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for first in range(0,len(source),256):
            for _ in pool.map(convert,range(first,min(first+256,len(source)))):pass
    output.source_info=dict(shared_policy=POLICY,compressed=compressed,seconds=time.perf_counter()-start)
    return output

def release_unused():
    gc.collect();torch.cuda.empty_cache();trim_unused()

"""Same lossless source-cache policy for every branch under a fixed RAM budget.

No resizing or pixel changes. Small source sets cache decodedRGB; large ones
retain original encoded bytes in RAM and decode within data-loader workers.
"""
import gc,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import simple_grouped_core as c
from saved5327_encoded_ram import EncodedRAM,load_encoded

DECODED_CACHE_LIMIT_BYTES=16*2**30

class ForkRAMImages(c.RAMImages):
    """Read-only pixel storage shared by Linux fork, without a /dev/shm quota."""
    def __init__(self,rows,arrays,source_info=None):
        self.rows=rows;self.offsets=[];self.shapes=[]
        total=sum(a.size for a in arrays);self.pixels=torch.empty(total,dtype=torch.uint8)
        cursor=0
        for i,a in enumerate(arrays):
            h,w,channels=a.shape
            if channels!=3 or min(h,w)<1:raise ValueError('Invalid RGB source')
            self.offsets.append(cursor);self.shapes.append((h,w))
            self.pixels[cursor:cursor+a.size].copy_(torch.from_numpy(a.reshape(-1)))
            cursor+=a.size;arrays[i]=None
        self.bytes=total;self.source_info=source_info or {}

class ForkEncodedRAM(EncodedRAM):
    def __init__(self,rows,total):
        self.rows=rows;self.pixels=torch.empty(total,dtype=torch.uint8)
        self.offsets=[];self.lengths=[];self.shapes=[];self.bytes=total;self.source_info={}

def install_fork_storage():
    import multiprocessing,sys
    assert sys.platform=='linux' and multiprocessing.get_start_method(allow_none=True) in [None,'fork']
    c.RAMImages=ForkRAMImages

def cache(rows,out,name,workers=16):
    before=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    total=sum(int(r['bytes']) for r in rows);source=ForkEncodedRAM(rows,total)
    cursor=0;identity=hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bound queued raw-byte objects as well as the final shared buffer.
        for first in range(0,len(rows),512):
            for i,(raw,digest,h,w) in enumerate(pool.map(load_encoded,rows[first:first+512]),first):
                n=len(raw);assert cursor+n<=total
                source.offsets.append(cursor);source.lengths.append(n);source.shapes.append((h,w))
                source.pixels[cursor:cursor+n].copy_(torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()))
                rows[i]['encoded_bytes']=n;cursor+=n
                identity.update((str(rows[i]['label'])+'|'+rows[i]['path']+'|'+digest+'\n').encode())
            c.say('BOUNDED_SOURCE_CACHE',split=name,images=min(first+512,len(rows)),total=len(rows))
    assert cursor==total
    decoded=sum(h*w*3 for h,w in source.shapes)
    # Verify the byte-cache path preserves the exact decoded source pixels.
    for i in sorted(set([0,len(rows)//2,len(rows)-1])):
        with Image.open(rows[i]['path']) as im:original=np.array(im.convert('RGB'),dtype=np.uint8)
        assert np.array_equal(original,source.rgb(i)),'Source decoding parity failed'
    encoded_load=time.perf_counter()-before
    if decoded<=DECODED_CACHE_LIMIT_BYTES:
        with ThreadPoolExecutor(max_workers=8) as pool:
            arrays=list(pool.map(source.rgb,range(len(source))))
        data=ForkRAMImages(rows,arrays);del arrays,source;gc.collect();mode='decodedRGB_fork_sharedRAM'
    else:data=source;mode='original_encoded_bytes_sharedRAM'
    info=dict(images=len(rows),seconds=time.perf_counter()-before,encoded_load_seconds=encoded_load,
        source_payload_bytes=total,source_decoded_RGB_bytes=decoded,source_cache_mode=mode,
        source_cache_actual_buffer_bytes=data.bytes,decoded_cache_limit_bytes=DECODED_CACHE_LIMIT_BYTES,
        content_fingerprint=identity.hexdigest(),original_pixels_unchanged=True,pixel_parity_samples_passed=True,
        repeated_source_image_file_reads_during_training=False,
        pixel_buffer_sharing='Linux fork copy-on-write, read-only datasets; no whole-source /dev/shm reservation',
        repeated_CPU_decoding_during_training=mode=='original_encoded_bytes_sharedRAM',
        policy='Identical16GiB decoded-source threshold for baseline,teacher,remaining; source size determines mode')
    data.source_info=info;c.dump(out/(name+'_cache.json'),info);return data,info



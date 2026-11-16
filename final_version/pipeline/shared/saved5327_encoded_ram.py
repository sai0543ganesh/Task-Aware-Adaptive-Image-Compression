"""Cache unchanged original80k encoded bytes; final mapped training pixels stay decodedRAM."""
import hashlib,io,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import simple_grouped_core as c

BASE_CACHE=c.cache

class EncodedRAM:
    def __init__(self,rows,total):
        stat=os.statvfs('/dev/shm');free=stat.f_bavail*stat.f_frsize
        if total>free*.8:raise RuntimeError('Insufficient sharedRAM for original encoded source cache')
        self.rows=rows;self.pixels=torch.empty(total,dtype=torch.uint8).share_memory_()
        self.offsets=[];self.lengths=[];self.shapes=[];self.bytes=total;self.source_info={}
    def __len__(self):return len(self.rows)
    def image(self,i):
        off=self.offsets[i];n=self.lengths[i]
        with Image.open(io.BytesIO(self.pixels[off:off+n].numpy().tobytes())) as im:return im.convert('RGB')
    def rgb(self,i):return np.array(self.image(i),dtype=np.uint8,copy=True)

def load_encoded(row):
    raw=Path(row['path']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest==row['source_sha256'],'Original source content differs from frozen lineage'
    with Image.open(io.BytesIO(raw)) as im:w,h=im.size
    return raw,digest,h,w

def bounded_load(pool, rows):
    for first in range(0, len(rows), 256):
        yield from pool.map(load_encoded, rows[first:first+256])

def cache(rows,out,name,workers=16):
    if name!='remaining80k_source':return BASE_CACHE(rows,out,name,workers)
    start=time.perf_counter();total=sum(int(r['bytes']) for r in rows);data=EncodedRAM(rows,total);cursor=0;fingerprint=hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(raw,digest,h,w) in enumerate(bounded_load(pool, rows)):
            n=len(raw);assert cursor+n<=total
            data.offsets.append(cursor);data.lengths.append(n);data.shapes.append((h,w))
            data.pixels[cursor:cursor+n].copy_(torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()));cursor+=n
            rows[i]['encoded_bytes']=n;fingerprint.update((str(rows[i]['label'])+'|'+rows[i]['path']+'|'+digest+'\n').encode())
            if (i+1)%5000==0:c.say('ORIGINAL_ENCODED_RAM_LOAD',images=i+1,total=len(rows))
    assert cursor==total
    info=dict(images=len(rows),source_payload_bytes=total,encoded_shared_RAM_bytes=total,
        decoded_source_cache=False,encoded_source_cache=True,all_source_file_hashes_verified=True,decoded_source_geometry_bytes=sum(h*w*3 for h,w in data.shapes),original_file_bytes_unchanged=True,seconds=time.perf_counter()-start,
        content_fingerprint=fingerprint.hexdigest(),repeated_image_file_reads_during_training=False,
        decode_cost='source decoded from RAM for features and each branch materialization; all decode times charged',
        final_mapped_training_images_cached_as_decoded_pixels=True)
    data.source_info=info;c.dump(Path(out)/(name+'_cache.json'),info);return data,info

def install():c.cache=cache




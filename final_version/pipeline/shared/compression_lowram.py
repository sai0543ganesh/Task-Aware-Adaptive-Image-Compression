"""Compression-only source lifetime optimization, preserving all output pixels.
Linux only. Destructive consumption applies only to the disposable decoded source
AFTER frozen features and predictions are complete. Source files are never changed.
"""
import ctypes,gc,json,math,mmap,multiprocessing,os,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import simple_grouped_core as c
from stream_mapping import map_selected_streaming as reference

def trim_unused():
    gc.collect();libc=ctypes.CDLL(None,use_errno=True)
    trim=getattr(libc,'malloc_trim',None)
    if trim is not None:trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int;return int(trim(0))
    return None

def checked_source(data):
    assert type(data) is c.RAMImages and os.name=='posix' and Path('/proc/self/maps').exists()
    x=data.pixels
    assert x.device.type=='cpu' and x.dtype==torch.uint8 and x.is_contiguous() and not x.is_shared()
    assert x.storage_offset()==0 and x._base is None and x.numel()==data.bytes
    assert x.untyped_storage().nbytes()==data.bytes and x.untyped_storage().data_ptr()==x.data_ptr()
    assert not multiprocessing.active_children(),'Source workers must finish before releasing source pages'
    assert not getattr(data,'_compression_consumed',False)
    pos=0
    for off,(h,w) in zip(data.offsets,data.shapes):assert off==pos;pos+=h*w*3
    assert pos==data.bytes
    # Only fully owned interior pages of the private anonymous source allocation.
    ptr=x.data_ptr();end=ptr+data.bytes;segments=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split();lo,hi=[int(s,16) for s in parts[0].split('-')]
        if hi>ptr and lo<end:
            assert parts[1]=='rw-p' and (len(parts)<6 or parts[5]=='[heap]'),line
            segments.append((max(lo,ptr),min(hi,end)))
    assert segments and sum(b-a for a,b in segments)==data.bytes
    return ptr

def consume_map(data,tiers,sides,floor,workers=4,ram_factory=None,wave=512,prefault=False):
    assert ram_factory is c.RAMImages and workers==4 and wave in [512,2048,8192]
    ptr=checked_source(data);trim_result=trim_unused();start=time.perf_counter()
    tiers=np.asarray(tiers);floor=np.asarray(floor);n=len(data)
    assert tiers.shape==floor.shape==(n,) and np.issubdtype(tiers.dtype,np.integer) and np.issubdtype(floor.dtype,np.integer)
    assert np.all((tiers>=0)&(tiers<=10)&(floor>=0)&(floor<=10))
    sides=sorted(set(map(int,sides)));assert len(sides)==11
    applied=np.maximum(tiers,floor);shapes=[];reduced=[];offsets=[];total=0
    for i,(h,w) in enumerate(data.shapes):
        q=float(c.TIERS[applied[i]]);hh=max(1,round(h*math.sqrt(q/100)));ww=max(1,round(w*math.sqrt(q/100)))
        allowed=[s for s in sides if s*s<=hh*ww]
        if not allowed:raise ValueError('Smallest frozen canvas exceeds pixel budget')
        side=max(allowed);offsets.append(total);total+=side*side*3;shapes.append((side,side));reduced.append((ww,hh))
    st=os.statvfs('/dev/shm');assert total<st.f_bavail*st.f_frsize*.8
    output=ram_factory.__new__(ram_factory);output.rows=[dict(r) for r in data.rows]
    output.offsets=offsets;output.shapes=shapes;output.bytes=total
    fd,path=tempfile.mkstemp(prefix='mtp-compressed-',dir='/dev/shm');os.close(fd)
    try:
        # Lazy shared RAM mapping: allocate physical pages only as output is written.
        output.pixels=torch.from_file(path,shared=True,size=total,dtype=torch.uint8)
        assert output.pixels.is_shared() and output.pixels.numel()==total
    finally:os.unlink(path)
    packed=output.pixels.numpy();page=os.sysconf('SC_PAGE_SIZE')
    cursor=(ptr+page-1)//page*page;first_page=cursor;released=0;calls=0
    libc=ctypes.CDLL(None,use_errno=True);advise=libc.madvise
    advise.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];advise.restype=ctypes.c_int
    def chunk(first,last):
        for i in range(first,last):
            im=data.image(i)
            if im.mode!='RGB':raise ValueError('Expected RGB')
            if float(c.TIERS[applied[i]])<100:im=im.resize(reduced[i],Image.Resampling.BOX)
            side=shapes[i][0];im=im.resize((side,side),Image.Resampling.BILINEAR)
            a=np.asarray(im,dtype=np.uint8).reshape(-1);off=offsets[i]
            np.copyto(packed[off:off+a.size],a)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0,n,wave):
            stop=min(first+wave,n)
            if prefault:
                out_end=offsets[stop-1]+math.prod(shapes[stop-1])*3
                packed[offsets[first]:out_end].fill(0)
            futures=[pool.submit(chunk,i,min(i+16,stop)) for i in range(first,stop,16)]
            for f in futures:f.result()
            # All reads from this prefix have finished. Never discard future image pages.
            consumed_end=data.offsets[stop-1]+math.prod(data.shapes[stop-1])*3
            end=(ptr+consumed_end)//page*page
            if end>cursor:
                result=advise(ctypes.c_void_p(cursor),end-cursor,mmap.MADV_DONTNEED)
                if result:raise OSError(ctypes.get_errno(),'Could not release consumed source pages')
                released+=end-cursor;calls+=1;cursor=end
    assert released<=data.bytes and cursor<=(ptr+data.bytes)//page*page
    data._compression_consumed=True
    def unavailable(*args,**kwargs):raise RuntimeError('Original pixels were consumed by compression; only metadata remains valid')
    data.rgb=unavailable;data.image=unavailable
    output.source_info=dict(compression_mode=('prefault' if prefault else 'consume')+str(wave),source_pages_released_bytes=released,
      release_calls=calls,source_cache_before_compression_unchanged=True,source_released_only_after_last_read=True,
      lazy_shared_RAM_output=True,source_files_unchanged=True,malloc_trim_result=trim_result,
      compression_seconds_excluding_initial_trim=time.perf_counter()-start)
    return output

def map_mode(mode,data,tiers,sides,floor,workers=4,ram_factory=None):
    if mode=='current':return reference(data,tiers,sides,floor,workers,ram_factory)
    if mode=='trim':
        trim_unused();out=reference(data,tiers,sides,floor,workers,ram_factory)
        out.source_info=dict(compression_mode='trim',source_buffer_preserved=True);return out
    if mode in ['prefault2048','prefault8192']:
        return consume_map(data,tiers,sides,floor,workers,ram_factory,wave=int(mode[8:]),prefault=True)
    if mode in ['consume512','consume2048']:
        return consume_map(data,tiers,sides,floor,workers,ram_factory,wave=int(mode[7:]))
    raise ValueError(mode)

def selected_map(data,tiers,sides,floor,workers=4,ram_factory=None):
    policy=Path(os.environ['MTP_RESEARCH_OUTPUT_ROOT'])/('job_'+os.environ['SLURM_JOB_ID'])/'compression_benchmark/SELECTED_POLICY.json'
    selection=json.loads(policy.read_text());assert selection['accepted'] and selection['pixels_identical']
    return map_mode(selection['mode'],data,tiers,sides,floor,workers,ram_factory)


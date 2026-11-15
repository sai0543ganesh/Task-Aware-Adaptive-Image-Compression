"""Tiny CPU checks of the common encoded loader and exact RGB materialization."""
import hashlib,json,math,pathlib,tempfile
import numpy as np
from PIL import Image
import simple_grouped_core as c
import shared_data_policy as shared

def main():
    c.torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as name:
        root=pathlib.Path(name);rows=[];expected=[]
        rng=np.random.default_rng(11)
        for i,(h,w) in enumerate([(128,192),(224,160),(321,211),(96,96)]):
            p=root/(str(i)+'.jpg');Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8)).save(p)
            raw=p.read_bytes();rows.append(dict(path=str(p),label=i,id=i,bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest()))
            with Image.open(p) as im:expected.append(np.array(im.convert('RGB')))
        src,info=shared.cache_source(rows,root,'test')
        decoded=shared.materialize(src)
        for i,a in enumerate(expected):assert np.array_equal(decoded.rgb(i),a)
        sides=[32,40,48,56,72,96,120,136,160,168,224];tiers=np.array([8,9,10,10]);floor=np.zeros(4,dtype=np.int64)
        mapped=shared.materialize(src,tiers,sides,floor)
        for i,a in enumerate(expected):
            h,w=a.shape[:2];q=float(c.TIERS[tiers[i]])
            ww=max(1,round(w*math.sqrt(q/100)));hh=max(1,round(h*math.sqrt(q/100)))
            im=Image.fromarray(a)
            if q<100:im=im.resize((ww,hh),Image.Resampling.BOX)
            side=max(s for s in sides if s*s<=ww*hh)
            truth=np.asarray(im.resize((side,side),Image.Resampling.BILINEAR))
            assert np.array_equal(mapped.rgb(i),truth)
        assert type(decoded) is type(mapped) is c.RAMImages
        assert not decoded.pixels.is_shared() and not mapped.pixels.is_shared()
        print(json.dumps(dict(passed=True,identity_RGB_equal=True,compressed_RGB_equal=True,same_buffer_class_and_allocator=True)))
if __name__=='__main__':main()

"""Shared native-pixel, RAM-backed ResNet implementation for every arm."""
import gc,hashlib,io,json,math,os,random,resource,shutil,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
from torchvision import models

MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]
TIERS=np.array([.1,.25,.5,1.,5.,7.,10.,15.,60.,90.,100.])
PAD=16
PIXELS=2_000_000
BATCH=64

def say(event,**values):print(json.dumps(dict(event=event,unix_time=time.time(),**values),default=str),flush=True)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def sync():torch.cuda.synchronize()

def decode_one(row):
    raw=Path(row['path']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if row.get('sha256') and digest!=row['sha256']:raise RuntimeError('Evaluation content changed: '+row['path'])
    with Image.open(io.BytesIO(raw)) as im:a=np.array(im.convert('RGB'),dtype=np.uint8,copy=True)
    return a,digest,len(raw)

class RAMImages(Dataset):
    def __init__(self,rows,arrays,source_info=None):
        self.rows=rows;self.offsets=[];self.shapes=[]
        total=sum(a.size for a in arrays)
        available=os.statvfs('/dev/shm');free=available.f_bavail*available.f_frsize
        if total>free*.8:raise RuntimeError(f'Insufficient sharedRAM space: need{total}, available{free}')
        self.pixels=torch.empty(total,dtype=torch.uint8).share_memory_()
        cursor=0
        for i,a in enumerate(arrays):
            h,w,c=a.shape
            if c!=3 or min(h,w)<1:raise RuntimeError('Invalid RGB image geometry')
            self.offsets.append(cursor);self.shapes.append((h,w))
            self.pixels[cursor:cursor+a.size].copy_(torch.from_numpy(a.reshape(-1)))
            cursor+=a.size;arrays[i]=None
        self.bytes=total;self.source_info=source_info or {}
    def __len__(self):return len(self.rows)
    def rgb(self,i):
        h,w=self.shapes[i];off=self.offsets[i]
        return self.pixels[off:off+h*w*3].view(h,w,3).numpy()
    def image(self,i):return Image.fromarray(self.rgb(i))
    def __getitem__(self,i):
        a=torch.from_numpy(self.rgb(i)).permute(2,0,1).float()/255
        return (a-MEAN)/STD,int(self.rows[i]['label']),i

def cache(rows,out,name,workers=16):
    started=time.perf_counter();arrays=[];digests=[];lengths=[];payload=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(a,d,n) in enumerate(pool.map(decode_one,rows,chunksize=16),1):
            arrays.append(a);digests.append(d);lengths.append(n);payload+=n
            if i%5000==0 or i==len(rows):say('RAM_LOAD',split=name,images=i,total=len(rows))
    for row,n in zip(rows,lengths):row['encoded_bytes']=n
    data=RAMImages(rows,arrays)
    identity=hashlib.sha256()
    for row,digest in zip(rows,digests):identity.update((str(row['label'])+'|'+row['path']+'|'+digest+'\n').encode())
    info=dict(images=len(rows),source_payload_bytes=payload,decoded_shared_RAM_bytes=data.bytes,
        seconds=time.perf_counter()-started,content_fingerprint=identity.hexdigest(),
        repeated_image_file_reads_during_training=False)
    data.source_info=info;dump(Path(out)/(name+'_cache.json'),info)
    return data,info

class SubsetRAM(Dataset):
    def __init__(self,source,indices):
        self.source=source;self.indices=list(map(int,indices));self.rows=[source.rows[i] for i in self.indices]
        self.shapes=[source.shapes[i] for i in self.indices]
    def __len__(self):return len(self.indices)
    def rgb(self,i):return self.source.rgb(self.indices[i])
    def image(self,i):return self.source.image(self.indices[i])
    def __getitem__(self,i):
        x,y,_=self.source[self.indices[i]];return x,y,i

class Buckets(Sampler):
    def __init__(self,data,shuffle):
        self.groups=defaultdict(list);self.shuffle=shuffle;self.epoch=0
        for i,(h,w) in enumerate(data.shapes):self.groups[((h+15)//16*16,(w+15)//16*16)].append(i)
        self.length=sum(math.ceil(len(ix)/self.cap(shape)) for shape,ix in self.groups.items())
    @staticmethod
    def cap(shape):return max(1,min(BATCH,PIXELS//math.prod(shape)))
    def __len__(self):return self.length
    def __iter__(self):
        rng=random.Random(42+self.epoch);batches=[]
        for shape,ix in self.groups.items():
            ix=ix.copy()
            if self.shuffle:rng.shuffle(ix)
            size=self.cap(shape);batches.extend(ix[j:j+size] for j in range(0,len(ix),size))
        if self.shuffle:rng.shuffle(batches)
        yield from batches

def collate(samples):
    keys={((x.shape[-2]+15)//16*16,(x.shape[-1]+15)//16*16) for x,_,_ in samples}
    if len(keys)!=1:raise RuntimeError('Mixed canvas buckets')
    h,w=next(iter(keys));result=torch.zeros(len(samples),3,h,w)
    for j,(x,_,_) in enumerate(samples):result[j,:,:x.shape[-2],:x.shape[-1]]=x
    return result,torch.tensor([y for _,y,_ in samples]),torch.tensor([i for _,_,i in samples])

def loader(data,training=False):
    sampler=Buckets(data,training)
    return DataLoader(data,batch_sampler=sampler,collate_fn=collate,num_workers=4,
        pin_memory=True,persistent_workers=True,prefetch_factor=2),sampler

def fresh_model(classes):
    seed();net=models.resnet18(weights=None,num_classes=len(classes)).cuda();return net

def evaluate(net,data,prediction_path=None,probabilities=False):
    before=time.perf_counter();stream,_=loader(data);net.eval()
    p=np.empty((len(data),101),np.float32);loss=0.;correct=correct5=0
    with torch.inference_mode():
        for x,y,indices in stream:
            y=y.cuda(non_blocking=True);logits=net(x.cuda(non_blocking=True))
            probability=logits.softmax(1);guess=logits.argmax(1);ix=indices.numpy()
            p[ix]=probability.cpu().numpy();correct+=int((guess==y).sum())
            correct5+=int((logits.topk(5,1).indices==y[:,None]).any(1).sum())
            loss+=float(nn.functional.cross_entropy(logits,y,reduction='sum'))
    sync();del stream
    if not np.isfinite(p).all() or not math.isfinite(loss):raise RuntimeError('Nonfinite ResNet evaluation values')
    metrics=dict(images=len(data),top1_percent=100*correct/len(data),top5_percent=100*correct5/len(data),
        correct_top1=correct,cross_entropy=loss/len(data),seconds=time.perf_counter()-before)
    if prediction_path:np.savez_compressed(prediction_path,p=p,labels=np.array([int(r['label']) for r in data.rows]))
    return (metrics,p) if probabilities else metrics

def geometry(data):
    sampler=Buckets(data,False);native=sum(h*w for h,w in data.shapes)
    padded=sum(((h+15)//16*16)*((w+15)//16*16) for h,w in data.shapes)
    short=np.array([min(s) for s in data.shapes])
    return dict(images=len(data),bucket_count=len(sampler.groups),base_microbatches=len(sampler),
        short_percentiles=np.percentile(short,[0,25,50,75,100]).tolist(),
        short_under32=int((short<32).sum()),short_under64=int((short<64).sum()),
        native_pixels=native,padded_pixels=padded,padding_overhead_percent=100*(padded/native-1))

class GuardRejected(Exception):pass

def train(net,data,val,classes,epochs,lr,out,phase,budget=None,prior_seconds=0.):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);local=Path(os.environ['TMPDIR'])/phase
    local.mkdir(parents=True,exist_ok=True)
    stream,sampler=loader(data,True);optimizer=torch.optim.SGD(net.parameters(),lr=lr,momentum=.9,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    start=time.perf_counter();history=[];best=-1.;best5=-1.;best5_epoch=None
    if any(layer.momentum!=.1 for layer in net.modules() if isinstance(layer,nn.BatchNorm2d)):
        raise RuntimeError('This protocol requires unchanged standard BatchNorm momentum0.1')
    for epoch in range(1,epochs+1):
        sampler.epoch=epoch;net.train();optimizer.zero_grad(set_to_none=True);sync();before=time.perf_counter()
        total=steps=calls=extra=0;loss_total=torch.zeros((),device='cuda');good=torch.zeros((),device='cuda')
        data_wait=0.;mark=time.perf_counter()
        for xx,yy,_ in stream:
            data_wait+=time.perf_counter()-mark;count=len(yy)
            x=xx.cuda(non_blocking=True);y=yy.cuda(non_blocking=True)
            # Standard BN needs >1 value/channel. An exceptionally tiny singleton
            # is duplicated within this forward only; no pixels or BN rules change.
            degenerate=count==1 and math.ceil(x.shape[-2]/32)*math.ceil(x.shape[-1]/32)==1
            if degenerate:x=x.repeat(2,1,1,1);y=y.repeat(2);extra+=1
            optimizer.zero_grad(set_to_none=True)
            logits=net(x);loss=nn.functional.cross_entropy(logits,y)
            loss.backward();optimizer.step();steps+=1;calls+=1
            loss_total+=loss.detach()*count;good+=(logits[:count].argmax(1)==y[:count]).sum();total+=count
            if budget and prior_seconds+time.perf_counter()-start>=budget:raise GuardRejected('Actual proposed budget exhausted')
            mark=time.perf_counter()
        sync();training_seconds=time.perf_counter()-before;scheduler.step()
        if total!=len(data):raise RuntimeError('Epoch omitted images')
        item=dict(epoch=epoch,images=total,train_top1_percent=100*float(good)/total,loss=float(loss_total)/total,
            training_seconds=training_seconds,data_iterator_wait_seconds=data_wait,forward_backward_calls=calls,
            optimizer_steps=steps,extra_duplicate_forward_images=extra,actual_forward_images=total+extra,
            gradient_accumulation=False,maximum_batch_images=BATCH,learning_rate_next_epoch=scheduler.get_last_lr()[0])
        if not math.isfinite(item['loss']):raise RuntimeError('Nonfinite training loss; rejecting corrupt run')
        if epoch<=5 or epoch%5==0 or epoch==epochs:
            item['validation']=evaluate(net,val)
            checkpoint=dict(model_state_dict=net.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),epoch=epoch,classes=classes,input_policy='native pixels,16pixel padding',
                normalization='standard BatchNorm momentum0.1',gradient_accumulation=False,
                actual_batch_updates=True,rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state())
            save_start=time.perf_counter();torch.save(checkpoint,local/'latest.pth')
            shutil.copyfile(local/'latest.pth',out/'latest.tmp');os.replace(out/'latest.tmp',out/'latest.pth')
            if item['validation']['top1_percent']>best:
                best=item['validation']['top1_percent'];shutil.copyfile(local/'latest.pth',out/'best.tmp');os.replace(out/'best.tmp',out/'best.pth')
            if item['validation']['top5_percent']>best5:
                best5=item['validation']['top5_percent'];best5_epoch=epoch
                shutil.copyfile(local/'latest.pth',out/'best_top5.tmp');os.replace(out/'best_top5.tmp',out/'best_top5.pth')
            item['checkpoint_export_seconds']=time.perf_counter()-save_start
        history.append(item);dump(out/'history.json',history);say('RESNET_EPOCH',phase=phase,**item)
        if budget and epoch>=4:
            recent=np.mean([r['training_seconds'] for r in history[1:][-3:]])
            elapsed=prior_seconds+time.perf_counter()-start
            estimate=elapsed+recent*(epochs-epoch)
            dump(out/'time_guard.json',dict(epoch=epoch,actual_elapsed=elapsed,estimate_only=True,projected_minimum=estimate,budget=budget))
            if estimate>=budget:raise GuardRejected('Projected training alone exceeds complete baseline budget')
    saved=torch.load(out/'best.pth',map_location='cpu',weights_only=True);net.load_state_dict(saved['model_state_dict'])
    del stream;gc.collect()
    return dict(epochs=epochs,images=len(data),presentations=len(data)*epochs,selected_epoch=saved['epoch'],
        actual_forward_presentations=sum(r['actual_forward_images'] for r in history),
        extra_duplicate_forward_images=sum(r['extra_duplicate_forward_images'] for r in history),
        best_validation_percent=best,best_validation_top5_percent=best5,best_validation_top5_epoch=best5_epoch,
        best_top5_checkpoint=str(out/'best_top5.pth'),seconds=time.perf_counter()-start,checkpoint=str(out/'best.pth'))

def floor_indices(shapes,minimum=96):
    floor=[]
    for h,w in shapes:
        required=min(h,w,minimum);allowed=[]
        for q in TIERS:
            dims=(max(1,round(h*math.sqrt(q/100))),max(1,round(w*math.sqrt(q/100))))
            allowed.append(min(dims)>=required)
        floor.append(int(np.argmax(allowed)))
    return np.array(floor,dtype=np.int64)

def reduced(data,indices,tier,out=None):
    arrays=[];rows=[];payload=0
    for j,i in enumerate(indices):
        q=float(TIERS[int(tier[j])]);im=data.image(int(i));w,h=im.size
        if q<100:im=im.resize((max(1,round(w*math.sqrt(q/100))),max(1,round(h*math.sqrt(q/100)))),Image.Resampling.BOX)
        a=np.array(im,dtype=np.uint8,copy=True);arrays.append(a);rows.append(dict(data.rows[int(i)]))
        payload+=a.size
    return RAMImages(rows,arrays)




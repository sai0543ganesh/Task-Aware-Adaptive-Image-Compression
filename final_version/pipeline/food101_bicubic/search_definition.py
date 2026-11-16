"""Fixed-label original-20k MLP search. Holdout is never used for selection."""
import time
START=time.perf_counter()
import csv,gc,hashlib,json,math,os,random,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler,QuantileTransformer
import simple_grouped_core as c

ROOT=Path(os.environ.get('MTP_PROJECT_ROOT', Path.cwd()))
PARENT=ROOT/'results/original_full_v1/job_5327'
UP=PARENT/'upstream'
OUT=ROOT/'results/original_mlp_optimize_v2'/('job_'+os.environ['SLURM_JOB_ID'])
SEEDS=[42,43,44]

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def metrics(y,p):
    d=p-y;cm=np.zeros((11,11),int);np.add.at(cm,(y,p),1)
    return dict(images=len(y),exact_percent=100*float((d==0).mean()),within_one_percent=100*float((abs(d)<=1).mean()),
        unsafe_at_least_two_tiers_percent=100*float((d<=-2).mean()),
        macro_recall_percent=100*float(np.mean([(p[y==i]==i).mean() for i in np.unique(y)])),
        true_counts=np.bincount(y,minlength=11).tolist(),predicted_counts=np.bincount(p,minlength=11).tolist(),confusion=cm.tolist())

def geometry_features(x,names,sides):
    h=x[:,names.index('original_height')];w=x[:,names.index('original_width')];rows=[]
    for hh,ww in zip(h,w):
        values=[]
        for q in c.TIERS:
            nh=max(1,round(float(hh)*math.sqrt(q/100)));nw=max(1,round(float(ww)*math.sqrt(q/100)))
            choices=[s for s in sides if s*s<=nh*nw]
            values.append(math.log(max(choices) if choices else min(sides)))
        rows.append(values+[math.log(max(hh,ww)/max(1,min(hh,ww))),math.log(max(1,hh*ww)),math.log(max(1,min(hh,ww)))])
    return np.asarray(rows,np.float32),['mapped_log_side_'+str(i) for i in range(11)]+['source_log_aspect','source_log_area','source_log_short']

class Block(nn.Module):
    def __init__(self,width,drop):
        super().__init__();self.body=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Dropout(drop),nn.Linear(width*2,width),nn.Dropout(drop))
    def forward(self,x):return x+self.body(x)

class Net(nn.Module):
    def __init__(self,n,cfg):
        super().__init__();self.periodic=cfg.get('periodic',False)
        self.register_buffer('frequencies',torch.tensor([.5,1.,2.]))
        self.entry=nn.Sequential(nn.Linear(n*(7 if self.periodic else 1),cfg['width']),nn.LayerNorm(cfg['width']),nn.GELU())
        self.blocks=nn.Sequential(*[Block(cfg['width'],cfg['drop']) for _ in range(cfg['depth'])])
        self.final=nn.LayerNorm(cfg['width']);self.tier=nn.Linear(cfg['width'],11)
        self.semantic=nn.Linear(cfg['width'],101);self.proxy=nn.Linear(cfg['width'],36)
    def forward(self,x,floor):
        if self.periodic:
            phase=x[:,:,None]*self.frequencies[None,None,:]
            x=torch.cat([x,phase.sin().flatten(1),phase.cos().flatten(1)],1)
        z=self.final(self.blocks(self.entry(x)));logits=self.tier(z)
        logits=logits.masked_fill(torch.arange(11,device=x.device)[None,:]<floor[:,None],-1e4)
        return logits,self.semantic(z),self.proxy(z)

def preprocess(x,y,fit,cfg,names):
    imp=SimpleImputer(strategy='median',keep_empty_features=True);a=imp.fit_transform(x[fit]);keep=[];seen=set()
    for j in range(a.shape[1]):
        if np.std(a[:,j])<=1e-10:continue
        digest=hashlib.sha256(np.ascontiguousarray(a[:,j]).tobytes()).digest()
        if digest not in seen:seen.add(digest);keep.append(j)
    if cfg['k'] and len(keep)>cfg['k']:
        scores,_=f_classif(a[:,keep],y[fit]);order=np.argsort(-np.nan_to_num(scores,nan=-1.,posinf=1e30))[:cfg['k']]
        keep=[keep[i] for i in order]
    if cfg['transform']=='quantile':scaler=QuantileTransformer(n_quantiles=min(256,len(fit)),output_distribution='normal',random_state=42)
    else:scaler=StandardScaler()
    scaler.fit(a[:,keep]);z=np.clip(scaler.transform(imp.transform(x)[:,keep]),-8,8).astype(np.float32)
    return z,dict(imputer=imp,scaler=scaler,keep=keep,feature_columns=[names[i] for i in keep],input_columns=names)

def probabilities(net,x,floor):
    net.eval();out=[]
    with torch.inference_mode():
        for first in range(0,len(x),2048):out.append(net(x[first:first+2048],floor[first:first+2048])[0].softmax(1).cpu().numpy())
    return np.concatenate(out)

def fit(z,y,floor,soft,proxy,train,valid,cfg,s,fixed=None):
    seed(s);net=Net(z.shape[1],cfg).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=72)
    tx=torch.tensor(z[train],device='cuda');ty=torch.tensor(y[train],device='cuda');tf=torch.tensor(floor[train],device='cuda')
    sx=torch.tensor(soft[train],device='cuda');pm=proxy[train].mean(0);ps=proxy[train].std(0).clip(.02)
    px=torch.tensor((proxy[train]-pm)/ps,device='cuda')
    vx=torch.tensor(z[valid],device='cuda');vf=torch.tensor(floor[valid],device='cuda')
    best=-1e9;state=None;bestep=1;stale=0;trace=[]
    for ep in range(1,(fixed or 72)+1):
        net.train()
        for ix in torch.randperm(len(train),device='cuda').split(256):
            logits,sem,pro=net(tx[ix]+cfg['noise']*torch.randn_like(tx[ix]),tf[ix]);p=logits.softmax(1);target=ty[ix]
            delta=torch.arange(11,device='cuda')[None,:]-target[:,None]
            loss=nn.functional.cross_entropy(logits,target)
            loss+=cfg['ordinal']*nn.functional.mse_loss(p.cumsum(1),(delta>=0).float())
            loss-=cfg['near']*min(1.,ep/12)*(p*(delta.abs()<=1)).sum(1).clamp_min(1e-8).log().mean()
            if cfg['aux']:
                loss+=cfg['aux']*4*nn.functional.kl_div((sem/2).log_softmax(1),sx[ix],reduction='batchmean')
                loss+=.05*nn.functional.smooth_l1_loss(pro,px[ix])
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        scheduler.step()
        if fixed:continue
        if ep%3==0:
            m=metrics(y[valid],probabilities(net,vx,vf).argmax(1));score=m['exact_percent']+.15*m['within_one_percent']
            trace.append(dict(epoch=ep,metrics=m,score=score))
            if score>best:
                best=score;bestep=ep;stale=0;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            else:stale+=3
            if ep>=24 and stale>=15:break
    if not fixed:net.load_state_dict(state)
    else:bestep=fixed
    return net,bestep,trace

# Frozen model and preprocessing utilities only; confirmation does not rerun model search.




# Generated from reviewed local project definitions; no legacy paths or model weights.
import hashlib, math
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as fn
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, QuantileTransformer
SEED = 42


def safe_float(x):
    x = float(x)
    return x if math.isfinite(x) else 0.0

def moments(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0, 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return 0.0, 0.0
    z = (x - mu) / sd
    return safe_float(np.mean(z ** 3)), safe_float(np.mean(z ** 4))

def corr_horizontal(gray):
    a = gray[:, :-1].ravel().astype(np.float64)
    b = gray[:, 1:].ravel().astype(np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return safe_float(np.corrcoef(a, b)[0, 1])

def stats(im):
    width,height=im.size;scale=min(1.,256/max(width,height))
    a=im.resize((max(2,round(width*scale)),max(2,round(height*scale))),Image.Resampling.LANCZOS) if scale<1 else im
    rgb=np.asarray(a,dtype=np.float32)/255.;r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    gray=.299*r+.587*g+.114*b;h,w=gray.shape
    mx=np.maximum(np.maximum(r,g),b);mn=np.minimum(np.minimum(r,g),b)
    saturation=np.where(mx>1e-8,(mx-mn)/np.maximum(mx,1e-8),0.)
    gx=np.zeros_like(gray);gy=np.zeros_like(gray)
    gx[:,1:-1]=(gray[:,2:]-gray[:,:-2])*.5;gy[1:-1]=(gray[2:]-gray[:-2])*.5
    grad=np.sqrt(gx*gx+gy*gy)
    lap=np.zeros_like(gray);lap[1:-1,1:-1]=gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]-4*gray[1:-1,1:-1]
    padded=np.pad(gray,1,mode='reflect');local=np.zeros_like(gray);squares=np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            part=padded[dy:dy+h,dx:dx+w];local+=part;squares+=part*part
    mean=local/9.;std=np.sqrt(np.maximum(squares/9.-mean*mean,0.))
    skew,kurt=moments(grad);rg=r-g;yb=.5*(r+g)-b
    colorful=math.sqrt(float(np.var(rg))+float(np.var(yb)))+.3*math.sqrt(float(np.mean(rg))**2+float(np.mean(yb))**2)
    vertical=abs(gray[:,8::8]-gray[:,7:-1:8]);horizontal=abs(gray[8::8]-gray[7:-1:8])
    return dict(width=width,height=height,aspect_ratio=width/max(height,1),pixel_area=width*height,log_pixel_area=math.log1p(width*height),short_side=min(width,height),long_side=max(width,height),log_short_side=math.log1p(min(width,height)),saturation_high_fraction=float(np.mean(saturation>.7)),S_mean=float(saturation.mean()),gradient_skewness=skew,gradient_kurtosis=kurt,texture_mean=float(std.mean()),laplacian_abs_mean=float(abs(lap).mean()),gradient_p90=float(np.percentile(grad,90)),jpeg_horizontal_block_diff=float(horizontal.mean()) if horizontal.size else 0.,jpeg_vertical_block_diff=float(vertical.mean()) if vertical.size else 0.,edge_density=float(np.mean(grad>.1)),colorfulness=colorful,laplacian_variance=float(np.var(lap)),tenengrad_mean=float(np.mean(grad*grad)),spatial_corr_horizontal=corr_horizontal(gray),gradient_std=float(grad.std()))

def attention_stats(raw,gray,short,tag):
    batch,height,width=raw.shape
    maximum=raw.amax((1,2));usable=maximum>1e-8
    a=raw/maximum[:,None,None].clamp_min(1e-8)
    mass=a.sum((1,2)).clamp_min(1e-8);p=a/mass[:,None,None]
    yy=torch.linspace(0,1,height,device=raw.device)[None,:,None]
    xx=torch.linspace(0,1,width,device=raw.device)[None,None,:]
    cx=(p*xx).sum((1,2));cy=(p*yy).sum((1,2))
    sx=torch.sqrt((p*(xx-cx[:,None,None])**2).sum((1,2)).clamp_min(0))
    sy=torch.sqrt((p*(yy-cy[:,None,None])**2).sum((1,2)).clamp_min(0))
    border=torch.zeros_like(a);border[:,0,:]=1;border[:,-1,:]=1;border[:,:,0]=1;border[:,:,-1]=1
    support=a>.25;xs=support.any(1);ys=support.any(2)
    bw=(width-xs.flip(1).float().argmax(1)-xs.float().argmax(1)).float()/width
    bh=(height-ys.flip(1).float().argmax(1)-ys.float().argmax(1)).float()/height
    bw*=usable;bh*=usable
    area=a.mean((1,2));entropy=-(p*p.clamp_min(1e-12).log()).sum((1,2))/math.log(height*width)
    entries=[('usable',usable.float()),('soft_area',area),('entropy',entropy),('concentration',p.square().sum((1,2))*height*width),('center_x',cx),('center_y',cy),('spread_x',sx),('spread_y',sy),('border_mass',(p*border).sum((1,2))),('bbox_width',bw),('bbox_height',bh)]
    for threshold in [.1,.25,.5,.75]:entries.append((f'fraction_{threshold:g}',(a>threshold).float().mean((1,2))))
    edge_by_scale={}
    for side in [64,128]:
        g=fn.interpolate(gray[:,None],size=(side,side),mode='bilinear',align_corners=False)[:,0]
        weight=fn.interpolate(a[:,None],size=(side,side),mode='bilinear',align_corners=False)[:,0]
        dx=torch.zeros_like(g);dy=torch.zeros_like(g)
        dx[:,:,1:-1]=(g[:,:,2:]-g[:,:,:-2])*.5;dy[:,1:-1,:]=(g[:,2:,:]-g[:,:-2,:])*.5
        gradient=torch.sqrt(dx.square()+dy.square())
        denominator=weight.sum((1,2)).clamp_min(1e-8)
        object_mean=(g*weight).sum((1,2))/denominator
        contrast=torch.sqrt(((g-object_mean[:,None,None]).square()*weight).sum((1,2))/denominator)
        edge=(gradient*weight).sum((1,2))/denominator
        background=(gradient*(1-weight)).sum((1,2))/(1-weight).sum((1,2)).clamp_min(1e-8)
        edge_by_scale[side]=edge
        entries.extend([(f'contrast{side}',contrast),(f'edge{side}',edge),(f'background_edge{side}',background)])
    entries.extend([('edge_scale_ratio',edge_by_scale[64]/edge_by_scale[128].clamp_min(1e-5)),('native_log_object_short',(.875*short).log()+.5*area.clamp_min(.01).log()),('native_log_object_area',2*(.875*short).log()+area.clamp_min(.01).log())])
    return torch.stack([value for _,value in entries],1),[f'cam_{tag}_{name}' for name,_ in entries]

class Net(nn.Module):
    def __init__(self,n,cfg,mu,scale,pixel):
        super().__init__();self.pixel=pixel;self.strength=cfg['residual']
        if pixel:self.pixel_net=nn.Sequential(nn.Linear(3072,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,24),nn.LayerNorm(24),nn.Tanh())
        self.body=nn.Sequential(nn.Linear(n+(24 if pixel else 0),384),nn.LayerNorm(384),nn.GELU(),nn.Dropout(.25),nn.Linear(384,192),nn.LayerNorm(192),nn.GELU(),nn.Dropout(.15),nn.Linear(192,128),nn.GELU())
        self.semantic=nn.Linear(128,100);self.offset=nn.Linear(128,2);self.residual=nn.Linear(128,11);self.proxy=nn.Linear(128,36)
        self.mu=nn.Parameter(torch.tensor(mu,dtype=torch.float32));self.scale=nn.Parameter(torch.tensor(np.log(np.expm1(scale-.05)),dtype=torch.float32))
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias);nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
    def forward(self,x,bounds,pixels=None):
        if self.pixel:x=torch.cat([x,self.pixel_net(pixels)],1)
        h=self.body(x);sem=self.semantic(h);off=self.offset(h)
        mu=self.mu[None,:]+.2*off[:,:1];scale=.05+nn.functional.softplus(self.scale[None,:]+.1*off[:,1:2])
        cdf=(torch.sigmoid((bounds[:,:,None]-mu[:,None,:])/scale[:,None,:])*sem.softmax(1)[:,None,:]).sum(2)
        p=torch.cat([cdf[:,:1],cdf[:,1:]-cdf[:,:-1],1-cdf[:,-1:]],1).clamp_min(1e-8)
        return p.log()+self.strength*self.residual(h),sem,self.proxy(h)

def fit_preprocess(F,y,tr,cols,cfg):
 imp=SimpleImputer(strategy='median',keep_empty_features=True);a=imp.fit_transform(F.iloc[tr][cols]);eligible=np.flatnonzero(np.std(a,axis=0)>1e-10);seen=set();kept=[]
 for j in eligible:
  digest=hashlib.sha256(np.ascontiguousarray(a[:,j]).tobytes()).digest()
  if digest not in seen:seen.add(digest);kept.append(j)
 keep=np.asarray(kept,dtype=int);a=a[:,keep];chosen=[cols[i] for i in keep]
 if cfg['k'] and cfg['k']<a.shape[1]:
  s,_=f_classif(a,y[tr]);sel=np.argsort(-np.nan_to_num(s,nan=-1.,posinf=1e30))[:cfg['k']];a=a[:,sel];chosen=[chosen[j] for j in sel]
 else:sel=np.arange(a.shape[1])
 state={'imputer':imp,'input_columns':cols,'keep':keep,'selected':sel,'feature_columns':chosen,'kind':cfg['transform']}
 if cfg['transform']=='standard':state['scaler']=StandardScaler().fit(a)
 elif cfg['transform']=='quantile':state['scaler']=QuantileTransformer(n_quantiles=min(256,len(tr)),output_distribution='normal',subsample=max(10000,len(tr)),random_state=SEED).fit(a)
 else:state['knots']=[np.unique(np.quantile(a[:,j],np.linspace(0,1,17))) for j in range(a.shape[1])]
 return state

def transform(F,idx,state):
 a=state['imputer'].transform(F.iloc[idx][state['input_columns']])[:,state['keep']][:,state['selected']]
 if state['kind']!='piecewise':return np.clip(state['scaler'].transform(a),-8,8).astype('float32')
 out=[]
 for j,k in enumerate(state['knots']):
  z=np.clip((a[:,j,None]-k[:-1][None,:])/np.maximum(np.diff(k),1e-10)[None,:],0,1);out.append(z)
 return np.concatenate(out,axis=1).astype('float32')



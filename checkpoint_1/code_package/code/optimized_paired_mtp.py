"""Original-first, fresh paired experiments; accurate clocks and early time guards.

One Slurm allocation, two existing source versions, common disjoint validation/test.
Frozen MobileNet features are optional; all task ResNets and tier MLPs are fresh.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import resource
import shutil
import subprocess
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

import optimized_mtp_helpers as helper

ROOT = Path('/userhome/mtech/s.chatharasupalli/imagenet100_mtp')
OLD = ROOT / 'results/recreated_20261018/current11_endtoend'
TIERS = np.array([.1, .25, .5, 1., 5., 7., 10., 15., 60., 90., 100.])
CUTS = TIERS[:-1]
SEED = 42
CONFIG = dict(baseline_epochs=50, teacher_epochs=20, downstream_epochs=45,
              batch_size=128, workers=8, learning_rate=.1, downstream_lr=.05,
              jpeg_quality=95, jpeg_subsampling=0, promotion=1,
              max_accuracy_drop_points=3., minimum_native_short=128,
              teacher_policy_drop_limit=.015, precision='FP32',
              early_time_guard=True, max_downstream_attempts=2)


def say(event, **values):
    print(json.dumps(dict(event=event, unix_time=time.time(), **values), default=str), flush=True)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def read_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def sync():
    torch.cuda.synchronize()


def eval_transform():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def train_transform():
    return transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                               transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def prepare(run):
    """Inventory is shared input preparation, recorded outside each measured arm."""
    started = time.perf_counter()
    original = read_rows(OLD / 'original_manifest_105000.csv')
    repaired = read_rows(OLD / 'manifest_105000.csv')
    ready = json.loads((OLD / 'MANIFEST_READY.json').read_text())
    if sha(OLD / 'manifest_105000.csv') != ready['sha256']:
        raise RuntimeError('Audited test manifest has changed')
    classes = sorted({r['class_name'] for r in original})
    if len(classes) != 100:
        raise RuntimeError('Expected 100 image categories')
    source = [r for r in original if r['part'] != 'test']
    test = [dict(r) for r in repaired if r['part'] == 'test']
    if Counter(r['part'] for r in source) != {'teacher': 20000, 'remaining': 80000} or len(test) != 5000:
        raise RuntimeError('Source partition sizes changed')
    for part,expected in [('teacher',200),('remaining',800)]:
        counts=Counter(r['class_name'] for r in source if r['part']==part)
        if len(counts)!=100 or any(n!=expected for n in counts.values()):
            raise RuntimeError('Source category balance changed: '+part)
    if len({r['sha256'] for r in test})!=5000 or Counter(r['class_name'] for r in test)!=dict.fromkeys(classes,50):
        raise RuntimeError('Primary evaluation identities/class counts changed')
    forbidden = {r['sha256'] for r in original} | {r['sha256'] for r in repaired}
    source_hashes = {r['sha256'] for r in source}
    if source_hashes & {r['sha256'] for r in test}:
        raise RuntimeError('Primary test overlaps original source pool')
    validation = []
    # Ten additional unused images/category. Neither teacher nor MLP sees these.
    for label, category in enumerate(classes):
        pool = sorted((ROOT / 'data/extracted' / category).glob('*'),
                      key=lambda p: hashlib.sha256(('mtp-optimization-dev-v1:' + p.name).encode()).digest())
        selected = 0
        for path in pool:
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'} or not path.is_file():
                continue
            digest = sha(path)
            if digest in forbidden:
                continue
            forbidden.add(digest)
            validation.append(dict(path=str(path), label=label, class_name=category, sha256=digest))
            selected += 1
            if selected == 10:
                break
        if selected != 10:
            raise RuntimeError('Insufficient independent validation images in ' + category)
    if any(classes[int(r['label'])]!=r['class_name'] for r in source+test+validation):
        raise RuntimeError('Image category order differs between training and evaluation manifests')
    random20 = read_rows(ROOT / 'results/recreated_20261018/mlp/randomized_source_v2/manifest.csv')
    random80 = read_rows(ROOT / 'results/recreated_20261021/proposed80k_v28_resnet5212/randomized_source80k_manifest.csv')
    maps = {'teacher': {r['image_path']: str(ROOT / 'data/randomized_source_v2' / r['image_path']) for r in random20},
            'remaining': {r['image_path']: r['source_path'] for r in random80}}
    if len(maps['teacher']) != 20000 or len(maps['remaining']) != 80000:
        raise RuntimeError('Randomized source manifests contain missing/duplicate identities')
    original_lookup={(r['part'],r['relative_path']):r['path'] for r in source}
    if any(r['original_path'] != original_lookup[('remaining',r['image_path'])] for r in random80):
        raise RuntimeError('Randomized 80k source lineage differs from the original 80k')
    inventories = {}
    for domain in ['original', 'randomized']:
        rows = []
        for r in source:
            path = r['path'] if domain == 'original' else maps[r['part']][r['relative_path']]
            rows.append(dict(id=r['id'], part=r['part'], relative_path=r['relative_path'], path=path,
                             original_path=r['path'], source_sha256=r['sha256'], label=int(r['label']), class_name=r['class_name']))
        with ThreadPoolExecutor(8) as pool:
            sizes = list(pool.map(lambda r: Path(r['path']).stat().st_size, rows))
        for r, size in zip(rows, sizes):
            r['bytes'] = size
        inventories[domain] = dict(images=len(rows), teacher_images=20000, remaining_images=80000,
                                   storage_bytes=sum(sizes), source_identity_changes=0)
        write_rows(run / (domain + '_sources.csv'), rows)
    for split, rows in [('test', test), ('validation', validation)]:
        with ThreadPoolExecutor(8) as pool:
            hashes = list(pool.map(lambda r: sha(r['path']), rows))
        if any(digest != row['sha256'] for row, digest in zip(rows, hashes)):
            raise RuntimeError('Audited evaluation image changed: ' + split)
        write_rows(run / (split + '.csv'), rows)
    columns = pd.read_csv(Path(__file__).with_name('optimized_v28_features.csv')).feature.tolist()
    if len(columns) != 215 or any('class_id' in c for c in columns):
        raise RuntimeError('Unexpected feature candidate schema')
    weights = Path(torch.hub.get_dir()) / 'checkpoints/mobilenet_v3_small-047dcff4.pth'
    if not weights.is_file():
        raise FileNotFoundError('Frozen optional MobileNet weights must already be cached')
    report = dict(config=CONFIG, classes=classes, tiers=TIERS.tolist(), data=inventories,
                  test_images=5000, validation_images=1000,
                  test_manifest_sha256=sha(run / 'test.csv'), validation_manifest_sha256=sha(run / 'validation.csv'),
                  common_inventory_and_integrity_seconds=time.perf_counter() - started,
                  training_sources_replaced=0, repeated_randomized_bank_generation=False,
                  source_disjointness='Fresh validation/test hashes compared with saved original training-content hashes',
                  class_id_input=False, source_features_only=True,
                  pretrained_encoder='Optional frozen MobileNetV3-Small ImageNet1K; not a pretrained task ResNet',
                  baseline_presentations=5000000, planned_proposed_presentations=4000000,
                  randomized_initial_retention_counts_20k=dict(Counter(r['initial_pixel_retention_pct'] for r in random20)),
                  randomized_initial_retention_counts_80k=dict(Counter(r['initial_pixel_retention_pct'] for r in random80)),
                  speed_interpretation='Schedule uses 20% fewer training presentations; speed is not attributable to compression alone')
    dump(run / 'protocol.json', report)
    say('COMMON_INPUTS_VERIFIED', **report)


class Images(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        self.transform = train_transform() if training else eval_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            value = self.transform(image.convert('RGB'))
        return value, int(row['label'])


def loader(rows, training=False, batch=None):
    return DataLoader(Images(rows, training), batch_size=batch or CONFIG['batch_size'], shuffle=training,
                      num_workers=CONFIG['workers'], pin_memory=True, persistent_workers=True,
                      generator=torch.Generator().manual_seed(SEED))


def fresh_resnet():
    seed()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model.cuda()


def evaluate(model, rows, predictions_path=None):
    started = time.perf_counter()
    data = loader(rows)
    model.eval()
    correct = torch.zeros((),dtype=torch.int64,device='cuda')
    correct5 = torch.zeros((),dtype=torch.int64,device='cuda')
    total = 0
    total_loss=torch.zeros((),device='cuda')
    class_correct=torch.zeros(100,dtype=torch.int64,device='cuda')
    class_count=torch.zeros(100,dtype=torch.int64,device='cuda')
    predictions=[]
    with torch.inference_mode():
        for images, labels in data:
            labels = labels.cuda(non_blocking=True)
            logits = model(images.cuda(non_blocking=True))
            guess=logits.argmax(1)
            good=guess==labels
            correct += good.sum()
            correct5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum()
            total_loss+=nn.functional.cross_entropy(logits,labels,reduction='sum')
            class_count+=torch.bincount(labels,minlength=100)
            class_correct+=torch.bincount(labels[good],minlength=100)
            if predictions_path is not None:
                for label,prediction in zip(labels.cpu().tolist(),guess.cpu().tolist()):
                    predictions.append(dict(path=rows[len(predictions)]['path'],label=label,prediction=prediction,
                                            correct=int(label==prediction)))
            total += len(labels)
    sync()
    del data
    if predictions_path is not None:
        write_rows(predictions_path,predictions)
    return dict(images=total, correct_top1=int(correct), top1_percent=100 * int(correct) / total,
                top5_percent=100 * int(correct5) / total,cross_entropy_loss=float(total_loss)/total,
                correct_by_class=class_correct.cpu().tolist(),count_by_class=class_count.cpu().tolist(),
                seconds=time.perf_counter() - started)


class RejectedConfiguration(Exception):
    pass


def train_resnet(model, rows, validation, epochs, lr, output, phase, proposed_start=None, baseline_seconds=None):
    output.mkdir(parents=True, exist_ok=False)
    data = loader(rows, True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: min(1., (e + 1) / 2) * .5 * (1 + math.cos(math.pi * min(e, epochs) / epochs)))
    history = []
    best = -1.
    training_start = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        sync()
        started = time.perf_counter()
        loss_sum = torch.zeros((),device='cuda')
        correct = torch.zeros((),dtype=torch.int64,device='cuda')
        total = 0
        model.train()
        for images, labels in data:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(labels)
            correct += (logits.argmax(1) == labels).sum()
            total += len(labels)
            if proposed_start is not None and time.perf_counter() - proposed_start >= baseline_seconds:
                dump(output / 'rejected.json', dict(reason='Measured proposed time already reached baseline time', epoch=epoch))
                raise RejectedConfiguration('hard_end_to_end_time_budget')
        scheduler.step()
        sync()
        item = dict(epoch=epoch, train_images=total, train_top1_percent=100 * int(correct) / total,
                    loss=float(loss_sum) / total, training_seconds=time.perf_counter() - started)
        if epoch % 5 == 0 or epoch == epochs:
            item['validation'] = evaluate(model, validation)
            if item['validation']['top1_percent'] > best:
                best = item['validation']['top1_percent']
                torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, best_validation_percent=best,
                                classes=getattr(model,'_class_names',[])), output / 'best_validation.pth')
        item['cumulative_training_process_seconds']=time.perf_counter()-training_start
        history.append(item)
        dump(output / 'history.json', history)
        say('RESNET_EPOCH', phase=phase, **item)
        if proposed_start is not None and epoch >= 3:
            recent = history[1:][-5:]
            mean_epoch = float(np.mean([r['training_seconds'] for r in recent]))
            evaluation_overhead = sum(r.get('validation', {}).get('seconds', 0.) for r in history)
            remaining_evaluations = (epochs - epoch + 4) // 5
            eval_mean = evaluation_overhead / max(1, sum('validation' in r for r in history))
            predicted_total = time.perf_counter() - proposed_start + mean_epoch * (epochs - epoch) + max(5., eval_mean) * (remaining_evaluations + 1) + 10.
            guard = dict(epoch=epoch, elapsed_proposed_seconds=time.perf_counter() - proposed_start,
                         projected_end_to_end_seconds=predicted_total, baseline_measured_seconds=baseline_seconds,
                         estimate_only=True, recent_training_seconds_per_epoch=mean_epoch)
            dump(output / 'time_guard.json', guard)
            say('TIME_GUARD', phase=phase, **guard)
            if predicted_total >= baseline_seconds:
                dump(output / 'rejected.json', dict(reason='Projected total cannot beat measured baseline', **guard))
                raise RejectedConfiguration('projected_end_to_end_time_budget')
    torch.save(dict(model_state_dict=model.state_dict(), epoch=epochs,classes=getattr(model,'_class_names',[])), output / 'final_epoch.pth')
    checkpoint = torch.load(output / 'best_validation.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    del data
    return dict(epochs=epochs, presentations=len(rows) * epochs, training_process_seconds=time.perf_counter() - training_start,
                best_validation_percent=best, selected_epoch=checkpoint['epoch'], checkpoint=str(output / 'best_validation.pth'))


def source_numeric(image):
    # Randomized images are the complete input in that domain; no hidden original-image read.
    stats = helper.stats(image)
    values = {'source_' + k: float(v) for k, v in stats.items()}
    values.update({'original_' + k: float(v) for k, v in stats.items()})
    w, h = image.size
    values.update({'state_log_' + k: math.log(v) for k, v in dict(assigned_ratio=1., original_pixels=w*h,
                  current_pixels=w*h, actual_ratio=1., original_short=min(w,h), current_short=min(w,h),
                  original_long=max(w,h), current_long=max(w,h)).items()})
    for q in [7, 10, 15, 30, 60, 90, 100]:
        values['state_source_level_' + str(q)] = float(q == 100)
    for q in [.1, .25, .5, 1, 2, 5, 7, 10, 15, 30, 60, 90]:
        nw, nh = max(1, round(w*math.sqrt(q/100))), max(1, round(h*math.sqrt(q/100)))
        values[f'state_q{q:g}_log_pixels'] = math.log(nw*nh)
        values[f'state_q{q:g}_log_short'] = math.log(min(nw,nh))
        values[f'state_q{q:g}_native_224_ratio'] = min(nw,nh)/224
    gray = np.asarray(image.resize((128,128), Image.Resampling.BOX).convert('L'), dtype=np.uint8)
    for tag, a in [('full',gray), ('center',gray[32:96,32:96])]:
        values[f'cheap_v3_{tag}_zlib_ratio'] = len(zlib.compress(a.tobytes(),1))/a.size
    hsv = np.asarray(image.resize((64,64),Image.Resampling.BOX).convert('HSV'), dtype=np.int32)
    category = np.minimum(hsv[:,:,0]*8//256,7)*4 + np.minimum(hsv[:,:,1]*4//256,3)
    a,b = category[:-1,:],category[1:,:]
    mask = a == 7
    values['cheap_v3_color_corr_y1_7'] = float(np.mean(a[mask] == b[mask])) if mask.any() else 0.
    bounds = np.log(np.minimum(np.maximum(1,np.round(w*np.sqrt(CUTS/100))),np.maximum(1,np.round(h*np.sqrt(CUTS/100))))).astype('float32')
    return values, bounds


def reduced_image(image, q):
    if q >= 100:
        return image
    w,h = image.size
    return image.resize((max(1,round(w*math.sqrt(q/100))),max(1,round(h*math.sqrt(q/100)))),Image.Resampling.BOX)


def jpeg_roundtrip(image):
    buffer = io.BytesIO()
    image.save(buffer,format='JPEG',quality=CONFIG['jpeg_quality'],subsampling=CONFIG['jpeg_subsampling'],optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert('RGB')


class FeatureImages(Dataset):
    def __init__(self, rows, columns, teacher_views=False, domain='original', use_encoder=True):
        self.rows, self.columns, self.teacher_views, self.domain = rows,columns,teacher_views,domain
        self.use_encoder=use_encoder
        self.transform = eval_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        started = time.perf_counter()
        with Image.open(row['path']) as handle:
            image = handle.convert('RGB')
        values,bounds = source_numeric(image)
        raw = np.zeros(len(self.columns),np.float32)
        for j,c in enumerate(self.columns):
            if not c.startswith(('cheap_v3_mobile_','extra_cam_')):
                if c not in values:
                    raise KeyError('Feature not implemented: '+c)
                raw[j] = values[c]
        source = self.transform(image) if self.teacher_views or self.use_encoder else torch.empty(0)
        cpu_seconds = time.perf_counter() - started
        if self.teacher_views:
            views = []
            for q in TIERS:
                if q >= 100:
                    current = image
                else:
                    # Same BOX pixel algorithm, one source decode, temporary buffers only.
                    # Existing source sets/quality banks are never rewritten.
                    current = jpeg_roundtrip(reduced_image(image,q))
                views.append(source if q>=100 else self.transform(current))
            return torch.stack(views),raw,bounds,min(image.size),int(row['label']),index,cpu_seconds
        return source,raw,bounds,min(image.size),index,cpu_seconds


def enrich_features(mobile, images, raw, short, columns):
    features = mobile.features(images)
    pooled = torch.cat([features.mean((2,3)),features.std((2,3),unbiased=False)],1)
    import re
    locations, takes = [], []
    for j,c in enumerate(columns):
        if c.startswith('cheap_v3_mobile_'):
            match = re.fullmatch(r'cheap_v3_mobile_(mean|std)(\d+)',c)
            locations.append(j)
            takes.append(int(match[2]) + (576 if match[1]=='std' else 0))
    raw[:,locations] = pooled[:,takes].cpu().numpy()
    mean = pooled[:,:576]
    pre = mobile.classifier[0](mean)
    logits = mobile.classifier[-1](nn.functional.hardswish(pre))
    top = logits.argmax(1)
    derivative = torch.where(pre<=-3,0.,torch.where(pre>=3,1.,pre/3+.5))
    weights = (mobile.classifier[-1].weight[top]*derivative)@mobile.classifier[0].weight
    heat = (features*weights[:,:,None,None]).sum(1).clamp_min(0)
    scale = torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    offset = torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    rgb = images*scale+offset
    gray = .299*rgb[:,0]+.587*rgb[:,1]+.114*rgb[:,2]
    cam,names = helper.attention_stats(heat,gray,short.cuda().float(),'top1')
    lookup = {'extra_'+name:j for j,name in enumerate(names)}
    cam_values = cam.cpu().numpy()
    for j,c in enumerate(columns):
        if c.startswith('extra_cam_'):
            raw[:,j] = cam_values[:,lookup[c]]
    return raw


def oracle_and_features(teacher, rows, columns, domain, output):
    started = time.perf_counter()
    mobile = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).cuda().eval()
    data = DataLoader(FeatureImages(rows,columns,True,domain),batch_size=12,num_workers=CONFIG['workers'],
                      pin_memory=True,persistent_workers=True,shuffle=False)
    x = np.empty((len(rows),len(columns)),np.float32)
    bounds = np.empty((len(rows),10),np.float32)
    response = np.empty((len(rows),11),bool)
    soft = np.empty((len(rows),100),np.float32)
    proxy = np.empty((len(rows),36),np.float32)
    short_sides = np.empty(len(rows),np.float32)
    teacher.eval()
    with torch.inference_mode():
        for batch,(views,raw,b,short,label,indices,cpu) in enumerate(data):
            size = len(label)
            joined = views.flatten(0,1)
            logits = torch.cat([teacher(chunk.cuda(non_blocking=True)) for chunk in joined.split(128)]).reshape(size,11,100)
            probability = logits.softmax(-1)
            guesses = logits.argmax(-1)
            correct = (guesses==label.cuda()[:,None])
            raw = enrich_features(mobile,views[:,-1].cuda(non_blocking=True),raw.numpy(),short,columns)
            ix = indices.numpy()
            x[ix],bounds[ix],response[ix],soft[ix],short_sides[ix] = raw,b.numpy(),correct.cpu().numpy(),probability[:,-1].cpu().numpy(),short.numpy()
            for j,tier in enumerate([0,2,3,4,7,8]):
                confidence = probability[:,tier].max(-1).values
                good = correct[:,tier].float()
                agree = (guesses[:,tier]==guesses[:,-1]).float()
                values = torch.stack([confidence,good,agree,confidence*good,confidence*agree,torch.full_like(confidence,math.log1p(TIERS[tier])/math.log1p(100))],1)
                proxy[ix,j*6:j*6+6] = values.cpu().numpy()
            if batch%100==0:
                say('TEACHER_LABEL_FEATURE_PROGRESS',images=min(len(rows),(batch+1)*12),total=len(rows))
    # Strict safe suffix: all higher-quality tiers must also classify correctly.
    suffix = np.logical_and.accumulate(response[:,::-1],axis=1)[:,::-1]
    y = np.where(suffix.any(1),suffix.argmax(1),10).astype(np.int64)
    np.savez_compressed(output/'teacher_labels_features.npz',x=x,bounds=bounds,response=response,soft=soft,proxy=proxy,y=y,short=short_sides,
                        identities=np.asarray([r['id'] for r in rows]))
    write_rows(output/'teacher_tier_labels.csv',[
        dict(id=r['id'],source_path=r['path'],preferred_tier=int(tier),retained_pixels_percent=float(TIERS[tier]),
             teacher_source_correct=int(correct)) for r,tier,correct in zip(rows,y,response[:,-1])])
    timing = dict(seconds=time.perf_counter()-started,rows=len(rows),raw_candidate_features=len(columns),
                  tier_count=11,label_counts=np.bincount(y,minlength=11).tolist(),teacher_source_top1_percent=100*response[:,-1].mean(),
                  label_definition='Minimum tier in an all-correct quality suffix; source-incorrect examples retain the source tier',
                  source_encoder_passes_per_image=1,teacher_views_per_initial_image=11,
                  teacher_network_passes_on_remaining80k=0,codec='BOX pixel resize + JPEG95/4:4:4 for q<100; unchanged source at q=100')
    dump(output/'teacher_label_feature_timing.json',timing)
    say('TEACHER_LABEL_FEATURES_COMPLETE',**timing)
    del data,mobile
    torch.cuda.empty_cache()
    return x,bounds,response,soft,proxy,y,short_sides,timing


def tier_metrics(y, prediction):
    delta = prediction-y
    recalls = [float(np.mean(prediction[y==q]==q)) for q in range(11) if np.any(y==q)]
    confusion=np.zeros((11,11),np.int64)
    np.add.at(confusion,(y,prediction),1)
    return dict(images=len(y),exact_percent=100*float(np.mean(delta==0)),within_one_percent=100*float(np.mean(abs(delta)<=1)),
                unsafe_at_least_two_tiers_percent=100*float(np.mean(delta<=-2)),
                macro_recall_percent=100*float(np.mean(recalls)),
                true_counts=np.bincount(y,minlength=11).tolist(),predicted_counts=np.bincount(prediction,minlength=11).tolist(),
                majority_class_accuracy_percent=100*float(np.bincount(y,minlength=11).max()/len(y)),confusion=confusion.tolist())


def split_indices(indices,y,count,random_seed,groups=None):
    try:
        training,validation=train_test_split(indices,test_size=count,random_state=random_seed,stratify=y[indices])
    except ValueError:
        # Retain eleven outputs even when a fresh teacher produces singleton/absent tiers.
        training,validation=train_test_split(indices,test_size=count,random_state=random_seed,stratify=None)
    if groups is not None:
        training,validation=list(training),list(validation)
        frequency=Counter(groups[indices])
        overlaps=set(groups[training])&set(groups[validation])
        for group in sorted(overlaps):
            moving=[i for i in validation if groups[i]==group]
            for index in moving:
                replacement=next((i for i in training if frequency[groups[i]]==1 and y[i]==y[index]),None)
                if replacement is None:
                    replacement=next(i for i in training if frequency[groups[i]]==1)
                training.remove(replacement)
                validation.remove(index)
                training.append(index)
                validation.append(replacement)
        if set(groups[training])&set(groups[validation]):
            raise RuntimeError('MLP source identities cross train/holdout partitions')
    return np.asarray(training,dtype=np.int64),np.asarray(validation,dtype=np.int64)


def mlp_probabilities(net,x,bounds):
    net.eval()
    values=[]
    with torch.inference_mode():
        for first in range(0,len(x),2048):
            logits,_,_=net(torch.as_tensor(x[first:first+2048],device='cuda'),
                           torch.as_tensor(bounds[first:first+2048],device='cuda'))
            values.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(values)


def fit_mlp(frame,y,bounds,soft,proxy,train,validation,columns,cfg,fixed_epochs=None):
    seed()
    state=helper.fit_preprocess(frame,y,train,columns,dict(transform='quantile',k=0))
    x=helper.transform(frame,train,state)
    val=helper.transform(frame,validation,state)
    target=y[train]
    b=bounds[train]
    low=b[np.arange(len(train)),np.maximum(target-1,0)].copy()
    high=b[np.arange(len(train)),np.minimum(target,9)].copy()
    low[target==0]=high[target==0]-.7
    high[target==10]=low[target==10]+.7
    midpoint=(low+high)/2
    weights=soft[train]
    denominator=weights.sum(0).clip(1e-6)
    mu=(weights*midpoint[:,None]).sum(0)/denominator
    scale=np.sqrt((weights*(midpoint[:,None]-mu[None,:])**2).sum(0)/denominator).clip(.15,1.5)
    net=helper.Net(x.shape[1],cfg,mu,scale,False).cuda()
    xx=torch.as_tensor(x,device='cuda')
    yy=torch.as_tensor(target,device='cuda')
    bb=torch.as_tensor(b,device='cuda')
    teacher_soft=torch.as_tensor(weights,device='cuda')
    proxy_mean=proxy[train].mean(0)
    proxy_std=proxy[train].std(0).clip(.02)
    pp=torch.as_tensor((proxy[train]-proxy_mean)/proxy_std,device='cuda')
    safe=(torch.arange(11,device='cuda')[None,:]-yy[:,None]).abs()<=1
    cdf_truth=(torch.arange(11,device='cuda')[None,:]>=yy[:,None]).float()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.008)
    best_score=-1.
    best_state=None
    best_epoch=1
    stale=0
    for epoch in range(1,(fixed_epochs or 36)+1):
        net.train()
        ramp=min(1.,max(0.,(epoch-6)/20))
        for ix in torch.randperm(len(train),device='cuda').split(256):
            logits,semantic,forecast=net(xx[ix],bb[ix])
            probability=logits.softmax(1)
            loss=nn.functional.cross_entropy(logits,yy[ix])
            loss-=cfg['near']*ramp*(probability*safe[ix]).sum(1).clamp_min(1e-8).log().mean()
            loss+=.15*nn.functional.mse_loss(probability.cumsum(1),cdf_truth[ix])
            loss+=cfg['aux']*4*nn.functional.kl_div((semantic/2).log_softmax(1),teacher_soft[ix],reduction='batchmean')
            loss+=.1*nn.functional.smooth_l1_loss(forecast,pp[ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
        if fixed_epochs:
            best_epoch=epoch
            continue
        if epoch%3==0:
            probability=mlp_probabilities(net,val,bounds[validation])
            metric=tier_metrics(y[validation],probability.argmax(1))
            score=metric['exact_percent']+.35*metric['within_one_percent']
            if score>best_score:
                best_score=score
                best_epoch=epoch
                best_state={key:value.detach().cpu().clone() for key,value in net.state_dict().items()}
                stale=0
            else:
                stale+=3
            if epoch>=18 and stale>=9:
                break
    if not fixed_epochs:
        net.load_state_dict(best_state)
    return net,state,best_epoch,mlp_probabilities(net,val,bounds[validation])


def policy(probability,short,rule,native_floor=128):
    if rule=='argmax':
        raw=probability.argmax(1)
    else:
        raw=(probability.cumsum(1)<float(rule)).sum(1).clip(0,10)
    short=np.asarray(short)
    achievable=np.maximum(1,np.round(short[:,None]*np.sqrt(TIERS[None,:]/100)))
    required=np.minimum(short,native_floor)
    floor_index=(achievable>=required[:,None]).argmax(1)
    # This is a documented decision constraint, never a change to teacher labels.
    operational_raw=np.maximum(raw,np.maximum(floor_index-1,0))
    promoted=np.minimum(operational_raw+CONFIG['promotion'],10)
    return operational_raw,promoted


def train_tier_model(x,bounds,response,soft,proxy,y,short,columns,output,groups):
    started=time.perf_counter()
    frame=pd.DataFrame(x,columns=columns)
    development,holdout=split_indices(np.arange(len(y)),y,4000,20261022,groups)
    fitting,calibration=split_indices(development,y,4000,20261023,groups)
    numeric=[c for c in columns if not c.startswith(('cheap_v3_mobile_','extra_cam_'))]
    records=[]
    for family,feature_list in [('numeric',numeric),('source_encoder',columns)]:
        for near in [.5,1.5]:
            cfg=dict(name='v28_optimized',near=near,aux=.2,residual=1.)
            net,state,epochs,p=fit_mlp(frame,y,bounds,soft,proxy,fitting,calibration,feature_list,cfg)
            metrics=tier_metrics(y[calibration],p.argmax(1))
            candidates=[]
            for rule in ['argmax',.5,.75,.9,.97]:
                raw,applied=policy(p,short[calibration],rule,CONFIG['minimum_native_short'])
                lost=response[calibration,-1]&~response[calibration,applied]
                candidates.append(dict(rule=rule,teacher_accuracy_loss=float(lost.mean()),
                                       mean_retained_pixels_percent=float(TIERS[applied].mean())))
            feasible=[r for r in candidates if r['teacher_accuracy_loss']<=CONFIG['teacher_policy_drop_limit']]
            decision=min(feasible,key=lambda r:r['mean_retained_pixels_percent']) if feasible else min(candidates,key=lambda r:(r['teacher_accuracy_loss'],r['mean_retained_pixels_percent']))
            record=dict(family=family,config=cfg,epochs=epochs,metrics=metrics,policy=decision,
                        raw_input_candidates=len(feature_list),effective_features=len(state['feature_columns']))
            records.append(record)
            say('MLP_DEVELOPMENT_CANDIDATE',**record)
            del net
            torch.cuda.empty_cache()
    selected=max(records,key=lambda r:r['metrics']['exact_percent']+.35*r['metrics']['within_one_percent'])
    # Drop the encoder only when development accuracy is effectively preserved.
    cheap=[r for r in records if r['family']=='numeric' and
           r['metrics']['exact_percent']>=selected['metrics']['exact_percent']-1. and
           r['metrics']['within_one_percent']>=selected['metrics']['within_one_percent']-2. and
           r['policy']['teacher_accuracy_loss']<=CONFIG['teacher_policy_drop_limit']]
    if cheap:
        selected=max(cheap,key=lambda r:r['metrics']['exact_percent']+.35*r['metrics']['within_one_percent'])
    chosen=numeric if selected['family']=='numeric' else columns
    net,state,epochs,p=fit_mlp(frame,y,bounds,soft,proxy,development,holdout,chosen,selected['config'],selected['epochs'])
    raw,applied=policy(p,short[holdout],selected['policy']['rule'],CONFIG['minimum_native_short'])
    diagnostic=dict(argmax=tier_metrics(y[holdout],p.argmax(1)),
                    operational_raw=tier_metrics(y[holdout],raw),promoted=tier_metrics(y[holdout],applied),
                    teacher_source_accuracy_percent=100*float(response[holdout,-1].mean()),
                    promoted_teacher_accuracy_percent=100*float(response[holdout,applied].mean()),
                    promoted_teacher_accuracy_loss_points=100*float((response[holdout,-1]&~response[holdout,applied]).mean()),
                    holdout_rows=4000,training_rows=16000,holdout_used_for_recipe_selection=False,source_group_overlap=0)
    np.savez_compressed(output/'mlp_holdout_predictions.npz',indices=holdout,y=y[holdout],p=p,raw=raw,promoted=applied)
    torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},preprocessing=state,
                    selected=selected,classes=11),output/'mlp_evaluation_16k.pt')
    del net
    all_indices=np.arange(len(y))
    net,state,epochs,_=fit_mlp(frame,y,bounds,soft,proxy,all_indices,all_indices[:1],chosen,selected['config'],selected['epochs'])
    torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},preprocessing=state,selected=selected,
                    class_id_input=False,feature_source='current input only',trained_rows=20000,classes=11),output/'mlp_full20k.pt')
    pd.DataFrame({'feature':state['feature_columns']}).to_csv(output/'selected_effective_features.csv',index=False)
    result=dict(selected=selected,development_candidates=records,diagnostic=diagnostic,
                final_training_rows=20000,candidate_feature_count=215,selected_effective_features=len(state['feature_columns']),
                class_id_input=False,seconds=time.perf_counter()-started,
                tier_levels_retained_current_input_pixels_percent=TIERS.tolist(),labels_edited=False)
    dump(output/'mlp_results.json',result)
    say('MLP_TRAINING_COMPLETE',**result)
    return net,state,selected,result


def deploy_features(rows,columns,use_encoder,output):
    started=time.perf_counter()
    mobile=models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).cuda().eval() if use_encoder else None
    data=DataLoader(FeatureImages(rows,columns,use_encoder=use_encoder),batch_size=128,num_workers=CONFIG['workers'],
                    pin_memory=True,persistent_workers=True,shuffle=False)
    x=np.empty((len(rows),len(columns)),np.float32)
    bounds=np.empty((len(rows),10),np.float32)
    short_sides=np.empty(len(rows),np.float32)
    cpu_total=0.
    encoder_seconds=0.
    with torch.inference_mode():
        for batch,(images,raw,b,short,indices,cpu) in enumerate(data):
            array=raw.numpy()
            if use_encoder:
                sync()
                encoder_start=time.perf_counter()
                array=enrich_features(mobile,images.cuda(non_blocking=True),array,short,columns)
                sync()
                encoder_seconds+=time.perf_counter()-encoder_start
            ix=indices.numpy()
            x[ix],bounds[ix],short_sides[ix]=array,b.numpy(),short.numpy()
            cpu_total+=float(cpu.sum())
            if batch%100==0:
                say('DEPLOYMENT_FEATURE_PROGRESS',images=min(len(rows),(batch+1)*128),total=len(rows))
    timing=dict(images=len(rows),seconds=time.perf_counter()-started,source_encoder_passes_per_image=int(use_encoder),
                teacher_resnet_passes=0,encoder_gpu_wall_seconds=encoder_seconds,
                decode_numeric_worker_seconds_sum=cpu_total,worker_seconds_are_not_wall_seconds=True)
    dump(output/'deployment_feature_timing.json',timing)
    del data,mobile
    torch.cuda.empty_cache()
    return pd.DataFrame(x,columns=columns),bounds,short_sides,timing


def encode_selected(task):
    path,q,destination=task
    destination=Path(destination)
    if q>=100:
        os.symlink(str(Path(path).resolve()),destination)
        return destination.stat().st_size,0
    with Image.open(path) as handle:
        image=reduced_image(handle.convert('RGB'),q)
    image.save(destination,format='JPEG',quality=CONFIG['jpeg_quality'],subsampling=CONFIG['jpeg_subsampling'],optimize=False)
    size=destination.stat().st_size
    return size,size


def materialize(rows,applied,output):
    started=time.perf_counter()
    folder=output/'selected80k'
    folder.mkdir(exist_ok=False)
    for category in {r['class_name'] for r in rows}:
        (folder/category).mkdir()
    tasks=[]
    selected=[]
    for row,tier in zip(rows,applied):
        destination=folder/Path(row['relative_path']).with_suffix('.jpg')
        tasks.append((row['path'],float(TIERS[tier]),str(destination)))
        selected.append(dict(row,path=str(destination),applied_tier=int(tier),retained_pixels_percent=float(TIERS[tier])))
    footprint=written=0
    with ProcessPoolExecutor(CONFIG['workers']) as pool:
        for index,(size,new_bytes) in enumerate(pool.map(encode_selected,tasks,chunksize=32)):
            footprint+=size
            written+=new_bytes
            if (index+1)%10000==0:
                say('PARALLEL_COMPRESSION_PROGRESS',images=index+1,total=len(rows))
    write_rows(output/'selected80k_manifest.csv',selected)
    source_bytes=sum(int(r['bytes']) for r in rows)
    timing=dict(images=len(rows),seconds=time.perf_counter()-started,effective_storage_bytes=footprint,
                new_physical_bytes=written,source_storage_bytes=source_bytes,
                storage_saved_percent=100*(1-footprint/source_bytes),counts=np.bincount(applied,minlength=11).tolist(),
                codec='JPEG95 4:4:4 after BOX pixel resize; source retained at 100%',workers=CONFIG['workers'])
    dump(output/'compression_timing_storage.json',timing)
    say('SELECTED_COMPRESSION_COMPLETE',**timing)
    return selected,timing


def baseline_arm(run,domain,output,rows,validation,test):
    model=fresh_resnet()
    model._class_names=json.loads((run/'protocol.json').read_text())['classes']
    training=train_resnet(model,rows,validation,CONFIG['baseline_epochs'],CONFIG['learning_rate'],
                          output/'training','baseline_'+domain)
    final_test=evaluate(model,test,output/'test_predictions.csv')
    result=dict(status='complete',domain=domain,arm='baseline',training=training,test=final_test,
                training_presentations=training['presentations'],source_images=len(rows),mlp_used=False,
                source_storage_bytes=sum(int(r['bytes']) for r in rows))
    return result


def proposed_arm(run,domain,output,rows,validation,test,started):
    baseline=json.loads((run/domain/'baseline/process_result.json').read_text())
    budget=baseline['externally_measured_process_seconds']
    teacher_rows=[r for r in rows if r['part']=='teacher']
    remaining=[r for r in rows if r['part']=='remaining']
    columns=pd.read_csv(Path(__file__).with_name('optimized_v28_features.csv')).feature.tolist()
    stages={}
    def check_budget(stage):
        dump(output/'stage_progress.json',dict(stages=stages,current_stage=stage,
                                              elapsed_seconds=time.perf_counter()-started,baseline_seconds=budget))
        if time.perf_counter()-started>=budget:
            raise RejectedConfiguration('baseline_time_budget_exhausted_before_'+stage)
    preflight_start=time.perf_counter()
    if len(teacher_rows)!=20000 or len(remaining)!=80000:
        raise RuntimeError('Proposed source partition sizes are wrong')
    stages['proposal_specific_preflight_seconds']=time.perf_counter()-preflight_start
    say('PROPOSED_INPUTS_VERIFIED',domain=domain,teacher_rows=len(teacher_rows),remaining_rows=len(remaining),
        all_source_paths_verified_by_common_inventory=True,teacher_probes_in_memory=True,quality_bank_files_created=0)
    teacher=fresh_resnet()
    teacher._class_names=json.loads((run/'protocol.json').read_text())['classes']
    before=time.perf_counter()
    teacher_training=train_resnet(teacher,teacher_rows,validation,CONFIG['teacher_epochs'],CONFIG['learning_rate'],
                                  output/'teacher20k','teacher_'+domain)
    stages['teacher_training_seconds']=time.perf_counter()-before
    check_budget('teacher_label_generation')
    x,bounds,response,soft,proxy,y,short,oracle=oracle_and_features(teacher,teacher_rows,columns,domain,output)
    stages['teacher_labels_and_initial_features_seconds']=oracle['seconds']
    check_budget('mlp_fitting')
    teacher_weights={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    del teacher
    torch.cuda.empty_cache()
    groups=np.asarray([r['source_sha256'] for r in teacher_rows])
    net,state,recipe,mlp=train_tier_model(x,bounds,response,soft,proxy,y,short,columns,output,groups)
    stages['mlp_search_diagnostic_and_full20k_fitting_seconds']=mlp['seconds']
    check_budget('80k_features')
    frame,b80,short80,feature_timing=deploy_features(remaining,columns,recipe['family']=='source_encoder',output)
    stages['remaining80k_feature_seconds']=feature_timing['seconds']
    check_budget('80k_mlp_prediction')
    before=time.perf_counter()
    transformed=helper.transform(frame,np.arange(len(frame)),state)
    stages['remaining80k_preprocessing_seconds']=time.perf_counter()-before
    if transformed.shape[1]!=net.body[0].in_features:
        raise RuntimeError('MLP effective feature dimension differs from fitted preprocessing')
    sync()
    before=time.perf_counter()
    probabilities=mlp_probabilities(net,transformed,b80)
    sync()
    stages['remaining80k_mlp_forward_seconds']=time.perf_counter()-before
    raw,applied=policy(probabilities,short80,recipe['policy']['rule'],CONFIG['minimum_native_short'])
    predictions=[dict(id=r['id'],source_path=r['path'],network_argmax=int(a),operational_raw=int(b),
                      applied_plus_one=int(c),retained_current_input_pixels_percent=float(TIERS[c]))
                 for r,a,b,c in zip(remaining,probabilities.argmax(1),raw,applied)]
    write_rows(output/'mlp_80k_predictions.csv',predictions)
    np.savez_compressed(output/'mlp_80k_probabilities.npz',p=probabilities,raw=raw,applied=applied)
    check_budget('80k_selected_compression')
    selected,compression=materialize(remaining,applied,output)
    stages['selected80k_parallel_compression_seconds']=compression['seconds']
    del net,frame,transformed,x,bounds,soft,proxy
    torch.cuda.empty_cache()
    attempts=[]
    final_training=None
    elapsed=time.perf_counter()-started
    if elapsed>=budget:
        return dict(status='time_guard_rejected_before_downstream',domain=domain,arm='proposed',stages=stages,
                    teacher_training=teacher_training,mlp=mlp,compression=compression,
                    measured_elapsed_seconds=elapsed,baseline_seconds=budget,acceptance=False)
    for attempt,epochs in enumerate([CONFIG['downstream_epochs'],30],1):
        model=fresh_resnet()
        model._class_names=json.loads((run/'protocol.json').read_text())['classes']
        model.load_state_dict(teacher_weights,strict=True)
        before=time.perf_counter()
        try:
            final_training=train_resnet(model,selected,validation,epochs,CONFIG['downstream_lr'],
                                         output/('remaining80k_attempt'+str(attempt)),'remaining_'+domain,
                                         proposed_start=started,baseline_seconds=budget)
            record=dict(attempt=attempt,status='complete',training=final_training,seconds=time.perf_counter()-before)
            attempts.append(record)
            break
        except RejectedConfiguration as rejection:
            elapsed=time.perf_counter()-before
            attempts.append(dict(attempt=attempt,status='discarded',reason=str(rejection),seconds=elapsed))
            say('DOWNSTREAM_CONFIGURATION_DISCARDED',domain=domain,attempt=attempt,reason=str(rejection),seconds=elapsed)
            del model
            torch.cuda.empty_cache()
            if attempt==CONFIG['max_downstream_attempts'] or time.perf_counter()-started>=budget*.8:
                break
    stages['all_downstream_attempts_seconds']=sum(r['seconds'] for r in attempts)
    if final_training is None:
        return dict(status='time_guard_rejected',domain=domain,arm='proposed',stages=stages,
                    teacher_training=teacher_training,mlp=mlp,compression=compression,attempts=attempts,
                    measured_elapsed_seconds=time.perf_counter()-started,baseline_seconds=budget,acceptance=False)
    validation_target=baseline['training']['best_validation_percent']-CONFIG['max_accuracy_drop_points']
    final_test=evaluate(model,test,output/'test_predictions.csv')
    result=dict(status='complete',domain=domain,arm='proposed',teacher_training=teacher_training,
                downstream_training=final_training,attempts=attempts,stages=stages,mlp=mlp,compression=compression,
                test=final_test,validation_target_percent=validation_target,
                validation_target_met=final_training['best_validation_percent']>=validation_target,
                training_presentations_completed_models=teacher_training['presentations']+final_training['presentations'],
                discarded_attempts_included_in_total_wall_time=True,
                accuracy_drop_points=baseline['test']['top1_percent']-final_test['top1_percent'],
                total_training_set_effective_storage_bytes=sum(int(r['bytes']) for r in teacher_rows)+compression['effective_storage_bytes'])
    return result


def arm(run,domain,which):
    if not torch.cuda.is_available():
        raise RuntimeError('Run only inside the allocated GPU job')
    seed()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    protocol=json.loads((run/'protocol.json').read_text())
    rows=read_rows(run/(domain+'_sources.csv'))
    validation,test=read_rows(run/'validation.csv'),read_rows(run/'test.csv')
    output=run/domain/which
    output.mkdir(parents=True,exist_ok=False)
    say('ARM_START',domain=domain,arm=which,gpu=torch.cuda.get_device_name(0),host=os.uname().nodename,
        training_images=len(rows),validation_images=len(validation),test_images=len(test),task_weights='fresh')
    if which=='baseline':
        result=baseline_arm(run,domain,output,rows,validation,test)
    else:
        try:
            result=proposed_arm(run,domain,output,rows,validation,test,started)
        except RejectedConfiguration as rejection:
            result=dict(status='preprocessing_time_guard_rejected',domain=domain,arm=which,reason=str(rejection),
                        progress=json.loads((output/'stage_progress.json').read_text()) if (output/'stage_progress.json').is_file() else {})
    result.update(gpu=torch.cuda.get_device_name(0),host=os.uname().nodename,
                  measured_worker_seconds=time.perf_counter()-started,
                  peak_main_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  classes=protocol['classes'],tier_count=11,test_manifest_sha256=protocol['test_manifest_sha256'])
    dump(output/'result.json',result)
    say('ARM_COMPLETE',domain=domain,arm=which,status=result['status'],test=result.get('test'))


def cpu_preflight():
    """Small meaningful schema/codec/model checks, never a second Slurm test job."""
    columns=pd.read_csv(Path(__file__).with_name('optimized_v28_features.csv')).feature.tolist()
    with Image.open(ROOT/'data/clean/compression_200/n01498041/0000.jpg') as handle:
        image=handle.convert('RGB')
    values,bounds=source_numeric(image)
    unsupported=[c for c in columns if not c.startswith(('cheap_v3_mobile_','extra_cam_')) and c not in values]
    if unsupported:
        raise RuntimeError('Unimplemented selected features: '+str(unsupported))
    for q in TIERS:
        resized=reduced_image(image,q)
        if q<100 and jpeg_roundtrip(resized).size!=resized.size:
            raise RuntimeError('Inconsistent codec dimensions')
    bank=ROOT/'results/compression_20000_4409/n01498041/0000/pixels_15pct.png'
    with Image.open(bank) as handle:
        cached=handle.convert('RGB')
    if not np.array_equal(np.asarray(cached),np.asarray(reduced_image(image,15.))):
        raise RuntimeError('RAM BOX implementation differs from the existing quality-bank algorithm')
    # Classifier dimensions explicitly follow the transformed feature count.
    dummy=helper.Net(17,dict(residual=1.),np.zeros(100,np.float32),np.ones(100,np.float32),False).eval()
    with torch.inference_mode():
        logits,_,_=dummy(torch.zeros(2,17),torch.as_tensor(np.stack([bounds,bounds])))
    if logits.shape!=(2,11) or not torch.isfinite(logits).all():
        raise RuntimeError('Invalid fresh MLP forward')
    say('STATIC_AND_CPU_PREFLIGHT_OK',candidate_features=len(columns),tier_outputs=11,classes=100)


def write_report(run,summary):
    lines=['# Original-first optimized MTP experiment','',
           'Each pair uses identical underlying 100k source identities, a common independent 1k validation set, and the same 5k test manifest. All task-specific ResNet and MLP weights are fresh. The two source versions are not regenerated.','',
           'The proposed schedule uses 20 teacher epochs plus 45 continuation epochs (4 million presentations) versus 50 full-data epochs (5 million) for the ordinary ResNet. A time-guard retry may use 30 continuation epochs. These schedule changes contribute to speed; the result is not a compression-only speed claim.','',
           'All runs in this job share one allocated GPU. Input inventory/hash checking is measured as shared experiment preparation and reported separately. Every arm clock is an actual fresh subprocess stopwatch. Proposed clocks include all attempted configurations, labels, features, MLP fits, prediction, output encoding and final evaluation.','']
    protocol=json.loads((run/'protocol.json').read_text())
    lines+=['Shared input preparation: %.3f seconds.'%protocol['common_inventory_and_integrity_seconds'],'']
    for domain,pair in summary['experiments'].items():
        lines+=['## '+domain,'','| Measure | Ordinary ResNet | Proposed MLP pipeline |','|---|---:|---:|']
        baseline=pair.get('baseline',{})
        proposed=pair.get('proposed',{})
        def value(model,key,default='pending'):
            return model.get(key,default)
        lines += [f"| Status | {value(baseline,'status')} | {value(proposed,'status')} |",
                  f"| Actual process seconds | {value(baseline,'externally_measured_process_seconds')} | {value(proposed,'externally_measured_process_seconds')} |",
                  f"| Test top-1 percent | {baseline.get('test',{}).get('top1_percent','pending')} | {proposed.get('test',{}).get('top1_percent','not measured')} |",
                  f"| Test top-5 percent | {baseline.get('test',{}).get('top5_percent','pending')} | {proposed.get('test',{}).get('top5_percent','not measured')} |",
                  f"| Peak GPU allocated bytes | {value(baseline,'peak_gpu_allocated_bytes')} | {value(proposed,'peak_gpu_allocated_bytes')} |",
                  f"| Peak main-process RSS KiB (workers excluded) | {value(baseline,'peak_main_process_rss_kib')} | {value(proposed,'peak_main_process_rss_kib')} |",
                  f"| Input/effective training payload bytes | {value(baseline,'source_storage_bytes')} | {value(proposed,'total_training_set_effective_storage_bytes')} |",'']
        if 'comparison' in pair:
            lines+=['Comparison: `'+json.dumps(pair['comparison'])+'`','']
        if proposed.get('mlp'):
            mlp=proposed['mlp']
            lines+=['### Tier MLP','',
                    'Inputs come only from the current source image. Class IDs are excluded. The numeric-only family removes the source encoder if development exact accuracy stays within 1 point and within-one stays within 2 points of the best development model. Otherwise a single frozen MobileNetV3-Small source pass supplies features; no ResNet passes are used on the remaining 80k.','',
                    f"Effective input features: {mlp['selected_effective_features']}; family: {mlp['selected']['family']}; output classes: 11.",'',
                    '| Decision on the independent 4k tier holdout | Exact % | Within-one % | Underpredicted by at least two tiers % |','|---|---:|---:|---:|']
            for name,m in mlp['diagnostic'].items():
                if isinstance(m,dict) and 'exact_percent' in m:
                    lines.append(f"| {name} | {m['exact_percent']:.3f} | {m['within_one_percent']:.3f} | {m['unsafe_at_least_two_tiers_percent']:.3f} |")
            lines+=['','Teacher labels are unchanged. The operational decision adds a minimum-native-resolution constraint before the one-tier promotion. Labels and within-one errors refer to ordered retained-pixel tiers, not JPEG quality numbers.','',
                    'Retained pixel percentages: '+str(TIERS.tolist()),'',
                    '### Actual proposed stage seconds','','| Stage | Seconds |','|---|---:|']
            for stage,seconds in proposed.get('stages',{}).items():
                lines.append(f'| {stage} | {seconds:.6f} |')
            lines+=['','Raw argmax, operational and promoted distributions, confusion matrices, majority-class scores and unsafe metrics are saved in `mlp_results.json`. All 80k operational decisions are in `mlp_80k_predictions.csv`.','']
    lines+=['Overall job state: '+summary['status'],'',
            'This is a single observed experiment per source version, not repeated-seed confidence evidence. Frozen ImageNet MobileNet pretraining is optional feature generation and is explicitly distinct from scratch task ResNet initialization. Exact hash checks do not exclude differently encoded near duplicates.','']
    (run/'COMPARISON_REPORT.md').write_text('\n'.join(lines))


def orchestrate(run):
    if not torch.cuda.is_available():
        raise RuntimeError('Experiment requires one allocated GPU')
    torch.set_num_threads(4)
    run.mkdir(parents=True,exist_ok=False)
    overall=time.perf_counter()
    code=run/'code'
    code.mkdir()
    for name in ['optimized_paired_mtp.py','optimized_mtp_helpers.py','optimized_v28_features.csv','run_optimized_paired_mtp.slurm']:
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    dump(run/'code_sha256.json',{p.name:sha(p) for p in code.iterdir()})
    cpu_preflight()
    # One in-allocation structural check catches encoder/schema failures before long training.
    columns=pd.read_csv(Path(__file__).with_name('optimized_v28_features.csv')).feature.tolist()
    sample=dict(path=str(ROOT/'data/clean/compression_200/n01498041/0000.jpg'),relative_path='n01498041/0000.jpg',label=0)
    views,raw,bounds,short,label,index,cpu=FeatureImages([sample],columns,True,'original')[0]
    mobile=models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).cuda().eval()
    with torch.inference_mode():
        checked=enrich_features(mobile,views[-1:].cuda(),np.stack([raw]),torch.tensor([short]),columns)
    if checked.shape!=(1,215) or not np.isfinite(checked).all():
        raise RuntimeError('Feature schema failed GPU preflight')
    del mobile,views
    # Check the former constant-feature failure and training/inference contracts.
    random_state=np.random.default_rng(SEED)
    synthetic=random_state.normal(size=(44,215)).astype(np.float32)
    synthetic[:,0]=1.
    synthetic[:,1]=synthetic[:,2]
    labels=np.tile(np.arange(11),4)
    b=np.tile(bounds,(44,1))
    soft=np.full((44,100),.01,np.float32)
    proxy=random_state.normal(size=(44,36)).astype(np.float32)
    net,state,ep,p=fit_mlp(pd.DataFrame(synthetic,columns=columns),labels,b,soft,proxy,
                         np.arange(33),np.arange(33,44),columns,dict(near=.5,aux=.2,residual=1.),fixed_epochs=1)
    if p.shape!=(11,11) or not np.isfinite(p).all() or len(state['feature_columns'])>=215:
        raise RuntimeError('MLP dynamic preprocessing/schema check failed')
    constrained,promoted=policy(p,np.full(11,256.),.9)
    if not np.array_equal(promoted,np.minimum(constrained+1,10)):
        raise RuntimeError('One-tier safety promotion contract failed')
    del net
    torch.cuda.empty_cache()
    say('IN_ALLOCATION_FEATURE_PREFLIGHT_OK',features=215,teacher_views=11,effective_synthetic_features=len(state['feature_columns']),mlp_backward_checked=True)
    prepare(run)
    summary=dict(status='running',run_dir=str(run),protocol=CONFIG,experiments={})
    dump(run/'experiment_status.json',summary)
    for domain in ['original','randomized']:
        pair={}
        for which in ['baseline','proposed']:
            log=run/(domain+'_'+which+'.out')
            started=time.perf_counter()
            with log.open('w',buffering=1) as handle:
                completed=subprocess.run([sys.executable,'-u',str(code/'optimized_paired_mtp.py'),'--run-dir',str(run),
                                          '--domain',domain,'--arm',which],stdout=handle,stderr=subprocess.STDOUT,check=False)
            measured=time.perf_counter()-started
            output=run/domain/which
            if completed.returncode!=0:
                summary.update(status='failed',failed_domain=domain,failed_arm=which,exit_code=completed.returncode,
                               measured_failed_process_seconds=measured,log=str(log))
                dump(run/'experiment_status.json',summary)
                raise RuntimeError('Experiment process failed; inspect '+str(log))
            result=json.loads((output/'result.json').read_text())
            result['externally_measured_process_seconds']=measured
            dump(output/'process_result.json',result)
            pair[which]=result
            summary['experiments'][domain]=pair
            dump(run/'experiment_status.json',summary)
            write_report(run,summary)
            say('MEASURED_PROCESS_COMPLETE',domain=domain,arm=which,actual_seconds=measured,status=result['status'])
        baseline,proposed=pair['baseline'],pair['proposed']
        faster=proposed['externally_measured_process_seconds']<baseline['externally_measured_process_seconds']
        accurate=proposed.get('test',{}).get('top1_percent',-100)>=baseline['test']['top1_percent']-CONFIG['max_accuracy_drop_points']
        pair['comparison']=dict(accepted=proposed['status']=='complete' and faster and accurate,
                                proposed_faster=faster,accuracy_within_three_points=accurate,
                                measured_seconds_saved=baseline['externally_measured_process_seconds']-proposed['externally_measured_process_seconds'],
                                measured_time_saved_percent=100*(1-proposed['externally_measured_process_seconds']/baseline['externally_measured_process_seconds']),
                                baseline_top1_percent=baseline['test']['top1_percent'],proposed_top1_percent=proposed.get('test',{}).get('top1_percent'),
                                proposal_all_attempts_included=True)
        summary['experiments'][domain]=pair
        dump(run/(domain+'_comparison.json'),pair['comparison'])
        dump(run/'experiment_status.json',summary)
        write_report(run,summary)
        say('DOMAIN_COMPARISON_COMPLETE',domain=domain,**pair['comparison'])
    summary.update(status='complete',measured_orchestration_seconds=time.perf_counter()-overall)
    dump(run/'experiment_status.json',summary)
    write_report(run,summary)
    say('ALL_EXPERIMENTS_COMPLETE',run_dir=str(run),comparisons={k:v['comparison'] for k,v in summary['experiments'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--domain',choices=['original','randomized'])
    parser.add_argument('--arm',choices=['baseline','proposed'])
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.preflight_only:
        cpu_preflight()
    elif args.arm:
        arm(args.run_dir,args.domain,args.arm)
    else:
        destination=args.run_dir or ROOT/'results/optimized_paired_v1'/('job_'+os.environ['SLURM_JOB_ID'])
        orchestrate(destination)



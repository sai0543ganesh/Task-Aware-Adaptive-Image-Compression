"""Create reproducible Food-101 manifests in a fresh Slurm scratch tree."""
import csv
import hashlib
import json
import math
import random
import sys
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1]).resolve()
with tarfile.open(root/'food-101.tar.gz') as bank:
    for m in bank:
        if not (m.isfile() or m.isdir()) or not (root/m.name).resolve().is_relative_to(root):
            raise ValueError(f'Unexpected archive member {m.name}')
        bank.extract(m, root)
base = root/'food-101'
classes = sorted((base/'meta/classes.txt').read_text().splitlines())
assert len(classes) == len(set(classes)) == 101
class_id = {name: i for i, name in enumerate(classes)}
train = (base/'meta/train.txt').read_text().splitlines()
test = (base/'meta/test.txt').read_text().splitlines()
assert len(train) == len(set(train)) == 75750 and len(test) == len(set(test)) == 25250
assert not set(train) & set(test)

def describe(key):
    name = key.split('/')[0]
    path = base/'images'/(key+'.jpg')
    data = path.read_bytes()
    with Image.open(path) as im:
        w, h = im.size
    return dict(id=key, path=str(path), label=class_id[name], class_name=name,
                bytes=len(data), source_sha256=hashlib.sha256(data).hexdigest(), width=w, height=h)

with ThreadPoolExecutor(max_workers=8) as pool:
    training = list(pool.map(describe, train))
    testing = list(pool.map(describe, test))
test_hashes = {r['source_sha256'] for r in testing}
seen = set(test_hashes)
unique = {name: [] for name in classes}
duplicates = 0
tiny_sources = 0
for row in sorted(training, key=lambda r: r['id']):
    if row['source_sha256'] in seen:
        duplicates += 1
        continue
    seen.add(row['source_sha256'])
    if row['width']*row['height'] < 32**2:
        tiny_sources += 1
        continue
    unique[row['class_name']].append(row)

rng = random.Random(42)
initial, rest, validation, holdout = [], [], [], []
for name, rows in unique.items():
    assert len(rows) >= 100, name
    rng.shuffle(rows)
    validation.extend(rows[:25])
    candidates = rows[25:]
    pivot = round(.2*len(candidates))
    holdout.extend(range(len(initial), len(initial)+max(1,round(.2*pivot))))
    initial.extend(candidates[:pivot])
    rest.extend(candidates[pivot:])
for part, rows in [('teacher',initial),('remaining',rest),('validation',validation),('test',testing)]:
    for row in rows:
        row['part'] = part
        row['sha256'] = row['source_sha256']
    assert set(Counter(r['class_name'] for r in rows)) == set(classes)
sides = [32,40,48,56,72,96,120,136,160,168,224]
tiers = [.1,.25,.5,1,5,7,10,15,60,90,100]
smallest = min(max(1, r['width'])*max(1, r['height']) for r in initial+rest)
assert smallest >= 32**2, 'A source is too small for the existing mapping grid'
saved = root/'food_manifest'
saved.mkdir()
for name, rows in [('sources',initial+rest),('validation',validation),('test',testing)]:
    with (saved/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
hold = set(holdout)
(saved/'grid_split_policy.json').write_text(json.dumps(dict(sides=sides,tiers=tiers,holdout_indices=holdout,
    fit_indices=[i for i in range(len(initial)) if i not in hold])))
protocol=dict(dataset='Food-101',classes=classes,initial=len(initial),remaining=len(rest),validation=len(validation),
              test=len(testing),training_entries_removed_as_exact_duplicates=duplicates,
              training_entries_below_minimum_canvas=tiny_sources,
              official_test_kept=True,validation_per_class=25,seed=42,
              decoded_training_bytes=sum(r['width']*r['height']*3 for r in initial+rest))
def mapped(w, h, q):
    scale = math.sqrt(q/100)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    return rw, rh, max((s for s in sides if s*s <= rw*rh), default=0)

floor_counts, distinct_counts = Counter(), Counter()
for row in initial+rest:
    w, h = row['width'], row['height']
    views = [mapped(w,h,q) for q in tiers]
    required = min(96,views[-1][2])
    floor = next(i for i,(rw,rh,side) in enumerate(views)
                 if min(rw,rh) >= min(w,h,96) and side >= required)
    floor_counts[floor] += 1
    distinct_counts[len({side for _,_,side in views[floor:]})] += 1
protocol.update(retained_area_percent=tiers,mapped_sides=sides,
                floor_tier_counts=dict(floor_counts),distinct_feasible_canvas_counts=dict(distinct_counts))
(saved/'protocol.json').write_text(json.dumps(protocol,indent=2))
(saved/'optimized_v28_features.csv').write_bytes((root/'optimized_v28_features.csv').read_bytes())
sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in saved.iterdir()}
(saved/'ASSET_MANIFEST.json').write_text(json.dumps(dict(sha256=sha)))
with tarfile.open(root/'cross_backbone_inputs.tar.gz','w:gz') as bank:
    for p in sorted(saved.iterdir()):
        bank.add(p,arcname=p.name)
print(json.dumps(protocol),flush=True)

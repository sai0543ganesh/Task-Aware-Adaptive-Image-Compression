"""Parallel implementation of 5327's unchanged BICUBIC -> BILINEAR mapping.

No quality changes, no augmentation, no source writes. Geometry floors are
supplied explicitly by the frozen teacher/deployment policy. Callers must count
materialization and RAM factory time together in complete pipeline timing.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

TIERS = np.array([.1, .25, .5, 1., 5., 7., 10., 15., 60., 90., 100.])


def map_selected_images(data, tiers, sides, floor, workers=4, ram_factory=None):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    tiers, floor = np.asarray(tiers), np.asarray(floor)
    n = len(data)
    if tiers.shape != (n,) or floor.shape != (n,):
        raise ValueError('one tier and frozen floor required per image')
    if not np.issubdtype(tiers.dtype, np.integer) or not np.issubdtype(floor.dtype, np.integer):
        raise ValueError('tier and floor indices must be integers')
    if ((tiers < 0) | (tiers > 10) | (floor < 0) | (floor > 10)).any():
        raise ValueError('tier indices must lie in 0..10')
    sides = sorted(set(map(int, sides)))
    if len(sides) != 11 or sides[0] < 1:
        raise ValueError('expected eleven positive frozen sides')
    applied = np.maximum(tiers, floor)

    def chunk(first):
        values = []
        for i in range(first, min(first+16, n)):
            im = data.image(i)
            if im.mode != 'RGB':
                raise ValueError('source dataset must supply RGB images')
            w, h = im.size
            q = float(TIERS[applied[i]])
            hh = max(1, round(h*math.sqrt(q/100)))
            ww = max(1, round(w*math.sqrt(q/100)))
            possible = [s for s in sides if s*s <= hh*ww]
            if not possible:
                raise ValueError('Frozen smallest canvas exceeds deployment pixel budget')
            if q < 100:
                im = im.resize((ww, hh), Image.Resampling.BICUBIC)
            im = im.resize((max(possible), max(possible)), Image.Resampling.BILINEAR)
            values.append((i, np.array(im, dtype=np.uint8, copy=True)))
        return values

    arrays = [None]*n
    if workers == 1:
        for first in range(0, n, 16):
            for i, value in chunk(first):
                arrays[i] = value
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for values in pool.map(chunk, range(0, n, 16)):
                for i, value in values:
                    arrays[i] = value
    rows = [dict(row) for row in data.rows]
    if ram_factory is None:
        return rows, arrays
    return ram_factory(rows, arrays)


def distinct_effective_teacher_views(floor):
    """Return effective indices and a reconstruction map for floor-clamped views.

    Pairs are (source index, actual applied tier), each generated once. An oracle
    can infer those images and reconstruct all original requests through inverse.
    Same-canvas but DIFFERENT applied-tier views must never be deduplicated.
    """
    floor = np.asarray(floor)
    if floor.ndim != 1 or not np.issubdtype(floor.dtype, np.integer):
        raise ValueError('floor must be an integer vector')
    if ((floor < 0) | (floor > 10)).any():
        raise ValueError('floor indices must lie in 0..10')
    lengths = 11-floor
    offsets = np.r_[0, lengths.cumsum()]
    image_indices = np.repeat(np.arange(len(floor)), lengths)
    effective_tiers = np.concatenate([np.arange(f, 11) for f in floor]) if len(floor) else np.empty(0, int)
    inverse = offsets[:-1, None] + np.maximum(np.arange(11)[None, :]-floor[:, None], 0)
    assert len(image_indices) == len(effective_tiers) == int(lengths.sum())
    return image_indices, effective_tiers, inverse




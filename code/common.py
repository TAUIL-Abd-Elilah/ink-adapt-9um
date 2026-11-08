"""Shared helpers (from the author's First Letters tooling, score_letters_0343.py): map loading and a rank AUC."""
import cv2
import numpy as np
import tifffile


def load(p, shape):
    """Read a .tif / .png probability map as float32, first channel, resized (bilinear) to shape (H, W) if needed."""
    m = tifffile.imread(p) if p.endswith('.tif') else cv2.imread(p, cv2.IMREAD_UNCHANGED)
    m = m.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m


def auc(pos, neg):
    """Mann-Whitney AUC of pos vs neg (each subsampled to <= 200,000 values with a fixed seed)."""
    rng = np.random.default_rng(0)
    pos = rng.choice(pos, min(len(pos), 200000), replace=False); neg = rng.choice(neg, min(len(neg), 200000), replace=False)
    allv = np.concatenate([pos, neg]); r = allv.argsort(kind='mergesort').argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

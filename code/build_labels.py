"""build_labels.py <zeroed render.zarr> <out pack.npz> --maps m1 m2 ... [--depth-centre 32.5] [--min-dist 2.9]
Machine-consensus pseudo-labels on a pool render, A-hn's rule (Warner & Bodill, MIT: m306/self/code/labels.py), MIT teachers only.
valid = pixels >= min-dist mm from the held-out region (dist_mm.npy of prep_pool.py) & CT on all 24 model layers & every teacher > 0.
Each teacher is z-scored over valid px, the z maps averaged and re-z-scored -> fz.
  main : ink = fz >= 2.5 in 8-connected blobs of 0.3-6 mm2; bg = fz <= 0.3 and >= 20 px from ink; rest ignored (255)
  decoy: main translated by (3.5, 3.0) mm, kept only where valid (hallucination control)
Pack (= A-hn packs/306_<r>.npz layout, so their trainer reads it): img (h, w, 24) uint8 = layers 21..44 of the 66-layer render in
DESCENDING order (= v8-in 'rev', the reading order on these renders), clip 200; lab_<v> uint8; org_<v> (n, 2) 64-px tile top-lefts on
a stride-32 grid fully inside valid with >= 20 % labelled; ink_<v> (n,) tile has >= 1 % ink; info (json). img / lab are padded by
64 px (0 / 255) and org is in the padded frame, as A-hn's packs."""
import argparse, json, os, sys
import numpy as np, cv2, zarr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load  # noqa: E402
PX_MM2 = (1000 / 8.64) ** 2; POS_T, NEG_T, MIN_MM2, MAX_MM2, NEAR = 2.5, 0.3, 0.3, 6.0, 20; DECOY_MM = (3.5, 3.0); T = 64


def blobs(m, lo, hi):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    a = st[:, cv2.CC_STAT_AREA] / PX_MM2; keep = (a >= lo) & (a <= hi); keep[0] = False
    return keep[lab]


def integral(m): return np.pad(m.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
def box(I, Y, X, s=T): return I[Y + s, X + s] - I[Y, X + s] - I[Y + s, X] + I[Y, X]


ap = argparse.ArgumentParser(); ap.add_argument('render'); ap.add_argument('out'); ap.add_argument('--maps', nargs='+', required=True)
ap.add_argument('--min-dist', type=float, default=2.9); ap.add_argument('--stride', type=int, default=32)
a = ap.parse_args()
g = zarr.open(a.render, mode='r'); arr = g['0']; N, H, W = arr.shape
s = int(round((N - 1) / 2 - 11.5)); vol = np.asarray(arr[s:s + 24])[::-1]                       # descending = model order
img = np.ascontiguousarray(np.minimum(vol, 200).astype(np.uint8).transpose(1, 2, 0)); del vol
dist = np.load(os.path.join(a.render, 'dist_mm.npy')).astype(np.float32)
valid = (dist >= a.min_dist) & (img.min(-1) > 0)
Z = {}
for p in a.maps:
    m = load(p, (H, W)); valid &= m > 0; Z[os.path.basename(p).split('__')[1]] = m
for k, m in Z.items(): v = m[valid]; Z[k] = (m - v.mean()) / max(v.std(), 1e-6)
f = np.mean(list(Z.values()), 0); v = f[valid]; fz = (f - v.mean()) / max(v.std(), 1e-6); fz[~valid] = 0
pos = blobs((fz >= POS_T) & valid, MIN_MM2, MAX_MM2)
near = cv2.dilate(pos.astype(np.uint8), np.ones((2 * NEAR + 1, 2 * NEAR + 1), np.uint8)) > 0
neg = (fz <= NEG_T) & valid & ~near
main = np.full((H, W), 255, np.uint8); main[neg] = 0; main[pos] = 1
dy, dx = int(round(DECOY_MM[0] * 1000 / 8.64)), int(round(DECOY_MM[1] * 1000 / 8.64))
dec = np.full((H, W), 255, np.uint8); dec[dy:, dx:] = main[:H - dy, :W - dx]; dec[~valid] = 255
Iv = integral(valid); Y, X = np.meshgrid(np.arange(0, H - T + 1, a.stride), np.arange(0, W - T + 1, a.stride), indexing='ij')
Y, X = Y.ravel(), X.ravel(); okv = box(Iv, Y, X) == T * T
PADM = 64                                                              # = A-hn train_self.PADM: context for the 128-px SSR window
arrs, infos = dict(img=np.ascontiguousarray(np.pad(img, ((PADM, PADM), (PADM, PADM), (0, 0))))), {}
for nm, L in (('main', main), ('decoy', dec)):
    Il, Ii = integral(L != 255), integral(L == 1); ok = okv & (box(Il, Y, X) >= 0.2 * T * T)
    nink = box(Ii, Y[ok], X[ok])
    arrs[f'lab_{nm}'] = np.pad(L, PADM, constant_values=255); arrs[f'org_{nm}'] = np.stack([Y[ok] + PADM, X[ok] + PADM], 1).astype(np.int64); arrs[f'ink_{nm}'] = nink >= 0.01 * T * T
    infos[nm] = dict(source=f'{a.render} MIT-consensus pseudo-labels ({nm})', teachers=sorted(Z), maps=a.maps, tiles=int(ok.sum()),
                     ink_tiles=int(arrs[f'ink_{nm}'].sum()), ink_mm2=round(float((L == 1).sum() / PX_MM2), 2),
                     bg_mm2=round(float((L == 0).sum() / PX_MM2), 1), valid_cm2=round(float(valid.sum() / PX_MM2 / 100), 3),
                     ink_blobs=int(cv2.connectedComponents((L == 1).astype(np.uint8))[0] - 1), order=f'render layers {s}..{s + 23} descending',
                     rule=dict(pos_t=POS_T, neg_t=NEG_T, blob_mm2=[MIN_MM2, MAX_MM2], near_px=NEAR, decoy_mm=DECOY_MM, min_dist_mm=a.min_dist))
np.savez(a.out + '.tmp.npz', info=json.dumps(infos), **arrs); os.replace(a.out + '.tmp.npz', a.out)
np.save(a.out.replace('.npz', '_fz.npy'), fz.astype(np.float16))
print(json.dumps(infos['main'])); print('decoy tiles', infos['decoy']['tiles'], 'ink tiles', infos['decoy']['ink_tiles'])

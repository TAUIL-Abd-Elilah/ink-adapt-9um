"""score_t1.py <maps dir> <letter_boxes.json> --spec spec.json [--depth +0] [--n-boot 1000] [--out res.json]
Pre-registered T1 score (PHerc0343 letters, Nieuwlaar's boxes on render66). Per map: z-score over valid px (> 0) as
`_fl/route24/score_letters_0343.py`; AUC of in-box px vs same-row out-of-box px inside the text span. Ensembles = mean of z maps.
Uncertainty: each out-of-box px is assigned to the nearest box of its row (by u); bootstrap resamples the 18 letters with
replacement (fixed per-letter px subsample 2000 in / 4000 out, seed 0), paired across maps; 95 % percentile CI of AUC and of
each listed difference. spec.json: {"maps": {"name": [model keys]}, "diffs": [["a", "b"], ...]} (diff = a - b)."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, auc  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument('maps'); ap.add_argument('boxes'); ap.add_argument('--spec', required=True)
ap.add_argument('--depth', default='+0'); ap.add_argument('--n-boot', type=int, default=1000); ap.add_argument('--out', default=None)
ap.add_argument('--crop-u0', type=int, default=400); ap.add_argument('--crop-v0', type=int, default=400)
a = ap.parse_args(); spec = json.load(open(a.spec))
files = {os.path.basename(p).split('__')[1]: p for p in glob.glob(os.path.join(a.maps, f'*__d{a.depth}_reverse.*')) if p.endswith(('.tif', '.png'))}
import tifffile
shape = tifffile.imread(next(p for p in files.values() if p.endswith('.tif'))).shape
L = json.load(open(a.boxes))['letters']; B = []
for l in L:
    u0, v0, u1, v1 = [int(round(x)) for x in l['box']]; B.append((u0 - a.crop_u0, v0 - a.crop_v0, u1 - a.crop_u0, v1 - a.crop_v0, l['row']))
inside = np.full(shape, -1, np.int32); rowm = {}
for i, (u0, v0, u1, v1, r) in enumerate(B): inside[max(v0, 0):v1, max(u0, 0):u1] = i
us = [b[0] for b in B] + [b[2] for b in B]; span = np.zeros(shape, bool); span[:, max(min(us), 0):max(us)] = True
owner = np.full(shape, -1, np.int32)                      # out-of-box px -> nearest box of the same row, by u-distance of box centre
uu = np.arange(shape[1])
for r in sorted(set(b[4] for b in B)):
    idx = [i for i, b in enumerate(B) if b[4] == r]
    rows = np.zeros(shape[0], bool)
    for i in idx: rows[max(B[i][1], 0):B[i][3]] = True
    cen = np.array([(B[i][0] + B[i][2]) / 2 for i in idx]); near = np.array(idx)[np.abs(uu[:, None] - cen[None]).argmin(1)]
    m = rows[:, None] & span & (inside < 0); owner[m] = np.broadcast_to(near[None], shape)[m]
Z = {}
def zmap(k):
    if k not in Z:
        m = load(files[k], shape); v = m > 0; z = (m - m[v].mean()) / (m[v].std() + 1e-6); z[~v] = np.nan; Z[k] = z
    return Z[k]
maps = {n: (np.nanmean(np.stack([zmap(k) for k in ks]), 0) if len(ks) > 1 else zmap(ks[0])) for n, ks in spec['maps'].items()}
rng = np.random.default_rng(0); pin, pout = [], []
for i in range(len(B)):
    yi, xi = np.nonzero(inside == i); yo, xo = np.nonzero(owner == i)
    si = rng.choice(len(yi), min(2000, len(yi)), replace=False); so = rng.choice(len(yo), min(4000, len(yo)), replace=False)
    pin.append((yi[si], xi[si])); pout.append((yo[so], xo[so]))
def vals(z, letters):
    vi = np.concatenate([z[pin[i]] for i in letters]); vo = np.concatenate([z[pout[i]] for i in letters])
    return vi[np.isfinite(vi)], vo[np.isfinite(vo)]
full = {n: auc(*[np.concatenate([z[inside >= 0]], 0)[np.isfinite(z[inside >= 0])], z[(owner >= 0)][np.isfinite(z[owner >= 0])]]) for n, z in maps.items()}
boot = {n: [] for n in maps}
for b in range(a.n_boot):
    lt = rng.integers(0, len(B), len(B))
    for n, z in maps.items(): boot[n].append(auc(*vals(z, lt)))
res = dict(depth=a.depth, n_letters=len(B), n_boot=a.n_boot, maps={}, diffs={})
for n in maps:
    bb = np.array(boot[n]); res['maps'][n] = dict(auc=round(full[n], 4), ci95=[round(float(np.percentile(bb, 2.5)), 4), round(float(np.percentile(bb, 97.5)), 4)], models=spec['maps'][n])
for x, y in spec.get('diffs', []):
    d = np.array(boot[x]) - np.array(boot[y])
    res['diffs'][f'{x} - {y}'] = dict(diff=round(full[x] - full[y], 4), ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], p_le0=round(float((d <= 0).mean()), 4))
print(json.dumps(res, indent=1))
if a.out: json.dump(res, open(a.out, 'w'), indent=1)

"""prep_pool.py <render.zarr> <mesh.tifxyz dir> <excl_nodes.npy> <out.zarr> [--zero-mm 2.2]
Exclusion hygiene for a pool render (A-hn's rule, Warner & Bodill, MIT): every render pixel gets its 3-D distance (mm) to the
nearest excluded node (the known-answer region, given as an (n, 3) xyz array in voxels); CT is zeroed on ALL layers where the
distance is < zero-mm, so no teacher or student ever sees CT from the held-out region or its buffer.  Pixel distance = min over
the 2 x 2 mesh nodes of its grid cell (conservative), 20 px per node (mesh scale 0.05).
Writes <out.zarr>/0 (same shape, uint8) and <out.zarr>/dist_mm.npy (float16, inf where no mesh)."""
import argparse, json, os
import numpy as np, tifffile, zarr
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument('render'); ap.add_argument('mesh'); ap.add_argument('excl'); ap.add_argument('out')
ap.add_argument('--zero-mm', type=float, default=2.2); ap.add_argument('--vox-um', type=float, default=8.64)
a = ap.parse_args()
X = np.stack([tifffile.imread(f'{a.mesh}/{c}.tif').astype(np.float64) for c in 'xyz'], -1); v = X[..., 2] > 0
step = int(round(1 / json.load(open(f'{a.mesh}/meta.json'))['scale'][0]))
E = np.load(a.excl); d = np.full(v.shape, np.inf); d[v] = cKDTree(E).query(X[v])[0] * a.vox_um / 1000
dp = np.pad(d, ((0, 1), (0, 1)), constant_values=np.inf)
cell = np.minimum(np.minimum(dp[:-1, :-1], dp[1:, :-1]), np.minimum(dp[:-1, 1:], dp[1:, 1:]))   # cell (i, j) = nodes i..i+1, j..j+1
src = zarr.open(a.render, mode='r'); arr = src['0'] if not hasattr(src, 'shape') else src
N, H, W = arr.shape
iy = np.minimum(np.arange(H) // step, cell.shape[0] - 1); ix = np.minimum(np.arange(W) // step, cell.shape[1] - 1)
dpx = cell[iy[:, None], ix[None, :]].astype(np.float16)
g = zarr.open_group(a.out, mode='w', zarr_format=2); o = g.create_array('0', shape=arr.shape, chunks=(N, 128, 128), dtype='u1', overwrite=True)
zero = dpx < a.zero_mm
for y0 in range(0, H, 512):
    blk = np.asarray(arr[:, y0:y0 + 512]); blk[:, zero[y0:y0 + 512]] = 0; o[:, y0:y0 + 512] = blk
g.attrs.update(dict(src=a.render, excl=a.excl, zero_mm=a.zero_mm, rule='A-hn exclusion: CT zeroed < zero_mm, pool tiles >= 2.9 mm'))
np.save(os.path.join(a.out, 'dist_mm.npy'), dpx)
print(f'{a.render}: {N}x{H}x{W}, zeroed {zero.mean() * 100:.1f} % of px, >= 2.9 mm {(dpx >= 2.9).mean() * 100:.1f} %, min dist {np.nanmin(d):.2f} mm')

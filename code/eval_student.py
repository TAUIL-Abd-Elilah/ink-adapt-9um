"""eval_student.py <render.zarr> <out dir> <tag> <student dir> [<student dir> ...] [--d 0] [--stride 21]
Runs v8-in's own predict.py (MIT, Youssef Nader) with each student's weights on layers 21+d..44+d of a 66-layer render, reversed
(= the order the students were trained in), and writes <out>/<render name>__<tag>__d<d>_reverse.tif = mean probability over the
students (float32), plus one map per student (<tag>s<i>)."""
import argparse, os, shutil, subprocess, sys, tempfile
import numpy as np, tifffile, zarr
V8 = os.environ.get('V8_DIR', 'D:/Competition/Vesuvius progress prizes/_fl/ckpt/v8in')
ap = argparse.ArgumentParser(); ap.add_argument('render'); ap.add_argument('out'); ap.add_argument('tag'); ap.add_argument('students', nargs='+')
ap.add_argument('--d', type=int, default=0); ap.add_argument('--stride', type=int, default=21); ap.add_argument('--device', default=None)
ap.add_argument('--forward', action='store_true', help='depth-order control: feed layers in render order (not reversed)')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
g = zarr.open(a.render, mode='r'); arr = g['0']; N = arr.shape[0]; s = int(round((N - 1) / 2 + a.d - 11.5))
name = os.path.basename(a.render.rstrip('/\')).replace('.zarr', '')
t = tempfile.mkdtemp(dir=a.out)
for i in range(24): tifffile.imwrite(os.path.join(t, f'{i:02d}.tif'), np.asarray(arr[s + i]))
maps = []
for i, sd in enumerate(a.students):
    o = os.path.join(t, f's{i}.npy')
    cmd = [sys.executable, os.path.join(V8, 'predict.py'), '--layers', t, '--output', o, '--model', sd, '--no-reverse' if a.forward else '--reverse',
           '--batch-size', '16', '--stride', str(a.stride)] + (['--device', a.device] if a.device else [])
    r = subprocess.run(cmd, capture_output=True, text=True); print(sd, 'rc', r.returncode, (r.stdout + r.stderr)[-300:].replace('\n', ' | '), flush=True)
    if r.returncode: sys.exit(r.returncode)
    m = np.load(o).astype(np.float32); maps.append(m)
    tifffile.imwrite(os.path.join(a.out, f'{name}__{a.tag}s{i}__d{a.d:+d}_reverse.tif'), m)
tifffile.imwrite(os.path.join(a.out, f'{name}__{a.tag}__d{a.d:+d}_reverse.tif'), np.mean(maps, 0).astype(np.float32))
shutil.rmtree(t, ignore_errors=True); print('wrote', a.tag, len(maps), 'students')

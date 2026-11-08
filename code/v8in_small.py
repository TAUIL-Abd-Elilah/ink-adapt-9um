"""v8in_small.py <render.zarr> <out.png> [--batch-size 4] [--cap-mib 20000] : the reverse-order v8-in map of one 66-layer render at depth d (default 0),
as _fl/route24/v8in_on_render.py makes it (the same 24 layers centred on layer 32.5, the same predict.py), but with a smaller
inference batch so it fits beside another session's job. Each tile is predicted independently in eval mode and the overlap
average is done on the CPU, so the batch size does not change the computation. It stops its predict.py child if total dedicated VRAM
passes --cap-mib (exit 3)."""
import argparse, os, shutil, subprocess, sys, tempfile, time
import numpy as np, tifffile, zarr

P = os.environ.get('FLS_ROOT', 'D:/Competition/Vesuvius progress prizes')
ap = argparse.ArgumentParser(); ap.add_argument('render'); ap.add_argument('out')
ap.add_argument('--batch-size', type=int, default=4); ap.add_argument('--cap-mib', type=int, default=20000)
ap.add_argument('--layers', type=int, default=66); ap.add_argument('--d', type=int, default=0); ap.add_argument('--model', default=f'{P}/_fl/ckpt/v8in')
a = ap.parse_args()
z = zarr.open(a.render, mode='r'); arr = z['0'] if not hasattr(z, 'shape') else z
s = int(round((a.layers - 1) / 2 + a.d - 11.5))
t = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
for i in range(24):
    tifffile.imwrite(os.path.join(t, f'{i:02d}.tif'), np.asarray(arr[s + i]))
cmd = [sys.executable, os.path.join(a.model, 'predict.py'), '--layers', t, '--output', a.out, '--model', a.model,
       '--reverse', '--batch-size', str(a.batch_size)]
p = subprocess.Popen(cmd); peak = 0; rc = None
while rc is None:
    time.sleep(5); rc = p.poll()
    try:
        used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        continue
    peak = max(peak, used)
    if rc is None and used > a.cap_mib:
        p.kill(); p.wait(); print(f'v8in_small: VRAM {used} MiB > cap {a.cap_mib}: stopped', flush=True)
        shutil.rmtree(t, ignore_errors=True); sys.exit(3)
shutil.rmtree(t, ignore_errors=True)
print(f'v8in_small: rc {rc}, peak total VRAM {peak} MiB', flush=True); sys.exit(rc)

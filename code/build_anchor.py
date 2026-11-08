"""build_anchor.py <HF 1447 dir> <packs dir> w058 w060 : human-label anchor packs (PHerc1447, Youssef Nader's labels, CC BY-NC 4.0)
with A-hn's src_1447 (organisers' tile grid, reversed depth order) -> packs/a_<w>.npz (img, lab float16, keep, org, group, info)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ahn'))
os.environ.setdefault('V8_DIR', 'D:/Competition/Vesuvius progress prizes/_fl/ckpt/v8in'); sys.path.insert(0, os.environ['V8_DIR'])
import v8306lib as L  # noqa: E402
L.D1447 = sys.argv[1]; PK = sys.argv[2]; os.makedirs(PK, exist_ok=True)
for w in sys.argv[3:]:
    d = L.src_1447(w); out = f'{PK}/a_{w}.npz'
    np.savez(out + '.tmp.npz', img=d['img'], lab=d['lab'].astype(np.float16), keep=d['keep'].astype(np.uint8), org=d['org'],
             group='1447', info=json.dumps(d['info'])); os.replace(out + '.tmp.npz', out)
    print(w, d['info']['tiles'], 'tiles', d['img'].shape, flush=True)

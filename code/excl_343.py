# excluded nodes for T1 = every node of Nieuwlaar's concat mesh inside one of the 18 letter boxes (xyz voxels)
import json, numpy as np, tifffile
M = 'D:/VesuviusCTCache/p343_calib/mesh/'; C = np.stack([tifffile.imread(M + c + '.tif').astype(np.float64) for c in 'xyz'], -1)
val = C[..., 2] > 0; ins = np.zeros(val.shape, bool)
for l in json.load(open(M + 'concat_w047-w048_R5B2_z9500-11000_letter_boxes.json'))['letters']:
    u0, v0, u1, v1 = l['box']; ins[int(v0 // 20):int(np.ceil(v1 / 20)) + 1, int(u0 // 20):int(np.ceil(u1 / 20)) + 1] = True
np.save('C:/VesuviusTemp/nov_ink/t1_343/excl_nodes.npy', C[ins & val]); print((ins & val).sum(), 'excluded nodes')

#!/bin/bash
# waits for the pool renders (file poll), then exclusion prep + CPU teachers (Reader v3, Reader v2) on the zeroed renders
N=C:/VesuviusTemp/nov_ink; T=$N/t1_343; PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe; CODE=$N/code
export TMP=$N/tmp TEMP=$N/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
until [ -f $T/renders/render.done ]; do sleep 120; done
mkdir -p $T/pool_maps
for w in w047 w048; do
  [ -f $T/renders/${w}_z.zarr/dist_mm.npy ] || $PY $CODE/prep_pool.py $T/renders/$w.zarr $T/meshes/${w}_R5B2_z9500-11000 $T/excl_nodes.npy $T/renders/${w}_z.zarr || exit 2
done
cd "/d/Competition/Vesuvius progress prizes/_fl/route24"
for w in w047 w048; do
  $PY run_ink_render_fast.py $T/renders/${w}_z.zarr $T/pool_maps --models reader_v3 reader_v2 --depths 0 --reverse-only --batch 2 || exit 3
done
echo done > $T/pool_maps/cpu_teachers.done

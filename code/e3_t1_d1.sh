#!/bin/bash
# NOVINK E3 T1 add-on (Amendment 7): v8-in reverse map of render66 at d+1, so teach4 exists at d = +1 for the secondary measure.
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe; REPO="D:/Competition/Vesuvius progress prizes/_public/ink-adapt-9um"; CODE="$REPO/code"
T=C:/VesuviusTemp/nov_ink/t1_343; LOG=$T/t1_gpu_steps.log; R66=D:/VesuviusCTCache/p343_calib/render66.zarr
OUT=$T/eval_maps/render66__v8inREV__d+1_reverse.png
[ -f $OUT ] && { echo "== $(date '+%F %T') d1 add-on: already there" >> $LOG; exit 0; }
$PY "$CODE/gpu_guard.py" >> $LOG 2>&1; echo "== $(date '+%F %T') v8in render66 d+1" >> $LOG
for att in 1 2 3; do
  $PY "$CODE/v8in_small.py" $R66 $OUT --d 1 >> $LOG 2>&1 && break
  echo "== v8in d+1 attempt $att failed" >> $LOG; $PY "$CODE/gpu_guard.py" >> $LOG 2>&1
done
for m in hecate__d+1_reverse.png reader_v2__d+1_reverse.tif reader_v3__d+1_reverse.tif; do cp -n $T/maps/render66__$m $T/eval_maps/; done
echo "== $(date '+%F %T') d1 add-on END" >> $LOG; echo done > $T/T1_D1.done

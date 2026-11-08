#!/bin/bash
# E3 T1 (PHerc0343) GPU job: GPU teachers on the zeroed pool renders -> pseudo-labels -> 3 students + decoy -> eval on render66.
# Queued only via _fl/route24/queue_head.py (NOVINK). Always writes C:/VesuviusTemp/nov_ink/t1_343/T1_GPU.done (rc). Resumable.
set -u
N=C:/VesuviusTemp/nov_ink; T=$N/t1_343; PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe; R24="D:/Competition/Vesuvius progress prizes/_fl/route24"
REPO="D:/Competition/Vesuvius progress prizes/_public/ink-adapt-9um"; CODE="$REPO/code"; CFG="$REPO/configs"
export TMP=$N/tmp TEMP=$N/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=$T/t1_gpu_steps.log; fin() { echo "rc=$1 $(date)" > $T/T1_GPU.done; echo "END rc=$1 $(date)" >> $LOG; exit $1; }
step() { echo "== $(date '+%F %T') $*" >> $LOG; }
guard() { $PY "$CODE/gpu_guard.py" >> $LOG 2>&1; }
rm -f $T/T1_GPU.done; step "start"
for w in w047 w048; do for m in reader_v3 reader_v2; do [ -f $T/pool_maps/${w}_z__${m}__d+0_reverse.tif ] || { step "missing CPU teacher $w $m"; fin 2; }; done; done
cd "$R24"
for w in w047 w048; do
  if [ ! -f $T/pool_maps/${w}_z__hecate__d+0_reverse.png ]; then guard; step "hecate $w"; $PY run_ink_render_fast.py $T/renders/${w}_z.zarr $T/pool_maps --models hecate --depths 0 --reverse-only >> $LOG 2>&1 || fin 3; fi
  if [ ! -f $T/pool_maps/${w}_z__v8inREV__d+0_reverse.png ]; then guard; step "v8in $w"; $PY v8in_on_render.py $T/renders/${w}_z.zarr $T/pool_maps --depths 0 --orders rev >> $LOG 2>&1 || fin 4
    mv $T/pool_maps/${w}_z__v8in__d+0_rev.png $T/pool_maps/${w}_z__v8inREV__d+0_reverse.png || fin 4; fi
done
mkdir -p $T/self/packs
for w in w047 w048; do
  [ -f $T/self/packs/pool_$w.npz ] || { step "labels $w"; $PY "$CODE/build_labels.py" $T/renders/${w}_z.zarr $T/self/packs/pool_$w.npz --maps $T/pool_maps/${w}_z__v8inREV__d+0_reverse.png $T/pool_maps/${w}_z__reader_v3__d+0_reverse.tif $T/pool_maps/${w}_z__reader_v2__d+0_reverse.tif $T/pool_maps/${w}_z__hecate__d+0_reverse.png >> $LOG 2>&1 || fin 5; }
done
[ -f $T/self/packs/a_w060.npz ] || { step anchors; $PY "$CODE/build_anchor.py" $N/data/hf1447 $T/self/packs w058 w060 >> $LOG 2>&1 || fin 6; }
export SELF_ROOT=$T/self V8_DIR="D:/Competition/Vesuvius progress prizes/_fl/ckpt/v8in" POOL_PREFIX=pool_
if [ ! -f $T/self/runs/t1_s0/done.json ]; then guard; step "bench t1_s0 (3 steps, peak VRAM)"; $PY "$CODE/ahn/train_cloud_hn.py" "$CFG/t1_s0.json" --bench 3 >> $LOG 2>&1; step "bench rc=$?"; fi
for c in t1_s0 t1_s1 t1_s2 t1_decoy; do
  for att in 1 2 3 4 5 6; do
    [ -f $T/self/runs/$c/done.json ] && break
    guard; step "train $c attempt $att"
    $PY "$CODE/ahn/train_cloud_hn.py" "$CFG/$c.json" --max-minutes 120 >> $LOG 2>&1 & tp=$!
    $PY "$CODE/vram_watch.py" $tp >> $LOG 2>&1 & wp=$!
    wait $tp; rc=$?; kill $wp 2>/dev/null; step "train $c attempt $att rc=$rc"
  done
  [ -f $T/self/runs/$c/done.json ] || { step "train $c failed"; fin 7; }
done
mkdir -p $T/eval_maps $T/eval_fwd; cp $T/maps/render66__{hecate__d+0_reverse.png,reader_v2__d+0_reverse.tif,reader_v3__d+0_reverse.tif,v8inREV__d+0_reverse.png} $T/eval_maps/
R66=D:/VesuviusCTCache/p343_calib/render66.zarr
guard; step "eval students"
[ -f $T/eval_maps/render66__E3T1__d+0_reverse.tif ] || $PY "$CODE/eval_student.py" $R66 $T/eval_maps E3T1 $T/self/runs/t1_s{0,1,2}/final >> $LOG 2>&1 || fin 8
[ -f $T/eval_maps/render66__E3T1decoy__d+0_reverse.tif ] || $PY "$CODE/eval_student.py" $R66 $T/eval_maps E3T1decoy $T/self/runs/t1_decoy/final >> $LOG 2>&1 || fin 8
[ -f $T/eval_fwd/render66__E3T1fwd__d+0_reverse.tif ] || $PY "$CODE/eval_student.py" $R66 $T/eval_fwd E3T1fwd $T/self/runs/t1_s{0,1,2}/final --forward >> $LOG 2>&1 || fin 8
step "eval d+1"; [ -f $T/eval_maps/render66__E3T1__d+1_reverse.tif ] || $PY "$CODE/eval_student.py" $R66 $T/eval_maps E3T1 $T/self/runs/t1_s{0,1,2}/final --d 1 >> $LOG 2>&1 || fin 8
fin 0

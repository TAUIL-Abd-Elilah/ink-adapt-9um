set -e
cd "/d/Competition/Vesuvius progress prizes/_fl/route24"
export ROUTE24_CACHE=C:/VesuviusTemp/nov_ink/ctcache ROUTE24_CAP_GB=15 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMP=C:/VesuviusTemp/nov_ink/tmp TEMP=C:/VesuviusTemp/nov_ink/tmp
for w in w047 w048; do
  C:/Users/PC/miniconda3/envs/vesuvius/python.exe render_tifxyz.py "C:/VesuviusTemp/nov_ink/t1_343/meshes/${w}_R5B2_z9500-11000" PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr C:/VesuviusTemp/nov_ink/t1_343/renders/${w}.zarr --umbilicus D:/VesuviusCTCache/p343_calib/mesh/umbilicus_PHerc0343.json --layers 66
  echo "$w render done $(date)"
done
echo ALL_DONE > C:/VesuviusTemp/nov_ink/t1_343/renders/render.done

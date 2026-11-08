"""gpu_guard.py [--max-mib 4000] [--poll 60] : block until the GPU is free enough to start a job, i.e. no other session's
`train.py` is running and dedicated VRAM in use is below --max-mib (one GPU job at a time; never spill into shared memory).
Prints one line when it starts waiting and one when it releases."""
import argparse, os, subprocess, time
import psutil
ap = argparse.ArgumentParser(); ap.add_argument('--max-mib', type=int, default=4000); ap.add_argument('--poll', type=int, default=60)
a = ap.parse_args(); me = os.getpid(); t0 = time.time(); said = False
def state():
    used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
    foreign = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            c = ' '.join(p.info['cmdline'] or [])
            if p.info['pid'] != me and 'python' in (p.info['name'] or '').lower() and 'train.py' in c and 'train_cloud_hn.py' not in c: foreign.append(p.info['pid'])
        except Exception: pass
    return used, foreign
while True:
    used, foreign = state()
    if used < a.max_mib and not foreign: break
    if not said: print(f'gpu_guard: waiting (VRAM {used} MiB, foreign train.py {foreign})', flush=True); said = True
    time.sleep(a.poll)
print(f'gpu_guard: free after {time.time() - t0:.0f} s (VRAM {used} MiB)', flush=True)

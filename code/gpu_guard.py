"""gpu_guard.py [--max-mib 4000] [--poll 60] : block until the GPU is free enough to start a job, i.e. dedicated VRAM in use is below --max-mib
(the user's rule allows only a small job, <= 4 GB, beside ours; a foreign trainer is logged, and vram_watch.py stops ours if the
total passes 22.5 GB). Formerly it also required that no other session's
training script (*train*.py) is running and dedicated VRAM in use is below --max-mib (one GPU job at a time; never spill into shared memory).
Prints one line when it starts waiting and one when it releases."""
import argparse, os, subprocess, time
import psutil


def is_foreign_trainer(cmdline):
    """Another session's training script: any *.py argument whose file name contains 'train', except our own trainer."""
    import os
    for arg in cmdline or []:
        b = os.path.basename(arg.replace(chr(92), '/'))
        if b.endswith('.py') and 'train' in b and b != 'train_cloud_hn.py':
            return True
    return False

def is_foreign_chain(p):
    """Amendment 5: another session's chain shell (bash running scratch/queue*.sh from D:/Competition/wundernn), which starts its next
    training run a few seconds after the last one ends. While it lives, the GPU is not free for a student."""
    if 'bash' not in (p.info['name'] or '').lower(): return False
    c = ' '.join(p.info['cmdline'] or []).replace(chr(92), '/')
    if 'queue' not in c or '.sh' not in c: return False
    try: cwd = p.cwd().replace(chr(92), '/').lower()
    except Exception: return False
    return cwd.startswith('d:/competition/wundernn')

ap = argparse.ArgumentParser(); ap.add_argument('--max-mib', type=int, default=4000); ap.add_argument('--poll', type=int, default=60)
a = ap.parse_args(); me = os.getpid(); t0 = time.time(); said = False
def state():
    used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
    foreign = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            c = ' '.join(p.info['cmdline'] or [])
            if p.info['pid'] != me and 'python' in (p.info['name'] or '').lower() and is_foreign_trainer(p.info['cmdline']): foreign.append(p.info['pid'])
            elif is_foreign_chain(p): foreign.append(f"chain {p.info['pid']}")
        except Exception: pass
    return used, foreign
quiet = 0
while True:
    used, foreign = state()
    quiet = quiet + 1 if (used < a.max_mib and not foreign) else 0
    if quiet >= 2: break                           # Amendment 4: free and no foreign trainer on two polls in a row (other sessions chain runs within ~20 s)
    if quiet == 0 and not said: print(f'gpu_guard: waiting (VRAM {used} MiB, foreign train.py {foreign})', flush=True); said = True
    time.sleep(a.poll)
print(f'gpu_guard: free after {time.time() - t0:.0f} s (VRAM {used} MiB, foreign trainers {foreign})', flush=True)

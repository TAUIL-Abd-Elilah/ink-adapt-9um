"""vram_watch_win.py [--limit-mib 21000] [--done FILE] : Amendment 4. The job script's vram_watch.py was given Git Bash's $! (an MSYS pid,
not a Windows pid), so it exited at once and never guarded anything. This watcher runs beside the job and finds our trainer by its
command line (train_cloud_hn.py). If a foreign trainer runs and dedicated VRAM passes --limit-mib, it kills our trainer and its children.
The trainer resumes from state.pt on the job's next attempt. It exits when --done appears."""
import argparse, os, subprocess, sys, time
import psutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_guard_fn import is_foreign_trainer

ap = argparse.ArgumentParser(); ap.add_argument('--limit-mib', type=int, default=21000)
ap.add_argument('--done', default='C:/VesuviusTemp/nov_ink/t1_343/T1_GPU.done'); a = ap.parse_args()
print(time.strftime('%F %T'), 'vram_watch_win: start', flush=True)
while not os.path.exists(a.done):
    try:
        used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.split()[0])
        ours, foreign = [], []
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' not in (p.info['name'] or '').lower(): continue
            cl = p.info['cmdline'] or []
            if any(os.path.basename(x.replace(chr(92), '/')) == 'train_cloud_hn.py' for x in cl): ours.append(p)
            elif is_foreign_trainer(cl): foreign.append(p.info['pid'])
        if used > a.limit_mib and foreign and ours:
            for p in ours:
                for k in p.children(recursive=True) + [p]:
                    try: k.kill()
                    except Exception: pass
            print(time.strftime('%F %T'), f'vram_watch_win: killed our trainer {[p.pid for p in ours]} (VRAM {used} MiB, foreign {foreign})', flush=True)
    except Exception as e:
        print('vram_watch_win error', e, flush=True)
    time.sleep(15)
print(time.strftime('%F %T'), 'vram_watch_win: job done, exit', flush=True)

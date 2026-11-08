"""vram_watch.py <pid> [--limit-mib 22500] : while <pid> lives, kill it (and its children) if dedicated VRAM passes the limit while
another session's train.py runs (two jobs would freeze each other, 7 Nov). The trainer resumes from its state.pt on restart."""
import argparse, subprocess, sys, time
import psutil
ap = argparse.ArgumentParser(); ap.add_argument('pid', type=int); ap.add_argument('--limit-mib', type=int, default=22500)
a = ap.parse_args()
while psutil.pid_exists(a.pid):
    try:
        used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
        foreign = [p.pid for p in psutil.process_iter(['name', 'cmdline']) if 'python' in (p.info['name'] or '').lower()
                   and 'train.py' in ' '.join(p.info['cmdline'] or []) and 'train_cloud_hn.py' not in ' '.join(p.info['cmdline'] or [])]
        if used > a.limit_mib and foreign:
            p = psutil.Process(a.pid); kids = p.children(recursive=True)
            for k in kids + [p]:
                try: k.kill()
                except Exception: pass
            print(f'vram_watch: killed {a.pid} (VRAM {used} MiB, foreign {foreign})', flush=True); sys.exit(9)
    except psutil.NoSuchProcess: break
    except Exception as e: print('vram_watch error', e, flush=True)
    time.sleep(20)

"""Shared helper for vram_watch_win.py (same rule as gpu_guard.py and vram_watch.py)."""
import os


def is_foreign_trainer(cmdline):
    """Another session's training script: any *.py argument whose file name contains 'train', except our own trainer."""
    for arg in cmdline or []:
        b = os.path.basename(arg.replace(chr(92), '/'))
        if b.endswith('.py') and 'train' in b and b != 'train_cloud_hn.py':
            return True
    return False

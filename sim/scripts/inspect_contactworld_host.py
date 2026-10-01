#!/usr/bin/env python3
"""Read-only server preflight. Does not install packages or require root."""
import json
import os
import platform
import shutil
import subprocess


def command(args):
    if not shutil.which(args[0]):
        return {'available': False}
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'available': True, 'returncode': p.returncode,
                'stdout': p.stdout[-10000:], 'stderr': p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {'available': True, 'timeout': True}


report = {
    'system': platform.platform(),
    'python': platform.python_version(),
    'user_home_writable': os.access(os.path.expanduser('~'), os.W_OK),
    'free_disk_gib_here': round(shutil.disk_usage('.').free / 2**30, 1),
    'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
    'conda': command(['conda', '--version']),
    'vulkan': command(['vulkaninfo', '--summary']),
    'compiler': command(['gcc', '--version']),
    'note': 'Missing vulkaninfo alone does not prove missing Vulkan support. A Gym camera probe is still required.',
}
print(json.dumps(report, indent=2))

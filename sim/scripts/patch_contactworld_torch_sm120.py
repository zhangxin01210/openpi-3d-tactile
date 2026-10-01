#!/usr/bin/env python3
"""Apply the minimal sm120 architecture parsing changes to Torch v2.4.1."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
args = p.parse_args()

def replace(relative, old, new):
    path = args.source / relative
    text = path.read_text()
    if new in text and old not in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one matching source fragment: {relative}')
    path.write_text(text.replace(old, new))

path = args.source / 'cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake'
lines = path.read_text().splitlines(keepends=True)
matches = [i for i, s in enumerate(lines) if 'if(arch_name MATCHES "^(' in s]
assert len(matches) == 1
i = matches[0]
lines[i] = lines[i].replace('[0-9]', '[0-9]+') if '[0-9]+' not in lines[i] else lines[i]
path.write_text(''.join(lines))
replace('cmake/public/utils.cmake',
        'set(TORCH_CUDA_ARCH_LIST TORCH_CUDA_ARCH_LIST ${CUDA_ARCH_NAME})',
        'set(TORCH_CUDA_ARCH_LIST ${TORCH_CUDA_ARCH_LIST} ${CUDA_ARCH_NAME})')
replace('torch/utils/cpp_extension.py', "'8.9', '9.0', '9.0a']", "'8.9', '9.0', '9.0a', '12.0']")
replace('torch/utils/cpp_extension.py', 'num = arch[0] + arch[2:].split("+")[0]',
        'num = arch.split("+")[0].replace(".", "")')
print('Minimal architecture patches applied; attention kernels not patched.')

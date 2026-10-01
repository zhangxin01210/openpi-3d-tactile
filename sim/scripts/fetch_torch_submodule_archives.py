#!/usr/bin/env python3
"""Fetch exact Torch submodule commits through direct source archives.

For a dedicated build checkout only. Existing mismatched module directories are
preserved under backup. Recursive gitlinks are resolved from the publisher API.
"""
import argparse
import concurrent.futures
import configparser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--backup', type=Path, required=True)
args = p.parse_args()
args.cache.mkdir(parents=True, exist_ok=True)
args.backup.mkdir(parents=True, exist_ok=True)

def request(url):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        urllib.request.Request(url, headers={'User-Agent': 'contactworld-build-audit'}), timeout=30)

def module_urls(root, parent_url=None):
    cfg = configparser.ConfigParser()
    cfg.read(root/'.gitmodules')
    result = {}
    for section in cfg.sections():
        rel, url = cfg[section]['path'], cfg[section]['url']
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Unsafe submodule path: '+rel)
        if url.startswith('.'):
            url = urllib.parse.urljoin(parent_url.rstrip('/')+'/', url)
        result[rel] = url.removesuffix('.git')
    return result

def install(rel, url, commit):
    dest = args.source/rel
    key = rel.replace('/', '__')+'-'+commit
    receipt = args.cache/(key+'.json')
    if receipt.exists() and dest.is_dir():
        return json.loads(receipt.read_text())
    if url.startswith('https://github.com/'):
        repo = url.removeprefix('https://github.com/')
        archive_url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
    elif url == 'https://gitlab.com/libeigen/eigen':
        repo = None
        archive_url = f'{url}/-/archive/{commit}/eigen-{commit}.tar.gz'
    else:
        raise ValueError('Unsupported publisher URL: '+url)
    archive = args.cache/(key+'.tar.gz')
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        with request(archive_url) as response, partial.open('wb') as f:
            shutil.copyfileobj(response, f)
        partial.replace(archive)
    saved = args.backup/rel
    # A failed nested download can leave a fully installed parent archive.
    # Directory rename follows full extraction, so this state is resumable.
    if not (dest.is_dir() and saved.exists() and not (dest/'.git').exists()):
        unpack = args.cache/(key+'.unpack')
        unpack.mkdir(exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(unpack, filter='data')
        roots = list(unpack.iterdir())
        assert len(roots) == 1
        if dest.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            if saved.exists():
                raise RuntimeError('Backup already exists: '+str(saved))
            shutil.move(str(dest), str(saved))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots[0]), str(dest))
        unpack.rmdir()
    children = []
    if (dest/'.gitmodules').exists():
        urls = module_urls(dest, url)
        if urls:
            if repo is None:
                raise RuntimeError('Nested GitLab submodules require a tree provider')
            with request(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1') as response:
                tree = json.load(response)
            if tree.get('truncated'):
                raise RuntimeError('Truncated Git tree for '+url)
            pins = {x['path']: x['sha'] for x in tree['tree'] if x['mode'] == '160000'}
            for subpath, suburl in urls.items():
                if subpath not in pins:
                    # Match git submodule behavior for stale .gitmodules entries.
                    children.append({'path': subpath, 'status': 'no gitlink at pinned commit; not fetched'})
                    continue
                children.append(install(rel+'/'+subpath, suburl, pins[subpath]))
    with archive.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    result = {'path': rel, 'url': archive_url, 'commit': commit,
              'bytes': archive.stat().st_size, 'sha256': digest,
              'proxy_used': False, 'children': children}
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'completed': rel, 'bytes': result['bytes']}), flush=True)
    return result

urls = module_urls(args.source)
listing = subprocess.check_output(['git', 'ls-tree', '-r', 'HEAD'], cwd=args.source, text=True)
pins = {line.split('\t')[1]: line.split()[2] for line in listing.splitlines() if line.startswith('160000')}
failures, results = [], []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    tasks = {pool.submit(install, rel, urls[rel], sha): rel for rel, sha in pins.items()}
    for future in concurrent.futures.as_completed(tasks):
        try:
            results.append(future.result())
        except Exception as exc:
            failure = {'path': tasks[future], 'error': str(exc)}
            failures.append(failure)
            print(json.dumps(failure), flush=True)
(args.cache/'manifest.json').write_text(json.dumps({'modules': results, 'failures': failures}, indent=2)+'\n')
raise SystemExit(bool(failures))

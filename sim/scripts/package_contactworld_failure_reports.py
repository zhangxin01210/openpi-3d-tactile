#!/usr/bin/env python3
"""Bundle ContactWorld failure pages and their media for offline viewing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path


REPORT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_action_effect_20261009')
OUTPUT = Path('sim/reports/2026-10-10/contactworld_failure_comparison_offline.zip')
FOLDER = 'contactworld_failure_comparison'
PAGES = ['failure_audit_by_seed.html', 'failure_audit.html'] + [
    f'failure_audit_model{i:02d}.html' for i in range(1, 14)
]


def rgb_name(row: dict) -> str:
    return f'media/rgb/{row["model"]}/seed_{row["seed"]}.mp4'


def landing(rows: list[dict]) -> str:
    counts = Counter(row['event'] for row in rows)
    links = '\n'.join(
        f'<li><a href="failure_audit_model{i:02d}.html">模型 {i:02d}</a></li>'
        for i in range(1, 14)
    )
    return f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ContactWorld 离线案例对比</title>
<style>body{{font:17px/1.6 system-ui;max-width:850px;margin:32px auto;padding:0 18px;
background:#142032;color:#eaf2fc}}a{{color:#9ad5ff}}li{{margin:5px 0}}</style>
<h1>ContactWorld：40 个初态 × 13 个模型</h1>
<p>此包包含 520 条测试轨迹、双视图视频、双指 Fx/Fy/Fz 视频及结果标记。
所有链接使用相对路径；解压完整文件夹后，直接打开本页即可离线浏览。</p>
<p><a href="failure_audit_by_seed.html"><strong>按测试初态对比 13 个模型</strong></a></p>
<p><a href="failure_audit.html">查看全部 520 条结果总表</a></p>
<h2>按模型查看失败案例</h2><ul>{links}</ul>
<p>记录数：最终成功 {counts['final_success']}；曾成功但末态失败 {counts['transient_success']}；
从未成功 {counts['contact_no_success'] + counts['no_detected_contact']}。
每个模型页面按这三类结果分组；从未成功还区分是否检测到 USB 与插座接触。</p>
<p><a href="case_metadata.json">下载机器可读的结果元数据</a></p>
</html>'''


def package(report: Path, output: Path) -> None:
    source = json.loads((report / 'failure_audit.json').read_text(encoding='utf-8'))
    rows = source['rows']
    if len(rows) != 520 or len({r['seed'] for r in rows}) != 40:
        raise ValueError('Expected 520 evaluations covering 40 seeds')
    if len({(r['model'], r['seed']) for r in rows}) != 520:
        raise ValueError('Duplicate model/seed pair')
    if any(not (report / page).is_file() for page in PAGES):
        raise FileNotFoundError('One or more report pages are missing')

    media: dict[str, Path] = {}
    replacements: dict[str, str] = {}
    for row in rows:
        rgb = Path(row['video'])
        if not rgb.is_file():
            raise FileNotFoundError(rgb)
        name = rgb_name(row)
        media[name] = rgb
        replacements[os.path.relpath(rgb, report)] = name
        for mode in ('base', 'local'):
            tactile = f'failure_tactile/{mode}/{row["model"]}/seed_{row["seed"]}.mp4'
            path = report / tactile
            if not path.is_file():
                raise FileNotFoundError(path)
            media[tactile] = path
    if len(media) != 1560:
        raise ValueError(f'Expected 1560 distinct videos, got {len(media)}')

    pages: dict[str, str] = {'index.html': landing(rows)}
    old_path_pattern = re.compile(r'\.\./contactworld_pi0_v3_13models_40trials_20261003/[^"\s<>]+\.mp4')
    for page in PAGES:
        body = (report / page).read_text(encoding='utf-8')
        for old, new in replacements.items():
            body = body.replace(old, new)
        if old_path_pattern.search(body) or '/home/sai/' in body:
            raise ValueError(f'Unportable path remains in {page}')
        pages[page] = body

    portable = dict(source)
    portable['rows'] = [dict(row, video=rgb_name(row)) for row in rows]
    metadata = json.dumps(portable, ensure_ascii=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, 'w', allowZip64=True) as archive:
            for name, body in sorted(pages.items()):
                archive.writestr(f'{FOLDER}/{name}', body.encode('utf-8'),
                                 compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            archive.writestr(f'{FOLDER}/case_metadata.json', metadata.encode('utf-8'),
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            for index, (name, path) in enumerate(sorted(media.items()), 1):
                archive.write(path, f'{FOLDER}/{name}', compress_type=zipfile.ZIP_STORED)
                if index % 200 == 0 or index == len(media):
                    print(f'MEDIA {index}/{len(media)}', flush=True)

        with zipfile.ZipFile(temporary) as archive:
            names = set(archive.namelist())
            expected = {f'{FOLDER}/{name}' for name in (*pages.keys(), 'case_metadata.json', *media.keys())}
            if names != expected:
                raise ValueError(f'Archive manifest mismatch: {len(names)} vs {len(expected)}')
            for name, body in pages.items():
                if archive.read(f'{FOLDER}/{name}').decode('utf-8') != body:
                    raise ValueError(f'Page mismatch: {name}')
            broken = archive.testzip()
            if broken:
                raise ValueError(f'CRC check failed: {broken}')
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    checksum = hashlib.sha256()
    with output.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    checksum_file = output.with_suffix(output.suffix + '.sha256')
    checksum_file.write_text(f'{digest}  {output.name}\n', encoding='utf-8')
    print(f'ARCHIVE {output} ({output.stat().st_size / 1024**2:.1f} MiB)', flush=True)
    print(f'SHA256 {digest}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    package(args.report, args.output)


if __name__ == '__main__':
    main()

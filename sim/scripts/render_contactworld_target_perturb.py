#!/usr/bin/env python3
"""Render paired target-shift rollouts for scientific review."""
import argparse
import html
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root;report=json.loads((root/'analysis/analysis.json').read_text());plan=json.loads((root/'plan.json').read_text())
    out=root/'analysis';rows=report['trials']
    def rel(path):return html.escape(os.path.relpath(path,out),quote=True)
    def fmt(v):return '—' if v is None else f'{v:.2f}'
    sections=[]
    for m in plan['configs']:
      for s in plan['seeds']:
        sub=[next(r for r in rows if r['model']==m and r['seed']==s and r['condition']==c) for c in plan['conditions']]
        tr=[];options=[]
        for r in sub:
            status='最终成功' if r['final'] else '曾成功' if r['ever'] else '失败'
            name=r['condition'];vid=rel(root/m/f'seed_{s}'/name/'front_wrist.mp4')
            options.append(f'<option value="{vid}">{name} · {status}</option>')
            tr.append(f'''<tr><td>{name}</td><td>{r['dx_mm']:+.0f}, {r['dy_mm']:+.0f}</td><td>{status}</td><td>{fmt(r['first_contact_step'])}</td><td>{fmt(r['first4_translation_command_rms_diff_mm'])}</td><td>{fmt(r['first20_translation_command_rms_diff_mm'])}</td><td>{fmt(r['first_replan_with_0p5mm_command_rms_difference'])}</td><td>{fmt(r['min_keypoint_error_mm'])}</td><td>{fmt(r['final_plug_socket_xy_mm'])}</td><td>{fmt(r['final_plug_socket_z_mm'])}</td></tr>''')
        sections.append(f'''<section id="{m}-{s}"><h2>{m} · seed {s}</h2><p>选择条件查看视频（左前视、右腕视）；所有试验的策略噪声在同一步完全相同。</p>
        <select onchange="this.nextElementSibling.src=this.value">{''.join(options)}</select>
        <video controls preload="none" src="{rel(root/m/f'seed_{s}'/'zero'/'front_wrist.mp4')}"></video>
        <table><thead><tr><th>条件</th><th>目标XY偏移 mm</th><th>结果</th><th>首次接触步</th><th>前4步动作差 mm</th><th>前20步动作差 mm</th><th>差异≥0.5 mm的首个规划步</th><th>最小关键点误差 mm</th><th>末帧USB横向差 mm</th><th>末帧USB Z差 mm</th></tr></thead><tbody>{''.join(tr)}</tbody></table></section>''')
    plots=''.join(f'<h3>{m}</h3><img src="{m}.png">' for m in plan['configs'])
    flips=sum(r['ever']!=r['baseline_ever'] for r in rows if r['condition']!='zero')
    final_flips=sum(r['final']!=r['baseline_final'] for r in rows if r['condition']!='zero')
    rgb=[r for r in rows if r['model']==plan['configs'][0] and r['condition']!='zero']
    multi=[r for r in rows if r['model']==plan['configs'][1] and r['condition']!='zero']
    rgb_first4=sorted(r['first4_translation_command_rms_diff_mm'] for r in rgb)
    multi_first4=sorted(r['first4_translation_command_rms_diff_mm'] for r in multi)
    rgb_median=(rgb_first4[3]+rgb_first4[4])/2
    multi_median=(multi_first4[3]+multi_first4[4])/2
    cases=len(rows)-len(plan['configs'])*len(plan['seeds'])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>固定噪声的USB目标扰动试验</title><style>body{{font:16px/1.65 system-ui;max-width:1500px;margin:30px auto;padding:0 28px;color:#25384a}}table{{border-collapse:collapse;display:block;overflow:auto;font-size:14px}}td,th{{text-align:left;white-space:nowrap;padding:7px;border-bottom:1px solid #dce4eb}}th{{background:#eff4f8}}section{{margin:30px 0;padding:18px;border:1px solid #cbd6e1;border-radius:7px}}img{{width:100%;max-width:1150px}}video{{display:block;width:750px;max-width:100%}}select{{padding:8px;margin:8px 0}}a{{color:#1765a5}}</style>
    <h1>毫米级目标扰动：配对因果试验</h1>
    <p>两个初态、两个模型；每组把插座目标沿世界X/Y移动±3 mm，并保持机器人、USB、相机、模型与策略采样噪声相同。基线和扰动都在reset后额外执行同一次物理刷新。保存了全部视频、逐步轨迹、动作块及多模态输入。共{len(rows)}条轨迹，16个非零偏移中有{flips}个相对基线改变了“曾成功”结果，{final_flips}个改变了“最终成功”结果。</p>
    <p>按初态看：10005 两个模型各0/5曾成功；10017 RGB模型0/5，13号多模态模型5/5曾成功。13号模型10017的Y正负偏移都曾成功，但到第180步未保持通过，主要出现插入后的姿态漂移。±3 mm扰动不足以让10005成功，也未让10017的13号模型失去曾成功。</p>
    <p>模型确实能对小偏移作出动作响应：16条非零偏移轨迹中，RGB模型前4步平移指令相对基线的每分量RMS差异中位数{rgb_median:.2f} mm，13号模型{multi_median:.2f} mm。RGB初始前视图整图平均像素差不超过1.1/255；左右初始触觉数值完全相同。第一规划步就超过0.5 mm动作差的情况及其后续时间见下表。</p>
    <p>初始图像对照：<a href="initial_rgb_seed_10005.jpg">10005</a>、<a href="initial_rgb_seed_10017.jpg">10017</a>。每行依次显示前视、腕视及相对零偏移的8倍差异图；差异集中在目标边缘。</p>
    <p>实验检查：初始未扰动状态逐元素相同，目标实际偏移与请求误差小于0.001 mm；所有条件在每个规划步使用相同随机噪声。表中动作差以执行控制量的平移分量计算；0.5 mm阈值用于定位第一个可见响应，并非统计显著性阈值。相同目标扰动也改变了场景图像与点云，因此对多模态模型不能分离视觉和点云各自的因果效应。这里只测试两个初态和一组固定噪声，不能推断总体成功率。</p>
    {plots}{''.join(sections)}<p><a href="paired_trials.csv">20条配对指标CSV</a> · <a href="analysis.json">完整数值与一致性检查</a> · <a href="../../contactworld_pi0_v3_13models_40trials_20261003/training_neighbors/phase_audit/index.html">历史520条接触阶段分析</a></p></html>'''
    (out/'index.html').write_text(page)
    print(out/'index.html')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Render the 520-trajectory contact-phase action audit."""
import csv
import json
from pathlib import Path

ROOT=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003')
OUT=ROOT/'training_neighbors'/'phase_audit'

def main():
    s=json.loads((OUT/'summary.json').read_text());rows=list(csv.DictReader((OUT/'all_520_contact_phase.csv').open()))
    def num(k):return f'{s["action_similarity"][k]["mean"]:.3f}'
    cases=[]
    for seed in [10005,10017,10034]:
        for model in [1,13]:
            x=next(r for r in rows if int(r['seed'])==seed and int(r['model'])==model)
            cases.append(f'<tr><td>{seed}</td><td>{model:02d}</td><td>{"成功" if int(x["ever"]) else "失败"}</td><td>{float(x["pre_action_cos3d"]):.3f}</td><td>{float(x["post_action_cos3d"]):.3f}</td><td>{float(x["pre8_action_cos3d"]):.3f}</td><td>{float(x["post8_action_cos3d"]):.3f}</td><td>{float(x["post_test_plug_delta_z_mm"]):.1f}</td></tr>' if x['post_action_cos3d'] else f'<tr><td>{seed}</td><td>{model:02d}</td><td>无接触，无法对齐</td></tr>')
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>接触前后动作相似性审核</title>
    <style>body{{font:16px/1.65 system-ui;max-width:1200px;margin:35px auto;padding:0 25px;color:#25394b}}a{{color:#2167a4}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #d7e0e9}}th{{background:#edf3f7}}img{{width:100%;max-width:1100px}}.callout{{background:#edf4fa;padding:15px;border-left:4px solid #2f77aa}}</style>
    <h1>接触前后动作是否从训练示范分开？</h1>
    <div class="callout">接近阶段与训练示范较相似，方向差异在接触前数步开始扩大，并在接触后继续扩大。这是普遍现象，并非失败独有。40个测试初态、13个模型共520条轨迹；513条可同时对齐接触前后，7条未检测到插件–插座接触。</div>
    <p>为每个测试初态固定此前按初始几何选出的最近训练示范。首次插件–插座接触定义为接触力模长首次超过0.1。接触前动作按各自接触时刻线性重采样为24点，接触后比较前20步；并额外比较接触前最后8步和接触后最初8步，检查示范结束造成的偏差。表中均为40个初态上的13模型平均，置信区间按初态重采样。</p>
    <table><thead><tr><th>动作方向余弦，1=同方向</th><th>接触前</th><th>接触后</th><th>变化</th></tr></thead><tbody>
    <tr><td>平移3D，整个阶段</td><td>{num('pre_cos3d')}</td><td>{num('post_cos3d')}</td><td>{num('post_minus_pre_cos3d')}</td></tr>
    <tr><td>平移XY，整个阶段</td><td>{num('pre_cos_xy')}</td><td>{num('post_cos_xy')}</td><td>{num('post_minus_pre_cos_xy')}</td></tr>
    <tr><td>平移3D，接触附近8步</td><td>{num('pre8_action_cos3d')}</td><td>{num('post8_action_cos3d')}</td><td>−0.175；按初态bootstrap 95%区间 [−0.255, −0.098]</td></tr>
    <tr><td>平移XY，接触附近8步</td><td>{num('pre8_action_cos_xy')}</td><td>{num('post8_action_cos_xy')}</td><td>−0.244</td></tr></tbody></table>
    <p>最近几何示范在95条训练示范里的动作相似度排名：接触前中位第7名，63.4%的轨迹进入前10；接触后中位第39名，10.5%进入前10。可见前段方向具有一定示范特异性，后段则明显减弱。但“方向相似”不等于空间轨迹重合：对齐接近阶段并各自减去起点后，末端路径形状误差中位数仍约24.5 mm（详见<a href="../behavior_comparison_520.csv">路径指标</a>）。</p>
    <img src="phase_comparison.png">
    <h2>差异从哪一步开始？</h2>
    <img src="aligned_step_similarity.png">
    <p>按各自首次接触对齐：接触前第8步3D方向余弦约0.81，前第4步约0.70，首次接触约0.66，接触后第3–4步约0.50。因此“接触后才开始不同”过于严格；策略在临近孔口时已经出现分歧。曲线只统计双方动作幅值足够大的步，阴影是以初态为单位的bootstrap区间；各时刻有效样本数见<a href="aligned_step_similarity.json">时间曲线JSON</a>。</p>
    <h2>动作分化与任务失败的关系</h2>
    <p>曾成功与失败轨迹的动作相似度都在接触后下降：失败轨迹接触前/后3D余弦中位数0.850/0.589，曾成功轨迹0.852/0.582。仅凭“接触后动作不同于最近示范”不能解释谁会失败。更能区分结果的是物体运动：接触后20步USB下降的中位数，失败轨迹约0.3 mm，曾成功轨迹约4.7 mm，最近训练示范约11 mm。失败时常见夹爪继续运动而USB不随动。</p>
    <p>动作方向余弦不评价动作幅值；接触后训练示范可能接近任务结束，时间长度也不同，因此这里的分化不等于“模型在接触瞬间识别错误”。后续毫米级扰动试验用于检验原因。</p>
    <h2>三个例子</h2><table><thead><tr><th>seed</th><th>模型</th><th>结果</th><th>接触前3D</th><th>接触后3D</th><th>接触前8步</th><th>接触后8步</th><th>接触后USB Z变化 mm</th></tr></thead><tbody>{''.join(cases)}</tbody></table>
    <p><a href="all_520_contact_phase.csv">520条轨迹指标</a> · <a href="per_seed_contact_phase.csv">40个初态聚合</a> · <a href="summary.json">汇总JSON</a> · <a href="../alias_followup/index.html">三例视觉与轨迹复核</a></p></html>'''
    (OUT/'index.html').write_text(page)
    print(OUT/'index.html')

if __name__=='__main__':main()

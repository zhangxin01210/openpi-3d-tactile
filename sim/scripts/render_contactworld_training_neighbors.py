#!/usr/bin/env python3
"""Render train-only nearest-neighbor findings, trajectories and local video links."""
import argparse
import csv
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_contactworld_training_neighbors import geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--dataset',type=Path,default=Path('data/contactworld_usb_v3_bilateral'))
    a=p.parse_args();a.dataset=a.dataset.resolve()
    out=a.root/'training_neighbors'
    report=json.loads((out/'analysis.json').read_text())
    manifest=json.loads((a.dataset/'spatial/manifest.json').read_text())
    entries={e['source_episode']:e for e in manifest['episodes'] if e['split']=='train'}
    models=sorted(a.root.glob('pi0_cw3_usb_*/summary.json'))
    summaries=[json.loads(x.read_text()) for x in models]
    behaviors=list(csv.DictReader((out/'behavior_comparison_520.csv').open()))
    neighbors=report['neighbors'];top5=list(csv.DictReader((out/'top5_training_neighbors.csv').open()))
    def link(path):return html.escape(os.path.relpath(path,out),quote=True)
    def fmt(x):return '—' if x is None or x=='' else f'{float(x):.2f}'
    def table(headers,rows):
        return '<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,['models_ever_success','models_final_success'],['Ever success','Final success']):
        x=[n['distance'] for n in neighbors];y=[n[key]/13 for n in neighbors]
        ax.scatter(x,y,color='#286bb0',alpha=.8)
        for n in neighbors:
            if n['seed'] in [10005,10008,10017,10034]:ax.annotate(str(n['seed']),(n['distance'],n[key]/13),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Nearest training initial-state distance D',ylabel='Successful models / 13',title=title,ylim=(-.04,1.07));ax.grid(alpha=.2)
    fig.savefig(out/'distance_success.png',dpi=160);plt.close(fig)
    sections=[];configs=[]
    for si,n in enumerate(neighbors):
        seed=n['seed'];source=n['nearest_source_episode'];entry=entries[source]
        folder=Path(manifest['capture_root'])/f'insertion_usb_episode_{source:03d}'
        with np.load(folder/'v2_capture.npz') as z:
            demo={k:z[k] for k in ['plug_pose','socket_pose','ee_pose']}
        dg=geometry(demo['plug_pose'],demo['socket_pose'],demo['ee_pose'])
        replay=json.loads((folder/'replay.json').read_text())
        trainforce=np.linalg.norm([r['plug_socket_force_post'] for r in replay['rows']],axis=1)
        contacts=np.flatnonzero(trainforce>.1);dc=int(contacts[0])+1 if len(contacts) else 0
        init=summaries[0]['results'][si]['initial_physical_state']
        tg=geometry(np.r_[init['plug_pos'],init['plug_quat']],np.r_[init['socket_pos_gt'],init['socket_quat']],np.r_[init['ee_pos'],init['ee_quat']])
        config={'seed':seed,'nearest_source_episode':source,'test_initial_physical_state':init,
                'train_initial_poses':{k:v[0].tolist() for k,v in demo.items()},
                'test_geometry':{k:v[0].tolist() for k,v in tg.items()},'train_geometry':{k:v[0].tolist() for k,v in dg.items()}}
        configs.append(config)
        fig,axs=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
        axs[0].plot(dg['ee_relative'][:,0]*1000,dg['ee_relative'][:,2]*1000,'k-',lw=2,label=f'Train {source}')
        axs[1].plot(np.arange(len(dg['relative']))-dc,dg['relative'][:,2]*1000,'k-',lw=2)
        axis=Rotation.from_quat(dg['rotation']).apply(np.tile([0,0,1],(len(dg['rotation']),1)))
        axs[2].plot(np.arange(len(axis))-dc,np.degrees(np.arccos(np.clip(axis[:,2],-1,1))),'k-',lw=2)
        for mi,color in [(0,'#e07b19'),(3,'#267bbc'),(12,'#319b61')]:
            tr=json.loads((models[mi].parent/f'seed_{seed}/trajectory.json').read_text())
            states=[tr['result']['initial_physical_state']]+tr['rows']
            g=geometry(np.array([r['plug_pos']+r['plug_quat'] for r in states]),np.array([r['socket_pos_gt']+r['socket_quat'] for r in states]),np.array([r['ee_pos']+r['ee_quat'] for r in states]))
            cf=np.linalg.norm([r['plug_socket_contact_force_xyz'] for r in tr['rows']],axis=1);ix=np.flatnonzero(cf>.1)
            tc=int(ix[0])+1 if len(ix) else 0
            label=f'M{mi+1:02d} '+('final' if tr['result']['source_final'] else 'ever only' if tr['result']['source_ever'] else 'fail')+(' (no contact)' if not len(ix) else '')
            axs[0].plot(g['ee_relative'][:,0]*1000,g['ee_relative'][:,2]*1000,color=color,label=label,alpha=.85)
            axs[1].plot(np.arange(len(states))-tc,g['relative'][:,2]*1000,color=color,alpha=.85)
            axis=Rotation.from_quat(g['rotation']).apply(np.tile([0,0,1],(len(states),1)))
            axs[2].plot(np.arange(len(states))-tc,np.degrees(np.arccos(np.clip(axis[:,2],-1,1))),color=color,alpha=.85)
        axs[0].set(xlabel='EE X relative to socket (mm)',ylabel='EE Z relative to socket (mm)',title=f'Seed {seed}: approach and insertion');axs[0].legend(fontsize=7)
        axs[1].set(xlabel='Step relative to first contact',ylabel='Plug root Z relative to socket (mm)',title='Contact-aligned insertion depth')
        axs[2].set(xlabel='Step relative to first contact',ylabel='Plug/socket axis angle (deg)',title='Contact-aligned tilt')
        for ax in axs:ax.grid(alpha=.2)
        fig.savefig(out/f'seed_{seed}.png',dpi=120);plt.close(fig)
        options=[];rows=[]
        for mi,model in enumerate(models):
            r=summaries[mi]['results'][si];b=next(x for x in behaviors if int(x['seed'])==seed and int(x['model'])==mi+1)
            status='最终成功' if r['source_final'] else '曾成功，最终未通过' if r['source_ever'] else '未成功'
            options.append(f'<option value="{link(model.parent/f"seed_{seed}/front_wrist.mp4")}">{mi+1:02d} {model.parent.name.removeprefix("pi0_cw3_usb_")} · {status}</option>')
            rows.append([f'{mi+1:02d}',status,fmt(b['test_contact_step']),fmt(b['approach_target_relative_ee_path_rmse_mm']),fmt(b['approach_displacement_shape_rmse_mm']),fmt(b['first20_translation_command_rms_mm']),fmt(b['first20_rotation_command_rms_deg'])])
        ep=n['nearest_episode_index']
        trainvids=''.join(f'<div>{view}<video controls preload="none" src="{link(a.dataset/f"videos/chunk-000/observation.images.{view}/episode_{ep:06d}.mp4")}"></video></div>' for view in ['front','wrist'])
        candidates=[r for r in top5 if int(r['seed'])==seed]
        ct=table(['排名','训练 source ID','LeRobot episode','D','目标差 mm','相对 USB 差 mm','姿态差 °','夹持差 mm','夹持角差 °'],[[r['rank'],r['source_episode'],r['episode_index']]+[fmt(r[k]) for k in ['distance','target_mm','relative_plug_mm','rotation_deg','grasp_mm','grasp_rotation_deg']] for r in candidates])
        sections.append(f'''<details id="seed{seed}"><summary>seed {seed} → 训练 source {source} · D={n['distance']:.2f} · 曾成功 {n['models_ever_success']}/13，最终 {n['models_final_success']}/13</summary>
        <p>最近邻按初态选定，未使用测试结局。所有训练观测帧中的最近邻：source {n['seen_frame_source_episode']}，frame {n['seen_frame']}，D={n['seen_frame_distance']:.2f}。</p>{ct}
        <h3>实际训练视频：source {source} / LeRobot episode {ep}</h3><div class="train">{trainvids}</div>
        <h3>测试视频（左前视、右腕视）</h3><select onchange="this.nextElementSibling.src=this.value">{''.join(options)}</select><video class="test" controls preload="none" src="{link(models[0].parent/f'seed_{seed}/front_wrist.mp4')}"></video>
        <p>视频保留各自原始时间，不自动同步；下方曲线按首次接触对齐。无接触轨迹以起始步为零。训练与测试均以首次出现超过0.1的插件–插座接触力的动作后时刻对齐；该阈值仅用于分段。</p><img loading="lazy" src="seed_{seed}.png">
        {table(['模型','结果','首次接触步','接近路径误差 mm','接近位移形状误差 mm','前20步平移指令差 mm','前20步旋转指令差 °'],rows)}</details>''')
    (out/'configurations.json').write_text(json.dumps(configs,indent=2)+'\n')
    names={'initial_full':'初态：完整几何','initial_task_only':'初态：仅目标/USB','initial_full_angle2.5':'初态：提高角度权重','initial_full_angle10':'初态：降低角度权重','initial_full_top5_mean':'初态：前5邻居平均','seen_frames_full':'所有有效观测帧：完整几何','seen_frames_task_only':'所有有效观测帧：仅目标/USB'}
    stats=table(['距离口径','曾成功 ρ','双侧置换 p','ρ 的95%区间','较近20例成功率','较远20例成功率','最终成功 ρ'],[[names[k],fmt(v['ever']['spearman_rho']),f"{v['ever']['permutation_p_two_sided']:.3f}",str([round(x,3) for x in v['ever']['seed_bootstrap95']]),f"{v['ever']['near20_mean_success']:.1%}",f"{v['ever']['far20_mean_success']:.1%}",fmt(v['final']['spearman_rho'])] for k,v in report['statistics'].items()])
    per_model=table(['模型','距离–曾成功 ρ','距离–最终成功 ρ','较近20例曾成功','较远20例曾成功'],[[f"{r['model']:02d}",fmt(r['rho_ever']),fmt(r['rho_final']),f"{r['near20_ever']}/20",f"{r['far20_ever']}/20"] for r in report['per_model']])
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>训练覆盖与测试行为匹配</title><style>body{{font:16px/1.65 system-ui;max-width:1300px;margin:35px auto;padding:0 24px;color:#243348}}h1,h2{{color:#17385d}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:7px;border-bottom:1px solid #dce4ed;text-align:left}}th{{background:#edf3f9}}details{{border:1px solid #ccd9e6;border-radius:8px;margin:18px 0;padding:16px;overflow:auto}}summary{{cursor:pointer;font-weight:650}}img{{max-width:100%}}.train{{display:flex;gap:16px}}.train video{{display:block;width:320px;max-width:100%}}.test{{display:block;width:720px;max-width:100%}}select{{max-width:100%;margin-bottom:10px;padding:8px}}.notice{{background:#eff6fc;padding:18px;border-left:4px solid #3276bb}}a{{color:#1c65a3}}</style>
    <h1>成功是否来自训练集中的相近例子？</h1>
    <p><a href="alias_followup/index.html">新增：三条示范的视觉相似、动作偏差与10034严格续跑核查</a></p>
    <p><a href="phase_audit/index.html">新增：520条轨迹的接触前后动作相似性统计</a></p>
    <p><a href="ensemble_audit/index.html">新增：动态训练帧近邻、3/5/10条示范集成与±3 mm近邻切换</a></p>
    <p><a href="../../contactworld_target_perturb_20261003/analysis/index.html">新增：固定策略噪声的目标位置±3 mm配对试验</a></p>
    <div class="notice">现有40个测试初态没有找到相同的训练几何配置。距离与成功率呈弱负相关，但七种距离口径的95%区间均跨过0。当前证据不足以支持“成功主要因为训练中有近邻、失败主要因为没有近邻”的解释，也不能排除覆盖不足的影响。</div>
    <p>使用实际训练划分的95条示范、5064个有效观测起始帧；排除验证集与测试集。40个初态 × 13个模型共520次评测。同初态不同模型不是独立样本：统计置换与bootstrap均以40个初态为单位，先计算该初态13个模型的平均成功率。</p>
    <h2>匹配定义与边界</h2><p>比较目标绝对位置、USB相对目标的位置/姿态、USB相对夹爪的位置/姿态。D=√[(目标差/10mm)²+(相对位置差/10mm)²+(相对角差/5°)²+(夹持位置差/5mm)²+(夹持角差/5°)²]。四元数比较处理q与−q等价。D是诊断距离，不是物理成功阈值；“最近”不代表“几乎相同”。零重复指位置1微米、角度0.0001°容差下无几何匹配，并不代表测试来自分布外。</p>
    <p>所有帧检索仅包含16步动作窗口的有效起始帧。比较的是数据中可供训练的观测，未声称逐帧核实了训练采样日志。此匹配没有纳入RGB、触觉、速度、接触缓存等隐状态，不能证明模型记忆或泛化机制。</p>
    <img src="distance_success.png">{stats}
    <p>主口径最终成功率：较近20例31.2%，较远20例30.8%。这些百分比是20个初态的模型平均，不把260次运行视为独立统计样本。七个口径为敏感性检查，不从中挑最有利的p值。</p>
    <details><summary>逐模型的覆盖关系（探索性统计）</summary>{per_model}<p>部分模型方向相反；05的曾成功、10的最终成功与距离的负相关相对较大，但40例与多个比较不足以据此确认模态依赖差异。</p></details>
    <h2>优先查看的对照</h2><ul><li><a href="#seed10005">10005 → source188</a>：D=3.94，13个模型全失败；与<a href="#seed10008">10008 → source106</a>的D=3.96相近，后者12/13最终成功。</li><li><a href="#seed10017">10017 → source14</a>：D=5.86，仍有12/13曾成功、8/13最终成功。</li><li><a href="#seed10034">10034 → source97</a>：D=7.22，全失败，是覆盖不足假说值得进一步追查的一例。</li><li><a href="#seed10014">10014 → source44</a>：04与13的接近路径误差约11.7/11.0mm，但04失败、13最终成功；初始邻近度相同，结果分化需要继续检查接触阶段。</li></ul>
    <h2>行为比较怎么读</h2><p>每个测试例的13个模型共享同一条最近初态示范。路径在插座坐标系中比较，初始到首次接触分别重采样为30点；路径误差保留起点差异，位移形状误差去掉各自起点。动作差比较执行裁剪后的前20步，乘控制缩放后报告每分量RMS，属于指令差，不是实际运动误差。速度不同会影响动作差，阶段重采样也会抹去时序差，因此两者联合查看。接近相似不等于动作复制，更不能凭这些观测认定“一点初态变化导致失败”。</p>
    <p>下一步宜围绕10005/10008、10034、10014做接触阶段覆盖分析；如需要因果验证，再从已知训练初态出发，只改变目标位置或夹持姿态，固定策略噪声与其它条件做小范围配对测试。当前结果无需重新训练即可得到。</p>
    <p>下载：<a href="nearest_training_per_seed.csv">每例最近邻</a> · <a href="top5_training_neighbors.csv">每例前5邻居</a> · <a href="behavior_comparison_520.csv">520次行为指标</a> · <a href="configurations.json">完整初始配置</a> · <a href="analysis.json">统计JSON</a></p>
    <h2>全部40个测试例</h2>{''.join(sections)}<script>function reveal(){{let e=document.querySelector(location.hash||'#none');if(e&&e.tagName==='DETAILS')e.open=true;}}window.addEventListener('hashchange',reveal);reveal();</script></html>'''
    (out/'index.html').write_text(text)
    print(out/'index.html')

if __name__=='__main__':main()

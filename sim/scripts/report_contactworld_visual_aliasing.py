#!/usr/bin/env python3
"""Produce a focused review of three visual-aliasing and horizon hypotheses."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_contactworld_training_neighbors import geometry

ROOT=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003')
TRAIN=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_bilateral_full_20261002/replay')
OUT=ROOT/'training_neighbors'/'alias_followup'
EXT=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_10034_horizon_probe_20261003/pi0_cw3_usb_13_fused_pc_local_map_3tok/seed_10034')
PREF=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_10034_prefix_continuation_20261003')
CASES={10005:(188,[1,13]),10017:(14,[1,13]),10034:(97,[1,13])}

def trial(seed,model):
    p=next(ROOT.glob(f'pi0_cw3_usb_{model:02d}_*/seed_{seed}/trajectory.json'))
    return json.loads(p.read_text()),p.parent

def init_geom(info):
    return geometry(np.r_[info['plug_pos'],info['plug_quat']],np.r_[info['socket_pos_gt'],info['socket_quat']],np.r_[info['ee_pos'],info['ee_quat']])

def testgeom(t):
    states=[t['result']['initial_physical_state']]+t['rows']
    return geometry(np.array([r['plug_pos']+r['plug_quat'] for r in states]),np.array([r['socket_pos_gt']+r['socket_quat'] for r in states]),np.array([r['ee_pos']+r['ee_quat'] for r in states]))

def mm(v):return f'{v:.1f}'

def relative(p):return os.path.relpath(p,OUT)

def main():
    visual={x['seed']:x for x in json.loads((OUT/'visual_distance.json').read_text())}
    neighbor={x['seed']:x for x in json.loads((ROOT/'training_neighbors'/'analysis.json').read_text())['neighbors']}
    visual_rho=float(spearmanr(
        [visual[s]['closest_visual_error'] for s in sorted(visual)],
        [neighbor[s]['models_ever_success']/13 for s in sorted(visual)]).statistic)
    records=[];sections=[]
    for seed,(src,models) in CASES.items():
        folder=TRAIN/f'insertion_usb_episode_{src:03d}'
        with np.load(folder/'v2_capture.npz') as z:
            pose={k:z[k] for k in ['plug_pose','socket_pose','ee_pose']};actions=z['action_executed']
        gtrain=geometry(pose['plug_pose'],pose['socket_pose'],pose['ee_pose'])
        rtrain=json.loads((folder/'replay.json').read_text())['rows']
        fc=np.flatnonzero(np.linalg.norm([r['plug_socket_force_post'] for r in rtrain],axis=1)>.1);ct=int(fc[0])+1 if len(fc) else None
        t0,_=trial(seed,models[0]);q=init_geom(t0['result']['initial_physical_state'])
        targetdelta=(q['target'][0]-gtrain['target'][0])*1000
        reldelta=(q['relative'][0]-gtrain['relative'][0])*1000
        gripperdelta=(q['ee_relative'][0]-gtrain['ee_relative'][0])*1000
        fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        axs[0].plot(gtrain['relative'][:,0]*1000,gtrain['relative'][:,1]*1000,'k-',lw=2,label=f'train source {src}')
        axs[1].plot(np.arange(len(actions)),gtrain['relative'][:,2]*1000,'k-',lw=2)
        axs[2].plot(np.arange(len(actions)),gtrain['ee_relative'][:,0]*1000,'k-',lw=2)
        facts=[]
        for mi,color in zip(models,['#de6f0b','#197caf']):
            t,path=trial(seed,mi);gg=testgeom(t);rr=t['rows'];forces=np.linalg.norm([r['plug_socket_contact_force_xyz'] for r in rr],axis=1)
            ix=np.flatnonzero(forces>.1);contact=int(ix[0])+1 if len(ix) else None
            u=np.array([r['action_sim_clipped'][:3] for r in rr[:20]]);v=actions[:20,:3]
            active=(np.linalg.norm(u,axis=1)>.1)&(np.linalg.norm(v,axis=1)>.1)
            cos=np.sum(u[active]*v[active],axis=1)/(np.linalg.norm(u[active],axis=1)*np.linalg.norm(v[active],axis=1))
            f={'seed':seed,'source':src,'model':mi,'target_delta_xyz_mm':targetdelta.round(2).tolist(),'relative_plug_delta_xyz_mm':reldelta.round(2).tolist(),'initial_ee_relative_delta_xyz_mm':gripperdelta.round(2).tolist(), 'train_contact_step':ct,'test_contact_step':contact,'first20_active_command_cosine_mean':float(cos.mean()),'train_contact_plug_xyz_mm':(gtrain['relative'][ct]*1000).round(2).tolist() if ct is not None else None,'test_contact_plug_xyz_mm':(gg['relative'][contact]*1000).round(2).tolist() if contact is not None else None,'test_end_plug_xyz_mm':(gg['relative'][-1]*1000).round(2).tolist(),'test_end_keypoint_error_mm':float(rr[-1]['keypoint_error_mm']),'test_min_keypoint_error_mm':float(t['result']['min_keypoint_error_mm']),'source_ever':bool(t['result']['source_ever']),'source_final':bool(t['result']['source_final'])}
            facts.append(f);records.append(f)
            axs[0].plot(gg['relative'][:,0]*1000,gg['relative'][:,1]*1000,color=color,label=f'M{mi:02d} {"success" if f["source_ever"] else "failure"}')
            axs[1].plot(np.arange(len(gg['relative'])),gg['relative'][:,2]*1000,color=color)
            axs[2].plot(np.arange(len(gg['ee_relative'])),gg['ee_relative'][:,0]*1000,color=color)
            if contact is not None:axs[0].scatter(* (gg['relative'][contact,:2]*1000),color=color,marker='x',s=60)
        axs[0].scatter(*(gtrain['relative'][ct,:2]*1000),color='k',marker='x',s=60)
        axs[0].set(xlabel='Plug relative X (mm)',ylabel='Plug relative Y (mm)',title='USB path in socket frame');axs[0].legend(fontsize=8)
        axs[1].axhline(0,color='gray',lw=.7);axs[1].set(xlabel='Action step',ylabel='Plug relative Z (mm)',title='Insertion height')
        axs[2].set(xlabel='Action step',ylabel='EE relative X (mm)',title='End-effector approach')
        for ax in axs:ax.grid(alpha=.2)
        fig.savefig(OUT/f'seed_{seed}_trajectories.png',dpi=150);plt.close(fig)
        rows=''.join(f'<tr><td>{x["model"]:02d}</td><td>{"是" if x["source_ever"] else "否"}</td><td>{x["first20_active_command_cosine_mean"]:.3f}</td><td>{x["test_contact_step"]}</td><td>{", ".join(map(mm,x["test_contact_plug_xyz_mm"])) if x["test_contact_plug_xyz_mm"] else "无"}</td><td>{", ".join(map(mm,x["test_end_plug_xyz_mm"]))}</td><td>{x["test_end_keypoint_error_mm"]:.1f}</td></tr>' for x in facts)
        videos=''.join(f'<p>M{mi:02d}: <a href="{relative(path/"front_wrist.mp4")}">测试视频</a></p>' for mi in models for _,path in [trial(seed,mi)])
        v=visual[seed]
        sections.append(f'''<section id="seed{seed}"><h2>seed {seed} 对照 source {src}</h2>
        <p>目标绝对位置差 ({', '.join(map(mm,targetdelta))}) mm；USB相对目标初始差 ({', '.join(map(mm,reldelta))}) mm。初始RGB在95条训练示范中的像素近邻：前视第{v['visual_front_rank']}、腕视第{v['visual_wrist_rank']}、两路合并第{v['visual_both_rank']}。</p>
        <img loading="lazy" src="seed_{seed}_initial_rgb.jpg" style="width:650px;max-width:100%"><img loading="lazy" src="seed_{seed}_trajectories.png">
        <p>训练 source {src} 首次插件–插座接触约第 {ct} 步，当时USB相对目标 (X,Y,Z)=({', '.join(map(mm,gtrain['relative'][ct]*1000))}) mm。下表的动作余弦只对前20步双方平移指令均有幅值的步求平均；1表示方向一致，不能证明模型记忆该轨迹。</p>
        <table><thead><tr><th>模型</th><th>曾成功</th><th>前20步动作方向余弦</th><th>首次接触步</th><th>接触时USB XYZ mm</th><th>末帧USB XYZ mm</th><th>末帧关键点误差 mm</th></tr></thead><tbody>{rows}</tbody></table>{videos}</section>''')
    extension=None
    if (EXT/'trajectory.json').is_file():
        t=json.loads((EXT/'trajectory.json').read_text());r=t['rows'];g=testgeom(t)
        original,_=trial(10034,13);ro=original['rows'];go=testgeom(original)
        diff=np.linalg.norm((g['relative'][:181]-go['relative'])*1000,axis=1)
        extension={'steps':len(r),'ever':t['result']['source_ever'],'final':t['result']['source_final'],'keypoint_180_mm':float(r[179]['keypoint_error_mm']),'keypoint_300_mm':float(r[-1]['keypoint_error_mm']),'plug_relative_180_xyz_mm':(g['relative'][180]*1000).round(2).tolist(),'plug_relative_300_xyz_mm':(g['relative'][-1]*1000).round(2).tolist(),'first180_plug_relative_mean_disagreement_mm':float(diff.mean()),'first180_plug_relative_max_disagreement_mm':float(diff.max())}
    continuation=json.loads((PREF/'summary.json').read_text()) if (PREF/'summary.json').is_file() else None
    (OUT/'findings.json').write_text(json.dumps({'cases':records,'extension':extension,'prefix_continuation':continuation,'visual_distance_success_spearman_40_seeds':visual_rho,'pixel_distance':'64x64 CIELAB RMS, only an initial-image similarity proxy'},indent=2)+'\n')
    exttext='延长评测仍在运行。'
    if extension:
        exttext=f'''M13 用相同 seed 从头运行到 {extension['steps']} 步，曾成功：{extension['ever']}；最终成功：{extension['final']}。第180步误差 {extension['keypoint_180_mm']:.1f} mm，末步 {extension['keypoint_300_mm']:.1f} mm；末步USB相对目标 ({', '.join(map(mm,extension['plug_relative_300_xyz_mm']))}) mm。
        新运行与原运行前180步的USB相对位姿平均位置差 {extension['first180_plug_relative_mean_disagreement_mm']:.1f} mm。这是同 seed、独立重新运行，不是从原第180步恢复；比较是否可信须看这个差异。<a href="{relative(EXT/'front_wrist.mp4')}">300步视频</a>'''
    if continuation:
        d=continuation['prefix_end_state_differences']
        continued=json.loads((PREF/'trajectory.json').read_text())['rows']
        fig,axs=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
        for ax,key,label in zip(axs,('keypoint_error_mm','plug_socket_xy_error_mm','plug_socket_z_mm'),
                                 ('Keypoint error (mm)','USB lateral offset (mm)','USB relative Z (mm)')):
            ax.plot([r['step']+1 for r in continued],[r[key] for r in continued],color='#2775af')
            ax.axvline(180,color='#c45c30',linestyle='--',label='Original episode ends')
            ax.set(xlabel='Step',ylabel=label);ax.grid(alpha=.25)
        axs[0].legend(fontsize=8)
        fig.savefig(OUT/'seed_10034_exact_continuation.png',dpi=150);plt.close(fig)
        exttext+=f'''<p>严格动作前缀检验：逐步重放原始180个动作后，插件位置与原运行相差 {d['plug_pos']*1000:.3f} mm，夹爪位置相差 {d['ee_pos']*1000:.3f} mm。之后由策略继续120步：曾成功={continuation['continuation_ever_success']}，末帧关键点误差 {continuation['continuation_end_keypoint_mm']:.1f} mm，末帧横向偏差 {continuation['continuation_end_plug_socket_xy_mm']:.1f} mm。续跑策略的随机采样重新开始，物理前缀已核对。<a href="{relative(PREF/'front_wrist.mp4')}">前缀重放+续跑视频</a></p>'''
        exttext+='<img src="seed_10034_exact_continuation.png" style="width:100%;max-width:1050px">'
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>视觉相似与插入失败复核</title><style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#273849}}section{{margin:35px 0;padding:22px;border:1px solid #cad5df;border-radius:9px}}img{{display:block;max-width:100%;margin:12px 0}}table{{border-collapse:collapse;display:block;overflow:auto}}td,th{{white-space:nowrap;padding:7px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#1b6aaf}}</style><h1>视觉相似、动作方向与失败位置</h1>
    <p>比较三条最近训练示范与测试轨迹。使用插座坐标系下的USB位置，避免画面角度掩盖实际偏差。图中叉号表示第一次插件–插座接触。初始图像的排名来自 64×64 CIELAB 像素误差；这是粗略视觉相似度，不能代表π0内部的视觉表征。</p>
    <p>10005与10017的失败主要出现在孔口接触后，USB根部停在约18 mm高度；10034更早偏离示范，模型在USB横向偏离孔位时已下降到孔口下方。看起来相似的图像可能掩盖厘米级空间差异；现有轨迹无法判断模型主观上“以为两例相同”。40个初态中，该粗略图像距离与13模型平均曾成功率的Spearman相关为 {visual_rho:.3f}，不能据此认为视觉相似性解释了大多数成功或失败。</p>
    <h2>10034延长到300步</h2><p>{exttext}</p>{''.join(sections)}
    <p>原始数值：<a href="findings.json">findings.json</a>、<a href="visual_distance.json">40例RGB近邻</a>。</p></html>'''
    (OUT/'index.html').write_text(html)
    print(OUT/'index.html')

if __name__=='__main__':main()

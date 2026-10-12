#!/usr/bin/env python3
"""Post-run interpretation for the reviewed 14-class tactile atlas."""
import sys, json, io, contextlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, str(Path.cwd()/'sim/scripts'))
from run_contactworld_tactile_cluster_atlas import load_episodes, normalized_force, concatenate, assignment
import argparse
parser=argparse.ArgumentParser(description='Audit K=14 tactile clusters with pre-success contact conditioning and action-history checks.')
parser.add_argument('--atlas',type=Path,default=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_clusters_k14_20261008'))
parser.add_argument('--dataset',type=Path,default=Path('data/contactworld_usb_v3_bilateral'))
args=parser.parse_args()
root=args.atlas
with contextlib.redirect_stdout(io.StringIO()):
    entries,episodes,parts=load_episodes(SimpleNamespace(dataset=args.dataset,smoke=False))
model=np.load(root/'cluster_model.npz');scale=model['scale'];centers=model['centers']
if len(centers)!=14:raise ValueError('This interpretation is specific to the reviewed K=14 model.')
D={}
for split in ('train','val','test'):
    rows=[]
    for e,d in parts[split]:
        n=len(d['force']);f=d['force'];t=np.arange(n)
        c,_=assignment(np.clip(f/scale[None,:,None,:],-3,3).reshape(n,840),centers)
        z=f[:,:,:,2];xy=f[:,:,:,:2]
        prev=np.vstack([np.full((1,3),np.nan),np.diff(d['relative'],axis=0)])
        item=dict(source=np.full(n,e['source_episode']),frame=t,cluster=c,contact=d['contact'],
                  before=~np.maximum.accumulate(d['success']),valid=(t>0)&(t<n-1),
                  fz_rms=np.sqrt(np.mean(z*z,axis=(1,2))),fz_signed=z.mean(axis=(1,2)),
                  fxy_rms=np.sqrt(np.mean(xy*xy,axis=(1,2,3))),
                  action=d['action'],motion=d['next_motion'],previous_motion=prev,
                  relative=d['relative'])
        rows.append(item)
    D[split]={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}

def auc(y,s):
    y=np.asarray(y,bool);s=np.asarray(s);n=y.sum();m=len(y)-n
    return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None

def scores(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool);tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fp),specificity=tn/max(1,tn+fp),balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)))
report={'classes':{},'signal_detection':{},'fz_quiet':{},'directions':{}}
for split,d in D.items():
    report['classes'][split]=[]
    for k in range(14):
        base=d['valid']&(d['cluster']==k)
        mask=base&d['before']&d['contact']&(d['action'][:,2]<-2)
        p=mask&(d['motion'][:,2]<-.4);b=mask&(d['motion'][:,2]>-.25)
        decisive=p|b
        demos=np.unique(d['source'][decisive]);counts=np.array([[np.sum(p&(d['source']==s)),np.sum(b&(d['source']==s))] for s in demos])
        ci=None;macro=None
        if len(counts):
            macro=float(np.mean(counts[:,0]/counts.sum(1)))
            rng=np.random.default_rng(20261008);samples=counts[rng.integers(len(counts),size=(2000,len(counts)))].sum(1)
            ci=np.quantile(samples[:,0]/samples.sum(1),[.025,.975]).tolist()
        record=dict(cluster=k,frames=int(base.sum()),demos=int(np.unique(d['source'][base]).size),contact=float(d['contact'][base].mean()),
          eligible=int(mask.sum()),progress=int(p.sum()),no_progress=int(b.sum()),ambiguous=int((mask&~decisive).sum()),
          outcome_demos=len(demos),progress_fraction=float(p.sum()/max(1,decisive.sum())),demo_macro_progress=macro,clustered_ci=ci)
        report['classes'][split].append(record)
    # Quiet means RMS Fz below 5% of the mean fixed training p99.5 scale.
    quiet=d['fz_rms'] < float(scale[:,2].mean())*.05
    m=d['valid'];y=d['contact']
    report['fz_quiet'][split]=dict(threshold=float(scale[:,2].mean())*.05,quiet_frames=int((m&quiet).sum()),quiet_contact=int((m&quiet&y).sum()),
        all_contact=int((m&y).sum()),quiet_contact_before_success=int((m&quiet&y&d['before']).sum()))
    # Direction of commanded and actual X within contact, before success; deadband avoids zero jitter.
    report['directions'][split]=[]
    for k in (8,9,10,11,12,13):
        m=d['valid']&d['before']&d['contact']&(d['cluster']==k)
        a=d['action'][m,0]
        report['directions'][split].append(dict(cluster=k,n=int(m.sum()),x_positive=int((a>2.5).sum()),x_negative=int((a< -2.5).sum()),x_small=int((abs(a)<=2.5).sum())))
for name in ('fz_rms','fz_signed','fxy_rms'):
    train=D['train'];val=D['val'];test=D['test']
    candidates=np.unique(np.quantile(train[name][train['valid']],np.linspace(0,1,201)))
    metric=[]
    for t in candidates:
        m=val['valid'];metric.append(scores(val['contact'][m],val[name][m]>t)['balanced_accuracy'])
    threshold=float(candidates[np.argmax(metric)])
    record={'threshold_val_selected':threshold}
    for split,d in D.items():
        m=d['valid'];record[split]={'auc':auc(d['contact'][m],d[name][m]),**scores(d['contact'][m],d[name][m]>threshold)}
    report['signal_detection'][name]=record
# Conditional relation of force intensity to next lack of axial progress.
report['no_progress_auc']={}
for split,d in D.items():
    m=d['valid']&d['before']&d['contact']&(d['action'][:,2]<-2)
    m &= (d['motion'][:,2]<-.4)|(d['motion'][:,2]>-.25)
    y=d['motion'][m,2]>-.25
    report['no_progress_auc'][split]={'frames':int(m.sum()),'no_progress':int(y.sum()),'fz_rms_auc':auc(y,d['fz_rms'][m]),'fxy_rms_auc':auc(y,d['fxy_rms'][m])}
report['force_motion_correlations']={}
for split,d in D.items():
    m=d['valid']&d['before']&d['contact']
    report['force_motion_correlations'][split]={name:float(spearmanr(d['fz_signed'][m],values[m]).statistic) for name,values in [('prev_dz',d['previous_motion'][:,2]),('next_dz',d['motion'][:,2]),('cmd_z',d['action'][:,2])]}
(root/'outcome_audit.json').write_text(json.dumps(report,indent=2))

out={}
for split in ('train','val','test'):
    chunks=[]
    for e,d in parts[split]:
        n=len(d['force']);f=d['force'];c,_=assignment(np.clip(f/scale[None,:,None,:],-3,3).reshape(n,840),centers)
        mask=(np.arange(n)>0)&(np.arange(n)<n-1)&d['contact']&~np.maximum.accumulate(d['success'])
        chunks.append(dict(cluster=c,source=np.full(n,e['source_episode']),mask=mask,fx=f[:,:,:,0].mean((1,2)),fz=f[:,:,:,2].mean((1,2)),
          prev_ax=np.r_[np.nan,d['action'][:-1,0]],curr_ax=d['action'][:,0],
          prev_grasp_dx=np.r_[np.nan,np.diff(d['grasp'][:,0])*1000],prev_grasp_dz=np.r_[np.nan,np.diff(d['grasp'][:,2])*1000],
          rel_x=d['relative'][:,0]))
    a={k:np.concatenate([q[k] for q in chunks]) for k in chunks[0]};m=a['mask']
    out[split]={'n':int(m.sum()),'fx_correlations':{k:float(spearmanr(a['fx'][m],a[k][m]).statistic) for k in ('prev_ax','curr_ax','prev_grasp_dx','rel_x')},
      'fz_prev_grasp_dz':float(spearmanr(a['fz'][m],a['prev_grasp_dz'][m]).statistic),'classes':{}}
    for k in (9,11,12):
        m=a['mask']&(a['cluster']==k);ax=a['curr_ax'][m];prev=a['prev_ax'][m]
        out[split]['classes'][str(k)]={'frames':int(m.sum()),'demos':int(np.unique(a['source'][m]).size),
          'previous_x_sign':[int((prev< -2.5).sum()),int((abs(prev)<=2.5).sum()),int((prev>2.5).sum())],
          'current_x_sign':[int((ax< -2.5).sum()),int((abs(ax)<=2.5).sum()),int((ax>2.5).sum())],
          'offset_x_sign':[int((a['rel_x'][m]< -.5).sum()),int((abs(a['rel_x'][m])<=.5).sum()),int((a['rel_x'][m]>.5).sum())],
          'prev_grasp_dx_median':float(np.median(a['prev_grasp_dx'][m]))}
(root/'history_correlations.json').write_text(json.dumps(out,indent=2))

h=json.loads((root/'history_correlations.json').read_text())
for split in ('train','val','test'):
    values={k:[] for k in ('fx_prev_grasp_dx','fx_prev_ax','fx_curr_ax','fx_rel_x','fz_prev_grasp_dz')}
    for e,d in parts[split]:
        f=d['force'];n=len(f);m=(np.arange(n)>0)&(np.arange(n)<n-1)&d['contact']&~np.maximum.accumulate(d['success'])
        if m.sum()<5:continue
        fx=f[:,:,:,0].mean((1,2));fz=f[:,:,:,2].mean((1,2))
        prev=np.vstack([np.full((1,3),np.nan),np.diff(d['grasp'],axis=0)*1000])
        ax=np.r_[np.nan,d['action'][:-1,0]]
        for k,x,y in [('fx_prev_grasp_dx',fx,prev[:,0]),('fx_prev_ax',fx,ax),('fx_curr_ax',fx,d['action'][:,0]),('fx_rel_x',fx,d['relative'][:,0]),('fz_prev_grasp_dz',fz,prev[:,2])]:
            if np.std(x[m])>0 and np.std(y[m])>0:values[k].append(float(spearmanr(x[m],y[m]).statistic))
    h[split]['per_demo_spearman']={k:dict(demos=len(v),median=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75))) for k,v in values.items()}
(root/'history_correlations.json').write_text(json.dumps(h,indent=2))
r=json.loads((root/'representatives.json').read_text());selected={}
for k in ('4','9','11','12'):
    selected[k]=[]
    for c in r[k]:
        d=episodes[(c['split'],c['source'])];t=c['frame']
        eligible=t>0 and t<len(d['force'])-1 and d['contact'][t] and not np.maximum.accumulate(d['success'])[t]
        if eligible:
            c={**c,'previous_action_xyz':d['action'][t-1,:3].tolist(),'relative_motion_prev_mm':((d['grasp'][t]-d['grasp'][t-1])*1000).tolist()}
            selected[k].append(c)
(root/'audited_video_examples.json').write_text(json.dumps(selected,indent=2))
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from run_contactworld_tactile_cluster_atlas import read_video_frames, composite
from run_contactworld_sensor_discovery import image_paths
from run_contactworld_matched_dynamics import encode_frames

figdir=root/'analysis_figures';figdir.mkdir(exist_ok=True)
audit=json.loads((root/'outcome_audit.json').read_text())
history=json.loads((root/'history_correlations.json').read_text())
examples=json.loads((root/'audited_video_examples.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
for split,color in [('train','#648ca7'),('val','#c79b47'),('test','#ca5a4c')]:
 rows=audit['classes'][split]
 xs=[r['cluster'] for r in rows if r['progress']+r['no_progress']>0]
 ys=[r['progress_fraction'] for r in rows if r['progress']+r['no_progress']>0]
 axes[0].plot(xs,ys,'o-',label=split,color=color,alpha=.85)
axes[0].set(xlabel='Tactile class',ylabel='Progress fraction',ylim=(-.05,1.05),xticks=range(14),title='Contact + pre-success + downward command')
axes[0].legend();axes[0].grid(alpha=.2)
metrics=['fx_prev_grasp_dx','fx_prev_ax','fx_curr_ax','fx_rel_x','fz_prev_grasp_dz']
labels=['Fx / preceding relative dx','Fx / preceding command x','Fx / current command x','Fx / socket offset x','Fz / preceding relative dz']
for i,(split,color) in enumerate([('train','#648ca7'),('val','#c79b47'),('test','#ca5a4c')]):
 axes[1].barh(np.arange(len(metrics))+(i-1)*.23,[history[split]['per_demo_spearman'][k]['median'] for k in metrics],height=.22,label=split,color=color)
axes[1].set(yticks=range(len(metrics)),yticklabels=labels,xlim=(-1,1),xlabel='Median within-demonstration Spearman correlation',title='History and geometry associations')
axes[1].axvline(0,color='gray',lw=.8);axes[1].legend()
fig.savefig(figdir/'corrected_associations.png',dpi=150);plt.close(fig)

# One continuous example retains its real temporal order and updates class IDs every frame.
d=episodes[('test',29)];n=len(d['force']);lo=58;hi=n
indices=list(range(lo,hi));paths=image_paths(args.dataset,int(d['entry']['episode_index']))
labels29,_=assignment(np.clip(d['force']/scale[None,:,None,:],-3,3).reshape(n,840),centers)
video=root/'analysis_source029.mp4'
case={'cluster':int(labels29[66]),'source':29,'frame':66,'split':'test','center_distance':0.0}
if not video.exists():
 front=read_video_frames(paths['front'],indices);wrist=read_video_frames(paths['wrist'],indices)
 left=read_video_frames(paths['tacff_left_preview'],indices);right=read_video_frames(paths['tacff_right_preview'],indices)
 def frames():
  for a,b,c,g,t in zip(front,wrist,left,right,indices):
   im=composite(a,b,c,g,d,scale,case,t)
   im[411:442,645:1275]=25
   cv2.putText(im,f'CURRENT CLASS {labels29[t]} | source 29 | frame {t}',(650,430),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,240,170),1,cv2.LINE_AA)
   im[625:660,645:1275]=25
   cv2.putText(im,'Real temporal sequence; class may change each frame',(650,645),cv2.FONT_HERSHEY_SIMPLEX,.51,(235,235,235),1)
   if t==n-1:
    im[580:615,645:1275]=25
    cv2.putText(im,'next actual plug dz: unavailable (last frame)',(650,600),cv2.FONT_HERSHEY_SIMPLEX,.53,(235,235,235),1)
   yield im
 encode_frames(video,1280,750,frames(),fps=10)

fig,axes=plt.subplots(4,1,figsize=(12,8),sharex=True,layout='constrained')
t=np.arange(lo,hi)
axes[0].plot(t,d['force'][lo:hi,:,:,0].mean((1,2)),label='mean Fx');axes[0].plot(t,d['force'][lo:hi,:,:,2].mean((1,2)),label='mean Fz');axes[0].legend();axes[0].set_ylabel('TacFF (sim units)')
axes[1].plot(t,d['action'][lo:hi,0],label='command X');axes[1].plot(t,d['action'][lo:hi,2],label='command Z');axes[1].legend();axes[1].set_ylabel('command mm/step')
axes[2].plot(t[:-1],d['next_motion'][lo:hi-1,2],label='next actual USB dz');axes[2].axhline(-.25,ls=':',color='gray');axes[2].legend();axes[2].set_ylabel('actual mm/step')
axes[3].step(t,labels29[lo:hi],where='mid',label='tactile class');axes[3].set(yticks=range(14),xlabel='source 29 control frame',ylabel='class')
for ax in axes:
 for mark in (66,72,86):ax.axvline(mark,color='gray',alpha=.5,ls='--')
 ax.grid(alpha=.15)
fig.savefig(figdir/'source029_timeline.png',dpi=140);plt.close(fig)

# Cases link to the same force-class gallery while the mechanism examples are filtered for pre-success contact.
rows=[]
for r in audit['classes']['test']:
 n=r['progress']+r['no_progress'];fraction=f"{r['progress_fraction']:.0%}" if n else '—'
 rows.append(f"<tr><td><a href='index.html#class-{r['cluster']}'>{r['cluster']}</a></td><td>{r['outcome_demos']}</td><td>{r['progress']}</td><td>{r['no_progress']}</td><td>{r['ambiguous']}</td><td>{fraction}</td></tr>")
def cards(k,limit=2):
 out=[]
 for c in examples[str(k)][:limit]:
  prev=', '.join(f'{x:+.1f}' for x in c['previous_action_xyz'])
  now=', '.join(f'{x:+.1f}' for x in c['action'][:3])
  dz=c['next_motion'][2]
  outcome=('当前不是明显下推指令' if c['action'][2]>=-2 else
           '下推后有向下进展（按本页阈值）' if dz<-.4 else
           '下推后缺乏明显向下进展' if dz>-.25 else '下推结果处于阈值之间')
  out.append(f"""<article><h3>类 {k} · {c['split']} source {c['source']} · 标记帧 {c['frame']}</h3>
<p><b>{outcome}</b>。前一步指令 XYZ：{prev}；当前指令：{now} mm/步。下一步实际 ΔZ：{c['next_motion'][2]:+.2f} mm。</p>
<img src='{html.escape(c['marker_force'])}' width='630' height='350'>
<video controls preload='none' poster='{html.escape(c['poster'])}' src='{html.escape(c['video'])}'></video>
<button onclick="const v=this.previousElementSibling;v.currentTime={c['marked_video_frame']/10};v.play()">从标记帧播放</button></article>""")
 return ''.join(out)
content=f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>14 类触觉：结果解读与口径复核</title>
<style>body{{max-width:1500px;margin:auto;padding:24px;font:17px/1.7 system-ui;background:#101827;color:#e8eff8}}a{{color:#9cd4ff}}section,article{{background:#1d2b40;padding:20px;margin:18px 0;border-radius:10px}}.notice{{background:#503c24;padding:16px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img,video{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #526079}}button{{padding:8px;margin-top:8px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>14 类触觉能告诉我们什么</h1><p><a href='index.html'>返回全部 14 类与 70 段代表视频</a></p>
<p class='notice'>复核发现：原网页“受阻”只是“下推但下一步没有明显下移”的简化统计，混入了成功后的停留和末尾缺少后续观测的帧。它不能直接等同机械卡阻。本页统计只使用：已接触、首次环境成功之前、有真实下一帧且当前下推的样本。聚类和类编号没有改。</p>
<p>范围是 120 条重放示范（95 训练、15 验证、10 测试）。这里的测试指留出示范，不是之前 13 个策略 × 40 个初态的闭环评测。动作是重放时的位姿跟踪控制量。</p>
<section><h2>1. 类 9 是比较稳定的“接触且下推缺乏进展”模式</h2>
<p>类 9 的两指 Fz 类中心为明显正值。在接触、未成功的有效下推帧中，训练为 32 推进 / 114 缺乏推进，验证为 5 / 19，测试为 3 / 30；测试覆盖 9 条独立示范，另有 1 帧在阈值中间。训练、验证、测试的推进比例约为 22%、21%、9%。因此这是有跨示范支持的运动受限线索，但不能单凭它判定卡在哪里或应该往哪边修正。</p>
<p>作为对照，同样限制在接触、未成功、明确下推结果的全部帧中，缺乏推进的比例为训练 57%、验证 58%、测试 62%；类 9 分别为 78%、79%、91%。下面同时保留不推进例子和推进反例；source 29 的 −0.47 mm 刚超过本页 −0.4 mm 阈值，不应把它称作顺利插入。</p>
<div class='grid'>{cards(9)}</div></section>
<section><h2>2. Fz 小也可能已接触；单独 Fz 不能充当完美开关</h2>
<p>用验证集选出的 Fz RMS 阈值检测插座接触，在测试集 283 个已接触帧中检出 175、漏掉 108；451 个未接触帧中误报 19。即触发时约 90% 确实接触，但仅检出约 62% 的接触帧。Fz 强通常有提示作用，Fz 弱不能排除接触。这里使用的是双指全部 taxel 的 RMS，未使用插座真值作为输入。</p>
<p>类 4 是直观反例：Fz 类中心很弱，但测试类内约 76% 的有效帧已接触；在接触且尚未成功的下推帧里，有 23 帧推进、25 帧未明显推进。同一触觉类仍可能对应不同的运动结果。</p>
<div class='grid'>{cards(4)}</div></section>
<section><h2>3. Fx 存在规律，但首先要解释动作历史</h2>
<p>类 11 的 Fx 类中心偏正，类 12 偏负。在接触且未成功的测试帧中，类 11 当前 X 指令为 +X 8 帧、−X 1 帧、小幅 9 帧；类 12 为 +X 1 帧、−X 6 帧、小幅 4 帧。训练和验证中也有相同趋势。这是值得继续研究的方向关联。</p>
<p>但这两个类同样明显关联前一步动作。逐条示范计算后，测试示范的相关系数中位数为：Fx 与前一步 X 指令 +0.56，与当前 X 指令 +0.26，与前一步 Δ(USB−夹爪)X 为 −0.61，与当前插座相对 X 偏移为 +0.28。Fz 与前一步 Δ(USB−夹爪)Z 为 +0.73。相对运动量是 base 下参考点之差的变化，没有完整扣除刚体旋转，不能直接当成胶垫真实滑移。</p>
<p>因此，更有证据的解释是：力场包含近期夹持相对运动/动作响应的信息；尚不能写成“Fx 正就应该向正 X 修正”。前一步与当前动作往往连续，本身就会制造下一动作相关性。</p>
<div class='grid'>{cards(11,1)}{cards(12,1)}</div></section>
<section><h2>同一条真实示范中的连续变化</h2>
<p>source 29：标记帧 66 属于类 12、72 属于类 9、86 属于类 11。下面完整保留这一段真实时间顺序，并在每帧更新当前类编号。请同时看 Fx/Fz、动作和下一步实际 Z 位移；图中虚线对应三个标记帧。各类图库原视频的“CLUSTER”是代表帧所属类，不表示整段视频始终属于该类。</p>
<video controls preload='metadata' src='analysis_source029.mp4'></video><img src='analysis_figures/source029_timeline.png'></section>
<section><h2>修正后的对照统计</h2>
<p>命令 Z &lt; −2 mm/步视为明显下推；下一步 USB 相对插座 Z 位移 &lt; −0.4 mm 视为推进，&gt; −0.25 mm 视为缺乏向下进展，二者之间单独列为中间结果。侧移、回撤以及转动也会影响这个指标；它不是纯下插可行性实验。独立示范很少的类不能据百分比下结论。</p>
<table><tr><th>类</th><th>有明确结果的测试示范数</th><th>推进帧</th><th>缺乏推进帧</th><th>中间帧</th><th>明确结果中的推进率</th></tr>{''.join(rows)}</table>
<img src='analysis_figures/corrected_associations.png'>
<p>图中每帧比例只作描述；相关性图先按每条示范计算再取中位数。逐类统计文件另存按示范重采样的区间，单条示范的退化区间不具总体置信含义。</p></section>
<section><h2>下一步最有价值的检验</h2><p>先固定当前位姿、前几步动作与实际运动，再比较“历史本身”和“历史＋当前触觉”对后续是否推进及横向调整的解释能力。这能检验触觉是否提供已知历史之外的增量信息。再针对类 9 或类 11/12 的代表状态做同状态不同动作分支，验证哪边调整真正有效。单靠聚类无法证明视觉无用、Fy 是噪声，或触觉能唯一决定正确动作。</p>
<p>分析文件：<a href='outcome_audit.json'>修正统计与接触检测</a> · <a href='history_correlations.json'>动作历史与逐示范相关性</a> · <a href='audited_video_examples.json'>已核对的视频例子</a>。</p></section></html>"""
(root/'analysis.html').write_text(content)
index=root/'index.html';s=index.read_text()
notice="<div id='outcome-audit-notice' class='note'><b>已完成结果复核：</b><a href='analysis.html'>阅读 14 类分析（含原始视频与动作历史）</a>。原文“受阻”统计含成功后停留和末尾缺测，不能直接视为机械卡阻；修正后的接触条件统计见分析页。</div>"
if "id='outcome-audit-notice'" not in s:
 pos=s.index('</h1>')+5;s=s[:pos]+notice+s[pos:];index.write_text(s)
print('ANALYSIS',root/'analysis.html')

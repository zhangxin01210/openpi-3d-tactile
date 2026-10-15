#!/usr/bin/env python3
"""Compare initial evaluation RGB with training RGB and inspect three mechanisms."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def first_frame(path):
    cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError(f'Video decode failed: {path}')
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


def features(rgb):
    image=cv2.resize(rgb,(64,64),interpolation=cv2.INTER_AREA)
    lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1)/255


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('root',type=Path)
    pa.add_argument('--dataset',type=Path,default=Path('data/contactworld_usb_v3_bilateral'))
    args=pa.parse_args();root=args.root;out=root/'training_neighbors'/'alias_followup';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.dataset/'spatial/manifest.json').read_text())
    entries=[e for e in manifest['episodes'] if e['split']=='train']
    tf=[];tw=[]
    for e in entries:
        ep=e['episode_index']
        tf.append(first_frame(args.dataset/f'videos/chunk-000/observation.images.front/episode_{ep:06d}.mp4'))
        tw.append(first_frame(args.dataset/f'videos/chunk-000/observation.images.wrist/episode_{ep:06d}.mp4'))
    feat_f=np.stack([features(x) for x in tf]);feat_w=np.stack([features(x) for x in tw]);records=[]
    for seed in range(10000,10040):
        with np.load(root/f'pi0_cw3_usb_01_rgb/seed_{seed}/inputs.npz') as z:front=z['front'][0];wrist=z['wrist'][0]
        dfront=np.mean((feat_f-features(front))**2,axis=1)**.5
        dwrist=np.mean((feat_w-features(wrist))**2,axis=1)**.5
        dboth=np.sqrt((dfront*dfront+dwrist*dwrist)/2)
        geom=next(x for x in json.loads((root/'training_neighbors'/'analysis.json').read_text())['neighbors'] if x['seed']==seed)
        src=geom['nearest_source_episode'];ind=next(i for i,e in enumerate(entries) if e['source_episode']==src)
        r={'seed':seed,'geometry_source':src,'visual_front_rank':int(np.argsort(dfront).tolist().index(ind)+1),'visual_wrist_rank':int(np.argsort(dwrist).tolist().index(ind)+1),'visual_both_rank':int(np.argsort(dboth).tolist().index(ind)+1),'visual_front_error':float(dfront[ind]),'visual_both_error':float(dboth[ind]),'closest_visual_source':entries[int(np.argmin(dboth))]['source_episode'],'closest_visual_error':float(dboth.min())}
        records.append(r)
        if seed not in (10005,10017,10034):continue
        W=768;H=885;canvas=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(canvas)
        another=int(np.argmin(dboth))
        for ri,(name,f,w) in enumerate([('test',front,wrist),(f'geometry neighbor source {src}',tf[ind],tw[ind]),(f'pixel neighbor source {entries[another]["source_episode"]}',tf[another],tw[another])]):
            y=ri*295;draw.text((10,y+8),f'{seed} {name}',fill='black')
            canvas.paste(Image.fromarray(f),(10,y+35));canvas.paste(Image.fromarray(w),(390,y+35))
            draw.text((10,y+270),'front',fill='black');draw.text((390,y+270),'wrist',fill='black')
        canvas.save(out/f'seed_{seed}_initial_rgb.jpg',quality=92)
    (out/'visual_distance.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps([x for x in records if x['seed'] in (10005,10017,10034)],indent=2))


if __name__=='__main__':main()

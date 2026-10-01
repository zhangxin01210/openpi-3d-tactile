"""Render recorded/live replay with aligned pre-action traces (H264/yuv420p)."""
import argparse,json,subprocess
from pathlib import Path
import cv2,numpy as np
from audit_contactworld import chart
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--demo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.run/'replay.json').read_text());d=np.load(a.demo);live=np.load(a.run/'pre_states.npz');n=len(live['ee_pos'])
errors=np.stack([np.linalg.norm(live[k]-d[k][:n],axis=1)*1000 for k in ['ee_pos','plug_pos']],1)
grip=np.stack([d['dof_pos'][:n,-2:].mean(1),live['dof_pos'][:,-2:].mean(1)],1)*1000
ff=np.stack([np.abs(d['tactile_force_field_right'][:n]).mean((1,2,3)),np.abs(live['tactile_force_field_right']).mean((1,2,3))],1)
force=np.array([np.linalg.norm(x['plug_socket_force_post']) for x in r['rows']]);first=int(np.flatnonzero(force>1e-5)[0]);div=int(np.flatnonzero(errors[:,1]>1)[0])
a.output.parent.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(str(a.run/'recorded_left_replay_right.mp4'))
proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x960','-r','5','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
for i in range(n):
 ok,im=cap.read();assert ok
 canvas=np.zeros((960,1280,3),np.uint8);canvas[:640,:640]=cv2.resize(im,(640,640))
 cv2.putText(canvas,('RECORDED | LIVE: POSE FEEDBACK' if r.get('pose_feedback') else 'RECORDED | LIVE: OPEN LOOP'),(15,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
 for j,(v,title,labels) in enumerate([(errors,'Pre-action position error (mm)',['EE','plug']),(grip,'Mean finger joint position (mm)',['recorded','live']),(ff,'Mean absolute TacFF (raw units)',['recorded','live'])]):
  canvas[j*320:(j+1)*320,640:]=cv2.resize(chart(v,i,title,labels,markers=[first,div]),(640,320))
 canvas[640:,:640]=cv2.resize(chart(d['action'][:n,:3],i,'Recorded XYZ action (dimensionless)',['x','y','z'],markers=[first,div]),(640,320))
 cv2.putText(canvas,f'frame {i} | live contact after action {first} | plug error >1mm at {div}',(15,622),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
 proc.stdin.write(canvas.tobytes())
 if i in [0,60,69,75]:cv2.imwrite(str(a.output.with_name(a.output.stem+'_frame_%03d.jpg'%i)),canvas)
proc.stdin.close();assert proc.wait()==0;cap.release()
metrics={'first_live_contact_after_action':first,'first_pre_action_plug_error_over_1mm':div,'max_pre_action_errors_mm':errors.max(0).tolist(),'first_gripper_mean_mm':grip[0].tolist(),'video_fps':5,'note':'Playback slowed; chart indices are authoritative. TacFF and live contact force are distinct signals.'}
a.output.with_suffix('.json').write_text(json.dumps(metrics,indent=2)+'\n');print(metrics)

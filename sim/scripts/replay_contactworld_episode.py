"""Complete USB/Peg replay; invoked by probe_contactworld_usb.py.
Records disagreement rather than declaring recorded-state restoration exact.
"""
import json
import hashlib
import subprocess
import numpy as np
import torch
from isaacgym import gymapi, gymtorch


def replay(env, demo_path, output, clear_actuation=False, staged_reset=False, restore_dt=None, start_frame=0, white_gel=False, white_mounts=False, socket_away=False, estimated_velocity=False, pose_feedback=False, hold_after=0):
    archive = np.load(demo_path)
    demo = {k:archive[k][start_frame:] for k in archive.files}
    if white_gel:
        for name in ['elastomer_left','elastomer_right']:
            body=env.gym.find_actor_rigid_body_index(env.env_ptrs[0],env.actor_handles['franka'],name,gymapi.DOMAIN_ACTOR)
            env.gym.set_rigid_body_color(env.env_ptrs[0],env.actor_handles['franka'],body,gymapi.MESH_VISUAL,gymapi.Vec3(1,1,1))
    if white_mounts:
        for name in ['mount_left','mount_right']:
            body=env.gym.find_actor_rigid_body_index(env.env_ptrs[0],env.actor_handles['franka'],name,gymapi.DOMAIN_ACTOR)
            env.gym.set_rigid_body_color(env.env_ptrs[0],env.actor_handles['franka'],body,gymapi.MESH_VISUAL,gymapi.Vec3(1,1,1))
    n = len(demo['action'])
    fields = ['dof_pos', 'dof_vel', 'plug_pos', 'plug_quat', 'socket_quat']
    init = {k: demo[k][0:1] for k in fields}
    init['socket_pos'] = demo['socket_pos_gt'][0:1].copy()
    if socket_away:
        init['socket_pos'][:,0] += 1.0
    if clear_actuation:
        env.ctrl_target_dof_pos.copy_(torch.as_tensor(init['dof_pos'],device=env.device))
        env._reset_franka_actuation(env.ctrl_target_dof_pos)
    if staged_reset:
        # Update articulated link transforms with the plug away from the fingers
        # before restoring the grasp. This is a diagnostic reset protocol only.
        away = dict(init)
        away['plug_pos'] = init['plug_pos'].copy() + np.array([[0,0,1]],dtype=np.float32)
        env.reset_to_dataset_state(**away)
        env.ctrl_target_dof_pos.copy_(torch.as_tensor(init['dof_pos'],device=env.device))
        env._reset_franka_actuation(env.ctrl_target_dof_pos)
    old_params = env.gym.get_sim_params(env.sim)
    old_dt = old_params.dt
    if restore_dt is not None:
        old_params.dt = restore_dt
        env.gym.set_sim_params(env.sim,old_params)
    if estimated_velocity:
        from scipy.spatial.transform import Rotation
        lo=max(0,start_frame-1);hi=min(len(archive['action'])-1,start_frame+1)
        elapsed=(hi-lo)*float(old_dt)*env.control_freq_inv
        linear=(archive['plug_pos'][hi]-archive['plug_pos'][lo])/elapsed
        angular=(Rotation.from_quat(archive['plug_quat'][hi])*Rotation.from_quat(archive['plug_quat'][lo]).inv()).as_rotvec()/elapsed
        env.dof_pos[:]=torch.as_tensor(init['dof_pos'],device=env.device)
        env.dof_vel[:]=torch.as_tensor(init['dof_vel'],device=env.device)
        env.ctrl_target_dof_pos[:]=env.dof_pos
        ids=env.actor_ids_sim_tensors['franka'].to(dtype=torch.int32)
        env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(ids),len(ids))
        for actor in ['plug','socket']:
            idx=getattr(env,actor+'_actor_id_env')
            env.root_pos[:,idx,:]=torch.as_tensor(init[actor+'_pos'],device=env.device)
            env.root_quat[:,idx,:]=torch.as_tensor(init[actor+'_quat'],device=env.device)
            env.root_linvel[:,idx,:]=torch.as_tensor(linear if actor=='plug' else np.zeros(3),device=env.device)
            env.root_angvel[:,idx,:]=torch.as_tensor(angular if actor=='plug' else np.zeros(3),device=env.device)
        ids=torch.cat([env.actor_ids_sim_tensors[k] for k in ['plug','socket']]).to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state),gymtorch.unwrap_tensor(ids),len(ids))
        env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True)
        env.refresh_all_tensors();env.compute_observations()
        (output/'estimated_velocity.json').write_text(json.dumps(dict(linear=linear.tolist(),angular=angular.tolist(),frame_pair=[lo,hi],dt_assumption_s=elapsed),indent=2)+'\n')
    else:
        env.reset_to_dataset_state(**init)
    if restore_dt is not None:
        old_params.dt = old_dt
        env.gym.set_sim_params(env.sim,old_params)
    def snapshot():
        return {k: v[0].detach().cpu().numpy().copy() for k, v in env.obs_dict.items()}
    before = snapshot()
    initial_audit = {
        'expected': {k: np.asarray(v).tolist() for k,v in init.items()},
        'observed': {k: before[k].tolist() for k in ['dof_pos','dof_vel','ee_pos','ee_quat','plug_pos','plug_quat','socket_pos_gt','socket_quat']},
        'body_positions': env.body_pos[0].detach().cpu().numpy().tolist(),
        'actor_roots': env.root_state.detach().cpu().numpy().tolist(),
        'plug_body_index': int(env.plug_body_id_env),
        'socket_body_index': int(env.socket_body_id_env),
    }
    (output/'initial_audit.json').write_text(json.dumps(initial_audit,indent=2)+'\n')
    np.savez_compressed(output/'initial_views.npz',front=before['front'],wrist=before['wrist'],recorded_front=demo['front'][0],recorded_wrist=demo['wrist'][0])
    initial_errors = {k: float(np.linalg.norm(before[k]-demo[k][0])) for k in ['dof_pos','dof_vel','plug_pos','ee_pos']}
    rows, states, matrices, pre_states = [], [], [], []
    properties = {}
    for actor in ['franka','plug','socket']:
        handle = env.actor_handles[actor]
        props = env.gym.get_actor_rigid_shape_properties(env.env_ptrs[0],handle)
        properties[actor] = [{k:float(getattr(q,k)) for k in ['friction','rolling_friction','torsion_friction','restitution','compliance','compliant_damping']} for q in props]
    (output/'contact_properties.json').write_text(json.dumps(properties,indent=2)+'\n')
    frames = []
    dt = float(env.gym.get_sim_params(env.sim).dt)*env.control_freq_inv
    for i in range(n):
        # Dataset row i is compared to pre-action observation; post-action to i+1.
        pre = snapshot()
        pre_states.append({k:pre[k] for k in ['ee_pos','ee_quat','plug_pos','plug_quat','dof_pos','tactile_force_field_right']})
        executed = demo['action'][i].copy()
        if pose_feedback:
            from scipy.spatial.transform import Rotation
            ps=np.asarray(env.cfg_task.rl.pos_action_scale)
            rs=np.asarray(env.cfg_task.rl.rot_action_scale)
            target_position=demo['ee_pos'][i]+demo['action'][i,:3]*ps
            target_rotation=Rotation.from_rotvec(demo['action'][i,3:6]*rs)*Rotation.from_quat(demo['ee_quat'][i])
            executed[:3]=(target_position-pre['ee_pos'])/ps
            executed[3:6]=(target_rotation*Rotation.from_quat(pre['ee_quat']).inv()).as_rotvec()/rs
        action = torch.tensor(executed[None], device=env.device)
        obs, _, _, _ = env.step(action)
        current = snapshot()
        assert all(np.isfinite(v).all() for v in current.values())
        images = env.get_camera_image_tensors_dict()
        view = np.asarray(env.gym.get_camera_view_matrix(env.sim,env.env_ptrs[0],env.camera_handles_list[0]['front_depth']))
        proj = np.asarray(env.gym.get_camera_proj_matrix(env.sim,env.env_ptrs[0],env.camera_handles_list[0]['front_depth']))
        # Invert the RELEASED conversion, including its column/row convention.
        # This checks online XYZ/RGB/depth synchronization, not metric correctness.
        assert np.linalg.norm(env.franka_base_pos[0].cpu().numpy()) < 1e-6
        assert np.linalg.norm(env.franka_base_quat[0].cpu().numpy()-[0,0,0,1]) < 1e-6
        pc = current['pointcloud']
        cam = np.c_[pc[:,:3],np.ones(len(pc))] @ view.T
        f = np.array([proj[0,0]*128,proj[1,1]*128])
        uv = cam[:,:2]/cam[:,2:3]*f+128
        pix = np.rint(uv).astype(int)
        valid = (pix>=0).all(1)&(pix<256).all(1)
        p=pix[valid]
        rgb=images['front'][0].cpu().numpy()/255.
        depth=images['front_depth'][0].cpu().numpy()
        row = dict(frame=i,simulation_time_after_action_s=(i+1)*dt,
            ee_pre_error_mm=float(np.linalg.norm(pre['ee_pos']-demo['ee_pos'][i])*1000),
            plug_pre_error_mm=float(np.linalg.norm(pre['plug_pos']-demo['plug_pos'][i])*1000),
            ff_pre_mae=float(np.abs(pre['tactile_force_field_right']-demo['tactile_force_field_right'][i]).mean()),
            ff_max=float(np.abs(current['tactile_force_field_right']).max()),
            cloud_valid_count=int(valid.sum()),cloud_rgb_mae=float(np.abs(pc[valid,3:]-rgb[p[:,1],p[:,0]]).mean()),
            cloud_depth_error_m=float(np.abs(cam[valid,2]-depth[p[:,1],p[:,0]]).max()),
            cloud_pixel_residual=float(np.abs(uv[valid]-p).max()),
            source_success=bool(env._check_success()[0].item()))
        for key in ('front', 'wrist', 'tactile_rgb_right', 'tactile_depth_right'):
            if key in demo and key in pre:
                row[key + '_pre_mae'] = float(np.abs(pre[key] - demo[key][i]).mean())
        if i+1<n:
            row['ee_next_error_mm']=float(np.linalg.norm(current['ee_pos']-demo['ee_pos'][i+1])*1000)
            row['plug_next_error_mm']=float(np.linalg.norm(current['plug_pos']-demo['plug_pos'][i+1])*1000)
        row['executed_action']=executed.tolist()
        row['recorded_action']=demo['action'][i].tolist()
        row['gripper_pre_m'] = pre['dof_pos'][-2:].tolist()
        row['plug_socket_force_post'] = env.contact_force_pairwise[0,env.plug_body_id_env,env.socket_body_id_env].detach().cpu().numpy().tolist()
        rows.append(row)
        states.append({k:current[k] for k in ['ee_pos','ee_quat','plug_pos','plug_quat','dof_pos','tactile_force_field_right']})
        mat={}
        for name in ['front_depth','wrist_depth']:
            handle=env.camera_handles_list[0][name]
            mat['view_'+name]=np.asarray(env.gym.get_camera_view_matrix(env.sim,env.env_ptrs[0],handle))
            mat['projection_'+name]=np.asarray(env.gym.get_camera_proj_matrix(env.sim,env.env_ptrs[0],handle))
        matrices.append(mat)
        # Same row: recorded pre-action view versus live pre-action view.
        frame=np.concatenate([np.concatenate([demo['front'][i],pre['front']],axis=1),np.concatenate([demo['wrist'][i],pre['wrist']],axis=1)],axis=0)
        frames.append(np.uint8(np.clip(frame*255,0,255)))
        if i%20==0:print('replay',i,row,flush=True)
    np.savez_compressed(output/'pre_states.npz',**{k:np.stack([s[k] for s in pre_states]) for k in pre_states[0]})
    video=output/'recorded_left_replay_right.mp4'
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','512x512','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    for frame in frames:proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    np.savez_compressed(output/'replay_states.npz',**{k:np.stack([s[k] for s in states]) for k in states[0]},**{k:np.stack([m[k] for m in matrices]) for k in matrices[0]})
    post_hold = []
    if hold_after:
        from scipy.spatial.transform import Rotation
        for hold_index in range(hold_after):
            env.step(torch.zeros((1, env.num_actions), device=env.device))
            current = snapshot()
            delta = current['plug_pos']-current['socket_pos_gt']
            plug_axis = Rotation.from_quat(current['plug_quat']).apply([0, 0, 1])
            socket_axis = Rotation.from_quat(current['socket_quat']).apply([0, 0, 1])
            axis_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(plug_axis, socket_axis)), 0, 1))))
            xyz_mm = float(np.linalg.norm(delta)*1000)
            z_mm = float(delta[2]*1000)
            success = bool(env._check_success()[0].item())
            post_hold.append(dict(step=hold_index+1, source_success=success,
                                  plug_socket_xyz_error_mm=xyz_mm,
                                  plug_socket_z_mm=z_mm,
                                  unsigned_axis_error_deg=axis_deg,
                                  strict_pose=success and xyz_mm<2 and z_mm<5 and axis_deg<5))
    report=dict(status='executed_all_actions',frames=n,demo=str(demo_path),
        demo_sha256=hashlib.sha256(demo_path.read_bytes()).hexdigest(),
        initial_errors=initial_errors,clear_reset_actuation=clear_actuation,staged_reset=staged_reset,restore_dt=restore_dt,
        pose_feedback=pose_feedback,socket_away=socket_away,estimated_velocity=estimated_velocity,start_frame=start_frame,white_gel=white_gel,white_mounts=white_mounts,control_dt_s=dt,playback_fps=10,playback_is_not_verified_collection_time=True,
        action_semantics=('closed-loop tracking of recorded EE control targets; executed actions differ from dataset' if pose_feedback else 'recorded scaled relative XYZ and left-multiplied axis-angle rotation; no action substitution'),
        pos_action_scale=list(env.cfg_task.rl.pos_action_scale),rot_action_scale=list(env.cfg_task.rl.rot_action_scale),
        final_source_success=rows[-1]['source_success'],
        post_hold=post_hold,
        post_hold_all_strict=bool(post_hold) and all(x['strict_pose'] for x in post_hold),
        final_ee_error_mm=float(np.linalg.norm(states[-1]['ee_pos']-demo['ee_pos'][-1])*1000),
        final_plug_error_mm=float(np.linalg.norm(states[-1]['plug_pos']-demo['plug_pos'][-1])*1000),
        source_pointcloud_alignment_passed=all(r['cloud_valid_count']==1024 and r['cloud_rgb_mae']<1e-5 and r['cloud_depth_error_m']<1e-5 for r in rows),
        limitations=['Initial restore advances physics one step; object velocities and contact solver history unavailable',
                     'Source pointcloud convention remains incorrect as a world/base metric cloud; matching RGB does not fix that',
                     'One episode only; no success-rate or tactile recovery claim'],rows=rows)
    (output/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)

"""CPU tests for official press-button RGBTrack orchestration."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / 'scripts/preprocess/rgbtrack_task.py'
_spec = importlib.util.spec_from_file_location('rgbtrack_task', SCRIPT)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def jwrite(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def fixture(tmp_path):
    repo = tmp_path / 'repo'
    task = '1028_press_button'
    dataset = repo / 'data' / task
    dataset.mkdir(parents=True)
    base = repo / 'output/preprocess' / task
    objects = ['button', 'box']
    jwrite(base/'sam2/manifest.json', {'dataset': str(dataset), 'objects': objects, 'camera': 'front', 'episode_ids': [0,23,50,76,99]})
    assets = {}
    for o in objects:
        p = base / f'geometry/approved/{o}.glb'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((o + ' fake GLB').encode())
        assets[o] = {'sha256': rt.digest(p), 'scale_m_per_unit':1.}
    jwrite(base / 'geometry/approved/manifest.json', {'dataset_id': task,'objects': objects, 'status':'human_accepted','assets': assets})
    for ep in [0,23,50,76,99]:
        root = base / f'sam2/episodes/e{ep:05d}'
        hashes={}
        for o in objects:
            stream=hashlib.sha256()
            for f in range(2):
                p=root/f'masks/{o}/{f:05d}.png'
                p.parent.mkdir(parents=True, exist_ok=True)
                byts=f'{ep}-{o}-{f}'.encode()
                p.write_bytes(byts)
                stream.update(byts)
            hashes[o]=stream.hexdigest()
        jwrite(root/'tracking_report.json',{'status':'tracking_complete_unreviewed','episode':ep,'objects':objects,'frames':2,'mask_stream_sha256':hashes})
    # Fake source files so that code fingerprints can be checked without importing CUDA.
    for rel in ('configs/ur7e_xhand/cameras_0903.json', 'configs/ur7e_xhand/extrinsic_0909.yaml',
                'configs/ur7e_xhand/front_calibration_candidate.json',
                'scripts/object_geometry/pose_tracking/rgbtrack_adapter.py',
                'src/openpi/spatial/rgbd_registration.py',
                'scripts/preprocess/rgbtrack_static_policy.py',
                'scripts/preprocess/rgbtrack_task_runner.py',
                'scripts/preprocess/rgbtrack_task.py'):
        p=repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('"--mesh", action="append"\n"--skip-preview-html"\n' if 'adapter' in rel else 'source content')
    return repo, base


def test_inputs_roles_and_episode_selection(tmp_path):
    repo, _ = fixture(tmp_path)
    info=rt.task_inputs(repo, '1028_press_button')
    assert info['objects']==['button','box']
    assert rt.select_episodes(None, info['episode_ids'], trial=True)==[0,23,50,76,99]
    assert rt.verify_episode(info,0)[1]==2
    rt.main(['--repo',str(repo),'--dataset','1028_press_button','check','--episodes','0,23'])


def test_policy_static_and_dynamic_are_distinct(tmp_path):
    repo, _=fixture(tmp_path)
    info=rt.task_inputs(repo, '1028_press_button')
    class Args:
        static_object='box'; dynamic_object='button'; mode='depth'
        init_tilt_deg=25.;init_rank=150;min_mask_iou=.35
    pol=rt.policy_settings(Args(),info)
    assert pol['method']=='static_upright_convex_mask_v1'
    assert pol['dynamic_object']=='button'
    Args.static_object='button'
    try:
        rt.policy_settings(Args(),info)
    except ValueError:
        pass
    else:
        assert False, 'Cannot configure same object static and dynamic'


def test_provenance_invalidated_by_calibration_change(tmp_path):
    repo, base=fixture(tmp_path)
    info=rt.task_inputs(repo,'1028_press_button')
    cal=repo/'configs/ur7e_xhand/front_calibration_candidate.json'
    p=rt.source_provenance(info,0,cal,{'static_object':'box'})
    assert p['schema_version']==2
    (repo/'configs/ur7e_xhand/cameras_0903.json').write_text('rotated again')
    q=rt.source_provenance(info,0,cal,{'static_object':'box'})
    assert p!=q


def test_dry_run_creates_no_inference_directory(tmp_path,monkeypatch):
    repo,base=fixture(tmp_path)
    # CPU-only dry-run skips checking calibrated runtime; outputs planned runner command.
    rt.main(['--repo',str(repo),'--dataset','1028_press_button','trial',
             '--episodes','0','--static-object','box','--dynamic-object','button',
             '--run-tag','press_sdk_v1','--dry-run'])
    assert not (base/'rgbtrack/runs/press_sdk_v1/episodes/e00000').exists()


def test_completed_provenance_and_frame_count(tmp_path):
    repo,base=fixture(tmp_path)
    info=rt.task_inputs(repo,'1028_press_button')
    cal=repo/'configs/ur7e_xhand/front_calibration_candidate.json'
    policy={'static_object':'box'}
    masks,n=rt.verify_episode(info,0)
    out=base/'rgbtrack/runs/press_sdk_v1/episodes/e00000'
    out.mkdir(parents=True)
    jwrite(out/'manifest.json',{'frames_written':n,'mask_root':str(masks),
        'mesh_conversions':{o:{'source_sha256':rt.digest(info['meshes'][o])} for o in info['objects']}})
    jwrite(out/'task_input_provenance.json',rt.source_provenance(info,0,cal,policy))
    for o in info['objects']:
        (out/f'{o}_poses.jsonl').write_text('{"frame_index":0}\n{"frame_index":1}\n')
    assert rt.completed(info,0,out,masks,n,cal,policy)
    (repo/'configs/ur7e_xhand/cameras_0903.json').write_text('changed')
    try:
        rt.completed(info,0,out,masks,n,cal,policy)
    except FileExistsError:
        pass
    else:
        assert False, 'Stale result should never silently resume'

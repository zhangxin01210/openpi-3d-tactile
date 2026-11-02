"""CPU offline chain: real worker/bridge/WebSocket/client, mocked vision weights and mesh."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import threading
import types

import numpy as np
import pytest
import websockets.asyncio.server

from scripts.deploy import run_online_perception as worker
from scripts.deploy.websocket_hand_scene_bridge import HandScenePerceptionBridge
from scripts.deploy.test_websocket_hand_scene_bridge import frame, observation
from deploy.ur7e_xhand_deploy_3d_tactile_multi import WebsocketClientPolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess, XHandSpatialOnlineRepack
from openpi.policies.xhand_policy import XHandInputs
from openpi.models.model import ModelType

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('task,names', [
    ('1028_press_button', ['button', 'box']),
    ('1030_yellow_socket', ['red', 'yellow']),
])
def test_worker_bridge_and_native_websocket(task, names, tmp_path, monkeypatch):
    from scripts.preprocess import sam2_task, export_object_geometry_rgbtrack
    from scripts.deploy import sam2_video_stream
    root = tmp_path / 'repo'
    approved = root / 'output/preprocess' / task / 'geometry/approved'
    approved.mkdir(parents=True)
    release_path = approved.parent.parent / 'rgbtrack/release_v1.json'
    release_path.parent.mkdir()
    entries = {}
    for name in names:
        path = approved / f'{name}.glb'
        path.write_bytes(name.encode())
        entries[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scale_m_per_unit': 1.}
    (approved / 'manifest.json').write_text(json.dumps({'status': 'human_accepted', 'dataset_id': task,
        'objects': names, 'assets': entries}))
    policy = {'method': 'static_upright_convex_mask_v1', 'mode': 'depth',
              'dynamic_object': names[0], 'static_object': names[1],
              'init_tilt_deg': 25., 'init_rank': 150, 'min_mask_iou': .35}
    release_path.write_text(json.dumps({'dataset': task, 'objects': names, 'policy': policy}))
    refs = root / 'assets/perception/sam2_references'
    refs.parent.mkdir(parents=True)
    refs.symlink_to(ROOT / 'assets/perception/sam2_references', target_is_directory=True)
    spec = worker._task_geometry_inputs(approved / 'manifest.json', root)

    # Mock only unavailable SAM2/RGBTrack, RGB-D camera registration and metric mesh decoding.
    monkeypatch.setattr(sam2_task, 'image_predictor', lambda _: object())
    calls = iter(names)
    def predict(*args):
        name = next(calls)
        mask = np.zeros(args[1].shape[:2], np.uint8)
        mask[30:60, 30:60] = 1 if name == names[0] else 0
        if name == names[1]: mask[100:130, 100:130] = 1
        return mask, .9, {'second_vs_best_ratio': 2.}
    monkeypatch.setattr(sam2_task, 'predict_object', predict)
    class Video:
        def __init__(self, **kwargs): pass
        def predict(self, index, rgb, seeds):
            return {name: (seeds[name + '_seed_mask'].astype(bool), -1., 'seed', True, False)
                    for name in names}, 0
    monkeypatch.setattr(sam2_video_stream, 'CausalVideoSAM2Stream', Video)
    monkeypatch.setattr(export_object_geometry_rgbtrack, 'load_surface_points',
                        lambda path, seed: np.full((512, 3), seed % 7 / 100, np.float32))
    camera = types.SimpleNamespace(depth_scale_m_per_unit=.001, K_color=np.eye(3), T_base_color=np.eye(4),
                                   color_intrinsics=types.SimpleNamespace(fx=1., fy=1., cx=0., cy=0.))
    class Tracker:
        def __init__(self, **kwargs): self.name = kwargs['object_name']
        def estimate(self, **kwargs):
            return types.SimpleNamespace(valid=True, confidence=.9, T_base_object=np.eye(4))
    adapter = types.ModuleType('rgbtrack_adapter')
    adapter.load_front_calibration = lambda *a: camera
    adapter._load_front_camera_for_depth_registration = lambda *a: camera
    adapter._rgb_registered_depth = lambda depth, cam, shape: depth.astype(np.float32) * cam.depth_scale_m_per_unit
    adapter.convert_glb_to_rgbtrack_mesh = lambda **kw: {'output_mesh': kw['source_glb']}
    adapter.RGBTrackObject = Tracker
    monkeypatch.setitem(sys.modules, 'rgbtrack_adapter', adapter)
    static = types.ModuleType('rgbtrack_static_policy')
    static.install_static_policy = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'rgbtrack_static_policy', static)
    checkpoint = tmp_path / 'sam.pt'; checkpoint.write_text('fixture')
    args = types.SimpleNamespace(sam2_backend='video_memory', register_missing_color_depth=True,
        sam2_config='configs/sam2.1/sam2.1_hiera_s.yaml', sam2_source=tmp_path, sam2_checkpoint=checkpoint,
        front_calibration_profile=Path('fixture'), weights_root=tmp_path, min_mask_area_px=1,
        max_mask_fraction=.8, min_sam2_score=.05, device='cpu', follow=False, max_frames=1)

    # Real spatial/tactile/FK preprocessor and the actual XHand input transform.
    class FakeModel:
        metadata = {'spatial_camera_roles': ['front']}
        def __init__(self):
            self.received = None
            self.preprocess = XHandSpatialOnlinePreprocess(repo_root=ROOT, camera_roles=('front',),
                use_visual=True, use_tactile=True, require_object_geometry=True, require_hand_keypoints=True)
        def infer(self, obs):
            out = self.preprocess(obs)
            repacked = XHandSpatialOnlineRepack()(out)
            xhand = XHandInputs(model_type=ModelType.PI0)(repacked)
            self.received = (out, xhand)
            return {'actions': np.full((16, 18), .125, np.float32)}
    model = FakeModel()
    bridge = HandScenePerceptionBridge(model, repo_root=root, perception_python=Path(sys.executable),
        workdir=tmp_path / 'spool', geometry_config=approved / 'manifest.json', calibration_profile=Path('fixture'))
    def launch():
        out = bridge.session_root / 'perception'
        (out / 'frames').mkdir()
        worker._run_task_geometry(args, spec, root, bridge.session_root / 'raw', out)
        bridge._started = True
    monkeypatch.setattr(bridge, '_launch', launch)
    server = WebsocketPolicyServer(bridge, metadata=bridge.metadata)
    ready = threading.Event(); stop = threading.Event(); info = {}
    def serve():
        async def run():
            async with websockets.asyncio.server.serve(server._handler, '127.0.0.1', 0, compression=None, max_size=None) as srv:
                info['port'] = srv.sockets[0].getsockname()[1]; ready.set()
                while not stop.is_set(): await asyncio.sleep(.01)
        asyncio.run(run())
    thread = threading.Thread(target=serve, daemon=True); thread.start()
    assert ready.wait(5)
    try:
        client = WebsocketClientPolicy('127.0.0.1', info['port'], proxy=None, open_timeout=2)
        raw = frame(0); raw['front_depth_native'][:] = 1000
        obs = observation('fixture', 1, [raw]); obs['prompt'] = 'test task'
        action = client.infer(obs)['actions']
        assert action.shape == (16, 18) and np.all(action == .125)
        out, xhand = model.received
        spatial = out['spatial']
        assert spatial['object_geometry']['manipulated_mask'].all()
        assert spatial['object_geometry']['target_mask'].all()
        assert spatial['hand_keypoints']['xyz_m'].shape == (18, 3)
        assert spatial['tactile']['xyz_m'].shape == (600, 3)
        assert spatial['visual']['xyz_m'].shape[1] == 3
        assert xhand['state'].shape == (18,)
        client._ws.close()
    finally:
        stop.set(); thread.join(timeout=5); bridge.close()


def test_native_server_loads_selected_config_and_guards_training_provenance(tmp_path, monkeypatch):
    from scripts import serve_policy
    from openpi.training import data_loader
    from scripts.preprocess.export_object_geometry_rgbtrack import SURFACE_SAMPLING_SEED
    task = 'future_task'
    root = tmp_path
    geo = root / 'output/preprocess' / task / 'geometry/approved/manifest.json'
    geo.parent.mkdir(parents=True)
    release = geo.parent.parent.parent / 'rgbtrack/release_v1.json'
    release.parent.mkdir()
    mesh = geo.parent / 'part.glb'; mesh.write_bytes(b'mesh')
    mesh2 = geo.parent / 'fixture.glb'; mesh2.write_bytes(b'fixture')
    entries = {p.stem: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'scale_m_per_unit': 1.}
               for p in (mesh, mesh2)}
    geo.write_text(json.dumps({'status': 'human_accepted', 'dataset_id': task,
                               'objects': list(entries), 'assets': entries}))
    release.write_text(json.dumps({'dataset': task, 'objects': list(entries), 'policy': {
        'method': 'static_upright_convex_mask_v1', 'mode': 'depth', 'dynamic_object': 'part',
        'static_object': 'fixture', 'init_tilt_deg': 25, 'init_rank': 150, 'min_mask_iou': .35}}))
    profile = root / 'profile.json'; profile.write_text('{}')
    sidecar = root / 'data' / task / 'object_geometry/v1/manifest.json'; sidecar.parent.mkdir(parents=True)
    manifest = {'status': 'complete', 'coordinate_frame': 'base_link',
        'object_geometry_contract': {'point_count': 512}, 'source_release_sha256': hashlib.sha256(release.read_bytes()).hexdigest(),
        'approved_geometry_sha256': hashlib.sha256(geo.read_bytes()).hexdigest(),
        'role_to_object': {'manipulated': 'part', 'target': 'fixture'},
        'surface_sampling_seed': SURFACE_SAMPLING_SEED,
        'approved_meshes_sha256': {name: item['sha256'] for name, item in entries.items()}}
    sidecar.write_text(json.dumps(manifest))
    config = types.SimpleNamespace(model=types.SimpleNamespace(encoder=types.SimpleNamespace(object_relation_mode='final_v2')),
        data=types.SimpleNamespace(object_geometry=types.SimpleNamespace(dataset_root=f'data/{task}', version='v1'),
                                   spatial=types.SimpleNamespace(front_calibration_profile=str(profile))))
    loaded = []
    monkeypatch.setattr(serve_policy._config, 'get_config', lambda name: config)
    monkeypatch.setattr(serve_policy._policy_config, 'create_trained_policy',
                        lambda cfg, directory, default_prompt=None: loaded.append((cfg, directory)) or types.SimpleNamespace(metadata={}))
    monkeypatch.setattr(data_loader, 'validate_object_geometry_calibration', lambda *a: True)
    monkeypatch.setenv('OPENPI_REPO_ROOT', str(root))
    monkeypatch.setenv('OPENPI_PERCEPTION_PYTHON', sys.executable)
    args = serve_policy.Args(policy=serve_policy.Checkpoint(config='training-name', dir='checkpoint-dir'))
    bridge = serve_policy.create_policy(args)
    assert loaded == [(config, 'checkpoint-dir')]
    assert bridge.generic_geometry and bridge.geometry_config == geo
    manifest['role_to_object'] = {'manipulated': 'fixture', 'target': 'part'}
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='disagree with training'):
        serve_policy.create_policy(args)

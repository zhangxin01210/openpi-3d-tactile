"""Small offline checks before expensive real Episode 23 checkpoint replay."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deploy.run_policy_dry_run import _gt_comparison
from scripts.deploy.export_training_episode_packets import export_episode


def test_first_action_and_50_step_chunk_compare_at_true_frame():
    gt={i:np.full(18,float(i)/100.,dtype=np.float32) for i in range(100)}
    prediction=np.stack([gt[16+i] for i in range(50)])
    result=_gt_comparison(prediction,16,gt)
    assert result["chunk_compared_steps"]==50
    assert result["first_arm6_mae_rad"]==0
    assert result["first_hand12_mae_rad"]==0
    assert result["chunk_mae_rad"]==0
    off=prediction.copy()
    off[:,6:]+=0.1
    result=_gt_comparison(off,16,gt)
    assert result["chunk_arm6_mae_rad"]==pytest.approx(0)
    assert result["chunk_hand12_mae_rad"]==pytest.approx(0.1,rel=1e-6)
    assert result["first_hand12_mae_rad"]==pytest.approx(0.1,rel=1e-6)


class FakeDataset:
    camera_roles=("front","left","right")
    depth_roles=("front",)

    def __init__(self,root,*,episode):
        assert episode==23

    def iter_rows(self,*,selected,depth_roles):
        assert tuple(depth_roles)==("front",)
        for i in sorted(selected):
            yield {"frame_index":i,"timestamp":i/15.,
                   "observation.state":np.zeros(1972,dtype=np.float32),
                   "observation.depths.cam_front":np.ones((4,5),dtype=np.uint16)*1000}

    def iter_video_frames(self,role,*,selected):
        for i in sorted(selected):
            yield SimpleNamespace(frame_index=i,rgb=np.zeros((4,5,3),dtype=np.uint8))


def test_uses_training_dataset_and_keeps_native_depth(tmp_path):
    camera=SimpleNamespace(depth_scale_m_per_unit=0.001)
    def register(depth,cam,hw):
        assert cam is camera
        assert hw==(4,5)
        assert depth.dtype==np.uint16
        return depth.astype(np.float32)*0.001

    output=tmp_path/"raw"
    count=export_episode(dataset_root=tmp_path,episode=23,max_frames=3,
                         output_dir=output,camera=camera,register_depth=register,
                         reader_class=FakeDataset)
    assert count==3
    with np.load(output/"frame_000002.npz",allow_pickle=False) as src:
        assert src["front_depth_native"].dtype==np.uint16
        np.testing.assert_array_equal(src["front_depth_native"],1000)
        np.testing.assert_allclose(src["front_depth_color_aligned"],1000)
        assert int(src["frame_index"])==2
        assert src["robot_state"].shape==(1972,)
    with pytest.raises(FileExistsError):
        export_episode(dataset_root=tmp_path,episode=23,max_frames=3,
                       output_dir=output,camera=camera,register_depth=register,
                       reader_class=FakeDataset)

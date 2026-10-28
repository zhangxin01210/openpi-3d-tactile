"""No real data/GPU: preview 1..6 uses reference image ONLY for operator display."""
import json
from types import SimpleNamespace

import cv2
import numpy as np

from scripts.deploy.render_scene_position_previews import generate,POSITIONS
from scripts.deploy.position_initialization import SCENE_POSITION_TO_EPISODE


class FakeDataset:
    def __init__(self, root, *, episode):
        self.episode=episode

    def iter_video_frames(self,role,*,selected):
        assert role=="front"
        assert selected=={0}
        rgb=np.full((480,640,3),self.episode,dtype=np.uint8)
        yield SimpleNamespace(frame_index=0,rgb=rgb)


def test_six_reference_images_and_one_montage_are_written(tmp_path):
    out=tmp_path/"previews"
    result=generate(dataset=tmp_path,output=out,reader_factory=FakeDataset)
    assert POSITIONS==(1,2,3,4,5,6)
    assert len(result["positions"])==6
    for record in result["positions"]:
        scene=record["position"]
        assert record["reference_training_episode"]==SCENE_POSITION_TO_EPISODE[scene]
        assert (out/f"scene_position_{scene}_reference_points.png").is_file()
        im=cv2.imread(str(out/record["preview"]))
        assert im.shape==(560,640,3)
        assert record["used_by_live_inference"].startswith("2D points ONLY")
    assert (out/"scene_positions_overview.png").is_file()
    montage=cv2.imread(str(out/"scene_positions_overview.png"))
    assert montage.shape==(1120,1920,3)
    assert result["positions"][0]["funnel_reference_episode"]==23
    assert result["positions"][0]["flask_reference_episode"]==0
    assert result["positions"][1]["funnel_reference_episode"]==23
    manifest=json.loads((out/"scene_positions_manifest.json").read_text())
    assert manifest["live_sam2_runs_on_current_live_camera_frame_zero"] is True

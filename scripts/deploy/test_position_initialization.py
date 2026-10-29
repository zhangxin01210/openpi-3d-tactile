"""Position templates are reviewed points, not fixed masks or auto-classification."""
import json

import pytest

from scripts.deploy.position_initialization import resolve_prompts


@pytest.mark.parametrize("episode", [0, 23, 38, 49, 58, 65])
def test_complete_six_position_reference_templates(repo_root, episode):
    result = resolve_prompts(
        repo_root=repo_root,
        prompt_file=None,
        template_episode=episode,
        template_confirmed=True,
        first_frame_rgb_shape=(480, 640, 3),
    )
    assert result["seed_frame"] == 0
    assert result["init_frames"] == {"funnel": 0, "target_flask": 0}
    assert result["initialization_provenance"]["template_episode"] == episode
    for name in ("funnel", "target_flask"):
        assert result["objects"][name]["positive"]


@pytest.fixture
def repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def test_template_requires_explicit_current_scene_review(repo_root):
    with pytest.raises(ValueError, match="template-reviewed-for-current-scene"):
        resolve_prompts(repo_root=repo_root, prompt_file=None, template_episode=23,
                        template_confirmed=False, first_frame_rgb_shape=(480, 640, 3))


def test_scene_1_uses_episode_zero_flask_and_episode23_funnel_clicks_only(repo_root):
    from scripts.deploy.position_initialization import SCENE_POSITION_TO_EPISODE
    assert SCENE_POSITION_TO_EPISODE == {1:0, 2:23, 3:38, 4:49, 5:58, 6:65}
    result=resolve_prompts(repo_root=repo_root,prompt_file=None,template_episode=0,
                           template_confirmed=True,first_frame_rgb_shape=(480,640,3))
    import json
    root=repo_root/"tools/hoi_level1_snapshot/metadata/six_positions"
    e0=json.loads((root/"episode_e00_flask_prompts.json").read_text())
    e23=json.loads((root/"episode_e23_seed_prompts.json").read_text())
    assert result["objects"]["target_flask"]==e0["objects"]["target_flask"]
    assert result["objects"]["funnel"]==e23["objects"]["funnel"]
    provenance=result["initialization_provenance"]
    assert provenance["scene_position"]==1
    assert provenance["borrowed_funnel_keypoints_from_episode"]==23
    assert provenance["reference_images_or_poses_reused"] is False


def test_wrong_image_resolution_rejected(repo_root):
    with pytest.raises(ValueError, match="image size"):
        resolve_prompts(repo_root=repo_root, prompt_file=None, template_episode=38,
                        template_confirmed=True, first_frame_rgb_shape=(720, 1280, 3))


def test_manual_prompt_file_can_supply_e0_after_review(repo_root, tmp_path):
    payload = {
        "frame_index": 0, "image_width": 12, "image_height": 10,
        "objects": {
            name: {"positive": [[3, 3]], "negative": [[5, 5]]}
            for name in ("funnel", "target_flask")
        },
    }
    f = tmp_path / "reviewed_e0.json"
    f.write_text(json.dumps(payload))
    result = resolve_prompts(repo_root=repo_root, prompt_file=f,
                             template_episode=None, template_confirmed=False,
                             first_frame_rgb_shape=(10, 12, 3))
    assert result["initialization_provenance"]["mode"] == "reviewed_prompts_file"

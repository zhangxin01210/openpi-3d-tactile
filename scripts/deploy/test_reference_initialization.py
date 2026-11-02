import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.deploy.position_initialization import initialize_masks
from scripts.preprocess import sam2_task

ROOT = Path(__file__).resolve().parents[2]

def image(root):
    return cv2.imread(str(root / 'references/e00000/reference_frame.jpg'))


def test_peg_selection_calls_existing_and_rejects_low_confidence(monkeypatch):
    root = ROOT / 'assets/perception/sam2_references/1018_peg_all/v1'
    frame = image(root)
    calls = []
    from scripts.preprocess import qa_peg_six_reference_selection as peg
    original = peg.pair_score
    def score(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(peg, 'pair_score', score)
    monkeypatch.setattr(sam2_task, 'sam_predict', lambda *a, **k: (np.ones(frame.shape[:2], np.uint8), .9))
    rejected = initialize_masks(root, frame, None)
    assert not rejected.success and not rejected.masks and len(calls) == 6
    accepted = initialize_masks(root, frame, None, max_reference_score=.1, min_reference_margin=0)
    assert accepted.success and accepted.provenance['reference_episode'] == 0
    assert len(accepted.provenance['candidates']) == 6
    assert initialize_masks(root, frame, None, manual_reference=23).success


@pytest.mark.parametrize('task', ['1028_press_button', '1030_yellow_socket'])
def test_single_reference_uses_preprocessing_and_failure_is_closed(task, monkeypatch):
    root = ROOT / 'assets/perception/sam2_references' / task / 'v1'
    frame = image(root)
    calls = []
    def predict(*args):
        calls.append(1)
        return np.ones(frame.shape[:2], np.uint8), .9, {'second_vs_best_ratio': 2.}
    monkeypatch.setattr(sam2_task, 'predict_object', predict)
    result = initialize_masks(root, frame, None)
    assert result.success and set(result.masks) == set(json.loads((root / 'manifest.json').read_text())['objects']) and len(calls) == 2
    monkeypatch.setattr(sam2_task, 'predict_object', lambda *a: (_ for _ in ()).throw(RuntimeError('SAM failed')))
    result = initialize_masks(root, frame, None)
    assert not result.success and result.masks == {} and 'SAM failed' in result.error

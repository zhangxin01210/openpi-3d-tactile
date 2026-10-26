"""Regression tests for the real Pi0 Gemma parameter-tree audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SCRIPT = Path(__file__).with_name("audit_finetune_strategy.py")
_SPEC = importlib.util.spec_from_file_location("audit_finetune_strategy_test_module", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
audit = _MODULE.audit


def test_dtcr_parameter_groups_use_instantiated_pi0_tree() -> None:
    # ``nnx.eval_shape`` still constructs the actual configured Pi0 module
    # graph (including the two Gemma experts and spatial router), but avoids
    # allocating the multi-billion-parameter arrays on the CPU test host.
    report = audit("pi0_xhand_spatial_dtcr_v1", abstract=True)
    groups = report["groups"]

    assert report["abstract_model"] is True
    assert groups["PaliGemma Gemma Base"]["parameters"] > 0
    assert groups["PaliGemma Gemma Base"]["frozen_parameters"] > 0
    assert groups["PaliGemma Gemma LoRA"]["parameters"] > 0
    assert groups["PaliGemma Gemma LoRA"]["trainable_parameters"] == groups["PaliGemma Gemma LoRA"]["parameters"]
    assert groups["Action Expert Gemma"]["parameters"] > 0
    assert groups["Action Expert Gemma"]["frozen_parameters"] == 0
    assert groups["Action Expert Gemma"]["trainable_parameters"] == groups["Action Expert Gemma"]["parameters"]
    assert groups["Action Expert Projections"]["frozen_parameters"] == 0
    assert groups["SigLIP"]["trainable_parameters"] == groups["SigLIP"]["parameters"]
    assert groups["Spatial Encoder"]["trainable_parameters"] > 0
    assert groups["Object Relation Encoder"]["trainable_parameters"] > 0
    assert groups["Spatial Suffix Adapter"]["trainable_parameters"] > 0

    assert report["total"]["parameters"] == report["trainable"]["parameters"] + report["frozen"]["parameters"]
    for group in groups.values():
        assert group["parameters"] == group["trainable_parameters"] + group["frozen_parameters"]

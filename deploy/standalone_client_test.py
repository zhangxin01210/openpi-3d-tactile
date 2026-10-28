"""The robot machine receives ONLY the deployment .py file, no openpi checkout."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CLIENT = Path(__file__).with_name("ur7e_xhand_deploy_3d_tactile_multi.py")


def test_single_file_outside_repo_has_no_local_contact_safety_dependency(tmp_path):
    copied = tmp_path / CLIENT.name
    copied.write_bytes(CLIENT.read_bytes())
    assert not (tmp_path / "contact_safety.py").exists()

    help_result = subprocess.run(
        [sys.executable, "-I", str(copied), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=15, check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--capture-observation-audit" in help_result.stdout
    assert "--contact-safety-config" in help_result.stdout
    assert "ModuleNotFoundError" not in help_result.stderr


def test_standalone_safety_is_embedded_even_with_isolated_python(tmp_path):
    copied = tmp_path / CLIENT.name
    copied.write_bytes(CLIENT.read_bytes())
    code = (
        "import runpy,sys; "
        "module=runpy.run_path(sys.argv[1],run_name='isolated_import'); "
        "assert 'ContactSafetyGuard' in module; "
        "assert 'SafetyTrip' in module; "
        "assert 'capture_observation_audit' in module"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(copied)],
        cwd=tmp_path, capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr

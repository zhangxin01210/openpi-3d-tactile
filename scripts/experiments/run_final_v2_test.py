import os
from pathlib import Path
import subprocess


def test_skip_preflight_keeps_original_training_entrypoint(tmp_path):
    source = Path(__file__).with_name("run_final_v2.sh")
    script = tmp_path / "scripts/experiments/run_final_v2.sh"
    script.parent.mkdir(parents=True)
    script.write_text(source.read_text())
    binary = tmp_path / "bin"
    binary.mkdir()
    git = binary / "git"
    git.write_text("#!/bin/sh\nprintf 'test-commit\\n'\n")
    git.chmod(0o755)
    python = binary / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_COMMAND_LOG"\n')
    python.chmod(0o755)
    log = tmp_path / "commands.log"
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", OPENPI_PYTHON=str(python), TEST_COMMAND_LOG=str(log))
    name = "pi0_xhand_1028_press_button_full_v2"
    subprocess.run(["bash", str(script), name, "--skip-preflight", "--resume"], env=env, check=True, capture_output=True)
    commands = log.read_text()
    assert "scripts/experiments/preflight_final_v2.py" not in commands
    assert f"scripts/train.py {name}" in commands
    assert "--resume" in commands

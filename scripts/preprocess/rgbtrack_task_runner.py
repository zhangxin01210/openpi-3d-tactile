#!/usr/bin/env python3
"""Formal RGBTrack execution adapter: reuse existing third-party and the verified static initializer."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--static-object')
    p.add_argument('--dynamic-object')
    p.add_argument('--dynamic-init-up-axis', choices=['+x','-x','+y','-y','+z','-z'])
    p.add_argument('--dynamic-init-max-tilt-deg', type=float, default=25.)
    p.add_argument('--dynamic-init-min-iou', type=float, default=.20)
    p.add_argument('--guard-max-step-m', type=float)
    p.add_argument('--guard-max-turn-deg', type=float)
    p.add_argument('--guard-recovery-max-turn-deg', type=float, default=60.0)
    p.add_argument('--init-tilt-deg', type=float, default=25.)
    p.add_argument('--init-rank', type=int, default=150)
    p.add_argument('--min-mask-iou', type=float, default=.35)
    selected, rest = p.parse_known_args()
    repo_arg = argparse.ArgumentParser(add_help=False)
    repo_arg.add_argument('--repo-root', type=Path, required=True)
    args, _ = repo_arg.parse_known_args(rest)
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo / 'scripts/object_geometry/pose_tracking'))
    sys.path.insert(0, str(repo / 'scripts/preprocess'))
    import rgbtrack_adapter as adapter
    from rgbtrack_static_policy import install_static_policy
    cmd = adapter.build_arg_parser().parse_args(rest)
    output = cmd.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists, refusing overwrite: {output}')
    if selected.static_object or selected.dynamic_init_up_axis:
        if selected.dynamic_init_up_axis and (not selected.dynamic_object or selected.dynamic_object not in cmd.objects):
            raise ValueError('Dynamic upright initialization requires a valid --dynamic-object')
        if selected.static_object and selected.static_object not in cmd.objects:
            raise ValueError(f'--static-object not among {cmd.objects}')
        from rgbtrack_adapter import load_front_calibration
        cam = load_front_calibration(repo, cmd.front_calibration_profile)
        # The selector patches only the selected object's estimator.
        # The other object's RGBTrack path is untouched.
        output.mkdir(parents=True, exist_ok=False)
        if selected.static_object:
            install_static_policy(adapter, selected.static_object,
                                  cam.T_base_color[:3, :3], output,
                                  selected.init_tilt_deg, selected.init_rank,
                                  selected.min_mask_iou)
        if selected.dynamic_init_up_axis:
            if selected.dynamic_object == selected.static_object:
                raise ValueError('Dynamic upright object must differ from static object')
            from rgbtrack_dynamic_init import install_dynamic_first_upright
            install_dynamic_first_upright(adapter,name=selected.dynamic_object,
                    base_R_camera=cam.T_base_color[:3,:3],output=output,
                    upright_axis=selected.dynamic_init_up_axis,
                    max_tilt_deg=selected.dynamic_init_max_tilt_deg,
                    max_rank=selected.init_rank,
                    min_iou=selected.dynamic_init_min_iou)
    writer = None
    if selected.guard_max_step_m is not None or selected.guard_max_turn_deg is not None:
        if selected.guard_max_step_m is None or selected.guard_max_turn_deg is None:
            raise ValueError('Supply both motion guard thresholds')
        if not selected.dynamic_object or selected.dynamic_object not in cmd.objects:
            raise ValueError('Guard requires --dynamic-object named in --objects')
        if selected.dynamic_object == selected.static_object:
            raise ValueError('Motion guard cannot target frozen static object')
        from rgbtrack_motion_guard import install_motion_guard
        writer = install_motion_guard(
            adapter, name=selected.dynamic_object, output=output,
            max_step_m=selected.guard_max_step_m,
            max_turn_deg=selected.guard_max_turn_deg,
            max_recovery_turn_deg=selected.guard_recovery_max_turn_deg)
    try:
        return adapter.run(cmd)
    finally:
        if writer:
            writer()

if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the ORIGINAL WebSocket server + trained Pi0 + EXACT perception producer.

This launcher is specialized only because the perception worker needs its
existing separate PyTorch/RGBTrack environment. Messages, action keys and
client remain the stock openpi WebsocketPolicyServer contract, extended with
causal hand_scene_history raw observations in the request.

CORRECTNESS-ONLY stage. Current blocking per-frame processing is too slow
for robot servoing: the standalone client forcibly requires --dry-run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.config import get_config
from scripts.deploy.websocket_hand_scene_bridge import HandScenePerceptionBridge

ROOT=Path(__file__).resolve().parents[1]
TASK="Pick up the glass funnel, align its stem with the opening of the small beaker, and insert it until it is fully seated."


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=ROOT)
    p.add_argument("--config",default="pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--port",type=int,default=8990)
    p.add_argument("--perception-python",type=Path,required=True,
                   help="Original RGBTrack/SAM2 virtualenv Python")
    p.add_argument("--workdir",type=Path,required=True,
                   help="New, empty server-local temporary perception spool")
    p.add_argument("--static-target-anchor-json",type=Path,required=True,
                   help="Human-reviewed anchor for CURRENT physical scene")
    group=p.add_mutually_exclusive_group()
    group.add_argument("--position-template-episode",type=int,default=23)
    group.add_argument("--prompts",type=Path)
    p.add_argument("--episode",type=int,default=23)
    p.add_argument("--timeout-s",type=float,default=180.)
    p.add_argument("--sam2-checkpoint",type=Path)
    p.add_argument("--sam2-source",type=Path)
    p.add_argument("--rgbtrack-weights",type=Path)
    p.add_argument("--scale-manifest",type=Path)
    args=p.parse_args()

    root=args.repo_root.resolve()
    model_path=args.checkpoint.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Trained Pi0 checkpoint missing: {model_path}")
    config=get_config(args.config)
    if config.data is None:
        raise ValueError("Training data config missing")
    trained=create_trained_policy(config,model_path,default_prompt=TASK)
    if getattr(config.data,"object_geometry",None) is None:
        # This field is in the CREATED data config, not factory.
        data_config=config.data.create(config.assets_dirs,config.model)
        if getattr(data_config,"object_geometry",None) is None:
            raise ValueError("This server requires an object-geometry-trained Pi0 config")

    worker=HandScenePerceptionBridge(
        trained,
        repo_root=root,
        perception_python=args.perception_python,
        workdir=args.workdir,
        anchor_json=args.static_target_anchor_json,
        episode=args.episode,
        position_template_episode=args.position_template_episode,
        prompt_file=args.prompts,
        timeout_s=args.timeout_s,
        sam2_checkpoint=args.sam2_checkpoint,
        rgbtrack_weights=args.rgbtrack_weights,
        sam2_source=args.sam2_source,
        scale_manifest=args.scale_manifest,
    )
    meta={
        **worker.metadata,
        "spatial_camera_roles":["front"],
        "spatial_use_visual":True,
        "spatial_use_tactile":True,
        "hand_scene_history_required":True,
    }
    print(f"[WEBSOCKET HAND-SCENE] port={args.port} checkpoint={model_path}",flush=True)
    print("[WEBSOCKET HAND-SCENE] BLOCKING DRY-RUN ONLY, no robot action transport in server",flush=True)
    server=WebsocketPolicyServer(policy=worker,host="0.0.0.0",port=args.port,metadata=meta, infer_threaded=True)
    try:
        server.serve_forever()
    finally:
        worker.close()


if __name__=="__main__":
    main()

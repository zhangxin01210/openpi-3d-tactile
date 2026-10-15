#!/usr/bin/env python3
# ruff: noqa: E402
"""Small OpenPI-wire-compatible reference server for the live loop gate.

Returns the saved joint target indexed by the client's source_index. This is
deliberately a reference controller, not a learned or observation-driven policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from websockets.sync.server import serve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "openpi-client" / "src"))
from openpi_client import msgpack_numpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with h5py.File(args.hdf5, "r") as h5:
        joints = np.asarray(h5["embodiment/joint"][:, :9], dtype=np.float32)

    def handler(connection) -> None:
        connection.send(msgpack_numpy.packb({"provider": "recorded_reference", "action_space": "9D_absolute_joint"}))
        while True:
            try:
                raw = connection.recv()
            except Exception:
                return
            request = msgpack_numpy.unpackb(raw)
            state = np.asarray(request["observation/state"])
            images = [request[key] for key in (
                "observation/head_rgb", "observation/wrist_rgb",
                "observation/left_tactile_rgb", "observation/right_tactile_rgb",
            )]
            if "spatial" in request:
                visual = request["spatial"]["visual"]
                if np.asarray(visual["xyz_m"]).shape != (2048, 3):
                    connection.send("Invalid live world pointcloud")
                    return
            index = int(request["source_index"])
            if state.shape != (9,) or not all(np.asarray(image).ndim == 3 for image in images):
                connection.send("Invalid live observation")
                return
            if not 0 <= index < len(joints):
                connection.send("source_index out of bounds")
                return
            connection.send(msgpack_numpy.packb({"actions": joints[index:index + 1]}))

    with serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        print(f"recorded reference server listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

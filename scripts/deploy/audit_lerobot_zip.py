"""Read-only CRC, Parquet and full video-decode validation of a LeRobot ZIP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
import zipfile

import av
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def audit_video(archive, name, expected, fps):
    result = {"file": name, "status": "PASS"}
    try:
        contents = archive.read(name)  # A complete read verifies the member CRC.
        count, previous, max_timestamp_error = 0, None, 0.0
        with av.open(io.BytesIO(contents)) as container:
            assert len(container.streams.video) == 1, "Expected one video stream"
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            assert (stream.width, stream.height) == (640, 480), "Unexpected resolution"
            assert abs(float(stream.average_rate) - fps) < 1e-6, "Unexpected video FPS"
            for frame in container.decode(stream):
                assert not frame.is_corrupt, f"Corrupt decoded frame {count}"
                assert (frame.width, frame.height) == (640, 480)
                assert frame.pts is not None, f"No timestamp at frame {count}"
                timestamp = float(frame.pts * frame.time_base)
                assert previous is None or timestamp > previous, "Nonmonotonic video timestamps"
                max_timestamp_error = max(max_timestamp_error, abs(timestamp - count / fps))
                previous = timestamp
                count += 1
            assert count == expected, f"Decoded {count}, expected {expected} frames"
            assert max_timestamp_error < 0.001, f"Video time mismatch: {max_timestamp_error}"
            result.update(
                frames=count,
                codec=stream.codec_context.name,
                width=stream.width,
                height=stream.height,
                max_timestamp_error_s=max_timestamp_error,
            )
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    return result


def audit_parquet(archive, name, episode, expected, start_index, fps, features):
    result = {"file": name, "episode": episode, "status": "PASS", "depth": {}}
    summary = {}
    try:
        contents = archive.read(name)
        parquet = pq.ParquetFile(io.BytesIO(contents))
        assert parquet.metadata.num_rows == expected, "Parquet row count != episode metadata"
        required = [key for key, feature in features.items() if feature["dtype"] != "video"]
        assert set(required) == set(parquet.schema_arrow.names), "Parquet columns != declared features"
        offset = 0
        for batch in parquet.iter_batches(batch_size=16, use_threads=False):
            for key in required:
                array = batch.column(batch.schema.get_field_index(key))
                assert array.null_count == 0, f"Null values in {key}"
                if key.startswith("observation.depths."):
                    assert np.all(pc.list_value_length(array).to_numpy() == 480), f"Depth height mismatch: {key}"
                    inner = array.flatten()
                    assert inner.null_count == 0
                    assert np.all(pc.list_value_length(inner).to_numpy() == 640), f"Depth width mismatch: {key}"
                    flat = inner.flatten()
                    assert flat.null_count == 0
                    values = flat.to_numpy().reshape(len(array), 480, 640)
                    assert values.dtype == np.uint16, f"Depth dtype != uint16: {key}"
                    nonzero = np.count_nonzero(values, axis=(1, 2))
                    depth = result["depth"].setdefault(
                        key, {"zero_frames": 0, "nonzero_pixels": 0, "pixels": 0, "max_raw": 0}
                    )
                    depth["zero_frames"] += int(np.count_nonzero(nonzero == 0))
                    depth["nonzero_pixels"] += int(nonzero.sum())
                    depth["pixels"] += values.size
                    depth["max_raw"] = max(depth["max_raw"], int(values.max()))
                else:
                    shape = features[key]["shape"]
                    if len(shape) == 1 and shape[0] > 1:
                        assert np.all(pc.list_value_length(array).to_numpy() == shape[0]), (
                            f"Vector dimension mismatch: {key}"
                        )
                        flat = array.flatten()
                        assert flat.null_count == 0
                        values = flat.to_numpy().reshape(len(array), shape[0])
                    else:
                        values = array.to_numpy()
                    assert np.isfinite(values).all(), f"NaN/Inf in {key}"
                    indices = np.arange(offset, offset + len(array))
                    if key == "frame_index":
                        np.testing.assert_array_equal(values, indices)
                    elif key == "index":
                        np.testing.assert_array_equal(values, indices + start_index)
                    elif key == "episode_index":
                        assert np.all(values == episode), "Episode indices incorrect"
                    elif key == "task_index":
                        assert np.all(values == 0), "Task indices incorrect"
                    elif key == "timestamp":
                        assert np.max(np.abs(values - indices / fps)) < 0.0002, "Timestamps != frame / fps"
                    if key in ("action", "observation.state", "action_target_position"):
                        s = summary.setdefault(
                            key,
                            {
                                "min": np.full(values.shape[1], np.inf),
                                "max": np.full(values.shape[1], -np.inf),
                                "sum": np.zeros(values.shape[1]),
                                "sum_sq": np.zeros(values.shape[1]),
                                "count": 0,
                            },
                        )
                        s["min"] = np.minimum(s["min"], values.min(axis=0))
                        s["max"] = np.maximum(s["max"], values.max(axis=0))
                        s["sum"] += values.sum(axis=0, dtype=np.float64)
                        s["sum_sq"] += np.square(values.astype(np.float64)).sum(axis=0)
                        s["count"] += len(values)
            offset += len(batch)
        assert offset == expected
        result["frames"] = offset
        for depth in result["depth"].values():
            depth["nonzero_fraction"] = depth["nonzero_pixels"] / depth["pixels"]
        result["vectors"] = {}
        for key, s in summary.items():
            result["vectors"][key] = {
                "count": s["count"],
                "min": s["min"].tolist(),
                "max": s["max"].tolist(),
                "mean": (s["sum"] / s["count"]).tolist(),
                "std": np.sqrt(np.maximum(0, s["sum_sq"] / s["count"] - np.square(s["sum"] / s["count"]))).tolist(),
            }
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    return result


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.archive.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = path.stat()
    start = time.perf_counter()
    report = {
        "archive": str(path),
        "size_bytes": before.st_size,
        "errors": [],
        "warnings": [],
        "parquets": [],
        "videos": [],
    }
    with ThreadPoolExecutor(max_workers=4) as pool, zipfile.ZipFile(path) as archive:
        digest_future = pool.submit(sha256, path)
        members = archive.infolist()
        files = [entry.filename for entry in members if not entry.is_dir()]
        assert len({entry.filename for entry in members}) == len(members), "Duplicate ZIP member paths"
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in files), (
            "Unsafe ZIP member paths"
        )
        roots = {Path(name).parts[0] for name in files}
        assert len(roots) == 1
        root = next(iter(roots)) + "/"
        info = json.loads(archive.read(root + "meta/info.json"))
        episodes = [json.loads(line) for line in archive.read(root + "meta/episodes.jsonl").splitlines()]
        tasks = [json.loads(line) for line in archive.read(root + "meta/tasks.jsonl").splitlines()]
        report["metadata"] = {key: value for key, value in info.items() if key != "features"}
        report["tasks"] = tasks
        n = info["total_episodes"]
        assert [row["episode_index"] for row in episodes] == list(range(n)), "Missing, duplicate or unordered episodes"
        assert sum(row["length"] for row in episodes) == info["total_frames"], "Episode lengths != total_frames"
        assert len(tasks) == info["total_tasks"] == 1
        assert tasks[0]["task_index"] == 0
        assert info["splits"]["train"] == f"0:{n}"
        assert info["total_videos"] == n * 3
        expected_files = {
            root + "meta/" + name
            for name in ("info.json", "stats.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl")
        }
        video_futures = []
        start_index = 0
        for ep in episodes:
            idx, length = ep["episode_index"], ep["length"]
            assert length > 0
            assert ep["tasks"] == [tasks[0]["task"]]
            chunk = idx // info["chunks_size"]
            name = root + info["data_path"].format(episode_index=idx, episode_chunk=chunk)
            expected_files.add(name)
            result = audit_parquet(archive, name, idx, length, start_index, info["fps"], info["features"])
            report["parquets"].append(result)
            start_index += length
            for camera in ("cam_front", "cam_left", "cam_right"):
                video = root + info["video_path"].format(
                    episode_index=idx, episode_chunk=chunk, video_key=f"observation.images.{camera}"
                )
                expected_files.add(video)
                video_futures.append(pool.submit(audit_video, archive, video, length, info["fps"]))
            if idx % 5 == 0 or idx == n - 1:
                print(
                    f"Parquet {idx + 1}/{n}: {result['status']}; elapsed={time.perf_counter() - start:.1f}s", flush=True
                )
                (output / "progress.json").write_text(json.dumps({"parquets_checked": idx + 1, "total": n}))
        report["missing_files"] = sorted(expected_files - set(files))
        report["extra_files"] = sorted(set(files) - expected_files)
        global_stats = json.loads(archive.read(root + "meta/stats.json"))
        stats_episodes = []
        with archive.open(root + "meta/episodes_stats.jsonl") as stream:
            for line in stream:
                row = json.loads(line)
                idx = row["episode_index"]
                stats_episodes.append(idx)
                assert 0 <= idx < n
                for key, stats in row["stats"].items():
                    for field, value in stats.items():
                        a = np.asarray(value)
                        assert np.isfinite(a).all(), f"Nonfinite episode stats {idx}/{key}/{field}"
                    assert np.all(np.asarray(stats["std"]) >= 0), f"Negative std in episode {idx}/{key}"
                actual = report["parquets"][idx]
                if actual["status"] == "PASS":
                    for key, values in actual["vectors"].items():
                        recorded = row["stats"][key]
                        assert recorded["count"] == [episodes[idx]["length"]], f"Stats count mismatch in {idx}/{key}"
                        for field in ("min", "max", "mean", "std"):
                            if not np.allclose(recorded[field], values[field], atol=1e-4, rtol=1e-4):
                                report["warnings"].append(
                                    f"Episode {idx} {key} {field} statistics differ from recomputation"
                                )
                if idx % 10 == 0:
                    print(f"Episode statistics {idx + 1}/{n}", flush=True)
        assert stats_episodes == list(range(n)), "Episode statistics missing or duplicated"
        for key, stats in global_stats.items():
            for field, value in stats.items():
                assert np.isfinite(np.asarray(value)).all(), f"Nonfinite global stats {key}/{field}"
            assert np.all(np.asarray(stats["std"]) >= 0)
        report["metadata_json_valid"] = True
        report["stats_depth_layout"] = "Per-pixel 480x640 depth statistics explain the large JSON files"
        for i, future in enumerate(video_futures):
            report["videos"].append(future.result())
            if (i + 1) % 30 == 0:
                print(f"Video full decode {i + 1}/{len(video_futures)}", flush=True)
        # Read any unexpected members too, so the CRC claim covers every file.
        for name in report["extra_files"]:
            with archive.open(name) as stream:
                while stream.read(8 * 1024 * 1024):
                    pass
        report["sha256"] = digest_future.result()
        report["zip_file_members"] = len(files)
        report["uncompressed_bytes"] = sum(entry.file_size for entry in members)
    after = path.stat()
    report["archive_unchanged_during_check"] = (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    )
    report["errors"] += [r["error"] for r in report["parquets"] + report["videos"] if r["status"] != "PASS"]
    if report["missing_files"]:
        report["errors"].append("Missing required files")
    if not report["archive_unchanged_during_check"]:
        report["errors"].append("Archive changed while checking")
    for r in report["parquets"]:
        for key, depth in r["depth"].items():
            if depth["zero_frames"]:
                report["warnings"].append(f"Episode {r['episode']} {key}: {depth['zero_frames']} empty depth frames")
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report["wall_s"] = time.perf_counter() - start
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    checksum = f"{report['sha256']}  {path.name}\n"
    (output / (path.name + ".sha256")).write_text(checksum)
    print(
        json.dumps(
            {key: report[key] for key in ("status", "errors", "warnings", "sha256", "wall_s")}, ensure_ascii=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

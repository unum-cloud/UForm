# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "av",
#     "huggingface_hub",
#     "numpy",
#     "pillow",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Idempotent downloader for the DROID 1K-trajectory subset (~5 GB) — pulls AND repacks in one file.

Modality: video-text act (vision-language-action). The full DROID release is 1.7 TB across 76K demos;
this downloader pulls the published 1K-trajectory subset, sized so the video-text act smoke runner
finishes well under a minute on a single H100.

LeRobot DROID stores the numeric stream — `action` (7-DoF), `observation.state`, `task_index`,
`episode_index`, `frame_index` — in a `data/**/file-*.parquet`, the natural-language goals in
`meta/tasks.parquet`, and the camera observations as concatenated per-chunk MP4 videos. The
data-parquet rows are in global frame order, so frame `i` of the concatenated camera video
corresponds to data row `i` — a sequential decode, no per-episode timestamp seek.

This single file does both halves: it snapshots the 1K subset, then decodes the camera MP4 frame by
frame (CPU / libav, offline), JPEG-encodes each frame, and writes one row
`[image: binary(jpeg), instruction: string, action: list<float32>]` — the same inline-compressed-bytes
shape COCO uses, so the video-text act loader decodes the images on the GPU via the existing
`ImageDecoderPool` at train time rather than paying a bespoke video-seek path. The repack is a single
per-corpus video decode, so it runs inline (no separate script).

CLI usage:
    uv run scripts/data/download_droid.py
    uv run scripts/data/download_droid.py --out-dir /custom/path
    uv run scripts/data/download_droid.py --camera observation.images.wrist_image_left --max-frames 400
"""

import argparse
import glob
import io
import os
import sys
from pathlib import Path

import av
import pyarrow
import pyarrow.parquet
from download_smokes import smoke_cache_root
from huggingface_hub import snapshot_download
from PIL import Image

# LeRobot's 100-trajectory DROID subset; `droid-dataset/droid_100` was removed from HuggingFace and
# DROID's full release now lives on a GCS bucket, but this LeRobot mirror keeps the same
# data/meta/videos layout the repack below consumes.
_PRIMARY_REPO_ID = "lerobot/droid_100"
#: Parquet row-group size — bounds writer memory on long corpora.
ROW_GROUP_SIZE: int = 2_000
#: Default camera feature whose concatenated MP4 supplies the observation frames.
_DEFAULT_CAMERA = "observation.images.wrist_image_left"


def _snapshot_one(repo_id: str, out_dir: Path) -> bool:
    """Try a single HuggingFace snapshot pull.

    Args:
        repo_id: The HuggingFace dataset repository id.
        out_dir: Local destination directory for the snapshot.

    Returns:
        `True` on a successful pull, `False` if the pull raised.
    """
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"download_droid: {repo_id} failed ({exc})")
        return False
    return True


def _load_tasks(root: Path) -> dict[int, str]:
    """Map `task_index` to its natural-language instruction from `meta/tasks.parquet`.

    The tasks table keys the instruction string by `task_index`; LeRobot stores the instruction as
    the (string) row index, surfaced by pyarrow as `__index_level_0__`.

    Args:
        root: Dataset root holding `meta/tasks.parquet`.

    Returns:
        `{task_index: instruction}`.
    """
    table = pyarrow.parquet.read_table(root / "meta" / "tasks.parquet")
    columns = table.column_names
    instruction_column = "__index_level_0__" if "__index_level_0__" in columns else columns[-1]
    indices = table.column("task_index").to_pylist()
    instructions = table.column(instruction_column).to_pylist()
    return {int(i): str(s) for i, s in zip(indices, instructions, strict=True)}


def _data_parquet(root: Path) -> str:
    """Return the single data-stream parquet path under `root/data/**`."""
    matches = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not matches:
        raise FileNotFoundError(f"no data parquet under {root / 'data'}")
    return matches[0]


def _camera_video(root: Path, camera: str) -> str:
    """Return the single MP4 path for `camera` under `root/videos/<camera>/**`."""
    matches = sorted(glob.glob(str(root / "videos" / camera / "**" / "*.mp4"), recursive=True))
    if not matches:
        raise FileNotFoundError(f"no mp4 under {root / 'videos' / camera}")
    return matches[0]


def repack(root: Path, camera: str, max_frames: int | None, overwrite: bool) -> int:
    """Decode the camera video and emit a flat `(image, instruction, action)` parquet.

    Args:
        root: LeRobot dataset root (the `lerobot/droid_100`-style layout).
        camera: Camera feature name (e.g. `observation.images.wrist_image_left`).
        max_frames: Cap on emitted rows for a prototype shard, or `None` for the full corpus.
        overwrite: Re-repack even if the output parquet already exists.

    Returns:
        Process exit code.
    """
    out_path = root / f"droid_{camera.replace('.', '_')}.parquet"
    if out_path.exists() and not overwrite:
        print(f"repack_droid: {out_path.name} exists; pass --overwrite to rebuild")
        return 0

    tasks = _load_tasks(root)
    data = pyarrow.parquet.read_table(_data_parquet(root), columns=["action", "task_index"])
    actions = data.column("action").to_pylist()
    task_indices = data.column("task_index").to_pylist()
    row_count = data.num_rows

    schema = pyarrow.schema(
        [
            ("image", pyarrow.binary()),
            ("instruction", pyarrow.string()),
            ("action", pyarrow.list_(pyarrow.float32())),
        ]
    )
    tmp_path = out_path.with_suffix(".parquet.tmp")
    writer = pyarrow.parquet.ParquetWriter(str(tmp_path), schema, compression="zstd")
    images: list[bytes] = []
    instructions: list[str] = []
    action_rows: list[list[float]] = []
    written = 0

    def flush() -> None:
        if not images:
            return
        writer.write_table(
            pyarrow.table({"image": images, "instruction": instructions, "action": action_rows}, schema=schema)
        )
        images.clear()
        instructions.clear()
        action_rows.clear()

    container = av.open(_camera_video(root, camera))
    try:
        frame_index = 0
        for frame in container.decode(video=0):
            if frame_index >= row_count:
                break  # video carries trailing padding frames beyond the data stream
            if max_frames is not None and written >= max_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
            images.append(buffer.getvalue())
            instructions.append(tasks.get(int(task_indices[frame_index]), ""))
            action_rows.append([float(x) for x in actions[frame_index]])
            written += 1
            frame_index += 1
            if len(images) >= ROW_GROUP_SIZE:
                flush()
        flush()
    finally:
        container.close()
        writer.close()
    os.replace(tmp_path, out_path)
    action_dim = len(actions[0]) if actions else 0
    print(f"repack_droid: wrote {written} (image, instruction, action[{action_dim}]) rows -> {out_path}")
    return 0


def download(
    out_dir: Path,
    repo_id: str | None = None,
    force: bool = False,
    camera: str = _DEFAULT_CAMERA,
    max_frames: int | None = None,
) -> int:
    """Download the DROID 1K subset to out_dir and repack the camera video. Skip if populated unless force."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".downloaded"
    if sentinel.exists() and not force:
        print(f"download_droid: already at {out_dir}; skip (pass --force to redownload)")
        return 0
    chosen_repo = repo_id or _PRIMARY_REPO_ID
    if not _snapshot_one(chosen_repo, out_dir):
        print("download_droid: HuggingFace mirror failed", file=sys.stderr)
        return 1
    code = repack(out_dir, camera, max_frames, force)
    if code != 0:
        return code
    sentinel.touch()
    print(f"download_droid: DROID 1K ready at {out_dir} (repo={chosen_repo})")
    return 0


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotent downloader + repacker for the DROID 1K-trajectory subset (~5 GB).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=smoke_cache_root() / "droid-1k",
        help="Destination directory (default: $UFORM_SMOKE_CACHE/droid-1k or ~/.cache/uform/smokes/droid-1k).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"Override HuggingFace dataset id (default: {_PRIMARY_REPO_ID}).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=_DEFAULT_CAMERA,
        help=f"Camera feature name whose MP4 supplies the observation frames (default: {_DEFAULT_CAMERA}).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Cap emitted rows for a prototype shard.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-repack even if the sentinel file exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    sys.exit(download(args.out_dir, args.repo_id, args.force, args.camera, args.max_frames))

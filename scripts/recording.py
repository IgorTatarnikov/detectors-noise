"""Save and load camera recordings: one folder of TIFFs plus metadata.json.

    data/live/<YYYYMMDD-HHMMSS>/
      read_noise/<gain>dB/            frame_0000.tif ... metadata.json
      dark/<gain>dB/<exposure>us/
      shot/<label>/                   illuminated
      shot/<label>_dark/              paired dark stack at identical settings
      saturation/<label>/
      results.json

Values are the camera's raw counts (Mono12: 0-4095), stored as uint16 without any shift.
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

DEFAULT_ROOT = Path("data/live")
METADATA = "metadata.json"


def new_session(root: Path | str = DEFAULT_ROOT) -> Path:
    path = Path(root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gain_label(gain_db: float) -> str:
    return f"{gain_db:g}dB"


def exposure_label(exposure_us: float) -> str:
    return f"{exposure_us:g}us"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"can't save {type(value).__name__} as JSON")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


class RecordingWriter:
    """Writes frames to disk in a background thread, keeping a subsampled copy in memory.

    Writing a full-size TIFF takes a few ms, so doing it on the grab thread would make the
    camera run out of buffers and drop frames.
    """

    def __init__(self, path: Path, metadata: dict, analysis_step: int = 1):
        self.path = Path(path)
        if self.path.exists() and any(self.path.glob("*.tif")):
            raise FileExistsError(f"{self.path} already has a recording")
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.step = analysis_step
        self.frames: list[np.ndarray] = []
        self.full_means: list[float] = []
        self.block_ids: list[int] = []
        self.timestamps: list[int] = []
        self._queue: queue.Queue = queue.Queue()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            index, image = item
            try:
                tifffile.imwrite(self.path / f"frame_{index:04d}.tif", image)
            except Exception as e:  # surfaced in close()
                self._error = e

    def add(self, image: np.ndarray, block_id: int, timestamp: int = 0) -> None:
        index = len(self.block_ids)
        self.frames.append(np.ascontiguousarray(image[:: self.step, :: self.step]))
        self.full_means.append(float(image.mean(dtype=np.float64)))
        self.block_ids.append(int(block_id))
        self.timestamps.append(int(timestamp))
        self._queue.put((index, image))

    def stack(self) -> np.ndarray:
        return np.stack(self.frames)

    def close(self, **extra) -> dict:
        from noise_analysis import check_block_ids

        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error
        missing, repeated = check_block_ids(self.block_ids)
        self.metadata.update(extra)
        self.metadata.update(
            frames=len(self.block_ids),
            block_ids=self.block_ids,
            timestamps=self.timestamps,
            frame_means=self.full_means,
            missing_frames=missing,
            repeated_frames=repeated,
            saved=datetime.now().isoformat(timespec="seconds"),
        )
        write_json(self.path / METADATA, self.metadata)
        return self.metadata


def load_metadata(path: Path | str) -> dict:
    return json.loads((Path(path) / METADATA).read_text())


def load_recording(path: Path | str, step: int = 1) -> tuple[np.ndarray, dict]:
    """Frames as uint16 (frames, rows, cols), every `step`th pixel, and the metadata."""
    path = Path(path)
    files = sorted(path.glob("frame_*.tif"))
    if not files:
        raise FileNotFoundError(f"no frames in {path}")
    first = tifffile.imread(files[0])[::step, ::step]
    frames = np.empty((len(files), *first.shape), dtype=first.dtype)
    frames[0] = first
    for i, f in enumerate(files[1:], start=1):
        frames[i] = tifffile.imread(f)[::step, ::step]
    return frames, load_metadata(path)


SUPERSEDED = "superseded"


def supersede(path: Path, session: Path) -> Path | None:
    """Move an earlier take out of the way, so `path` always holds the latest recording.

    The notebook reads fixed paths such as read_noise/12dB/, so a new take must replace
    the old one there rather than land beside it. The old take is kept under
    <session>/superseded/<time>/ with the same relative path.
    """
    path, session = Path(path), Path(session)
    if not path.exists():
        return None
    target = session / SUPERSEDED / datetime.now().strftime("%Y%m%d-%H%M%S-%f") / path.relative_to(session)
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    return target


def find_recordings(session: Path | str) -> list[Path]:
    """Every current recording folder in a session (not superseded takes), in a stable order."""
    session = Path(session)
    return sorted(
        p.parent for p in session.rglob(METADATA) if SUPERSEDED not in p.parent.relative_to(session).parts
    )

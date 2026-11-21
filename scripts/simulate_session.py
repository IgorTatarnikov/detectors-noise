#!/usr/bin/env python
"""Write a complete simulated camera session, in the same layout basler_live.py records.

Use it to try the notebook or slides before any live data exists. Write it somewhere other
than the folder the live session will use, and point data_dir at it:

    .venv/bin/python scripts/simulate_session.py data/live/simulated
    .venv/bin/python scripts/simulate_session.py data/live/simulated --shape 300 480 --frames 50

Every recording's metadata says camera_model "Simulated ...", and the notebook prints it.

The simulated camera has known values (see basler_camera.SimulatedCamera), so the
results show what the analysis should recover.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from basler_camera import SimulatedCamera, camera_metadata
from recording import RecordingWriter, exposure_label

GAINS = (0.0, 1.0, 12.0, 24.0)
DARK_EXPOSURES_MS = (0.04, 1, 10, 100, 1000, 10000)
BLACK_LEVEL = 64


def record(cam: SimulatedCamera, path: Path, frames: int, metadata: dict) -> dict:
    writer = RecordingWriter(path, {**metadata, **camera_metadata(cam)})
    cam.grab(1)  # the GUI discards the first frame after a settings change
    for _ in range(frames):
        f = cam.grab(1)
        writer.add(f.image, f.block_id, f.timestamp)
    return writer.close(cancelled=False)


def simulate_session(
    session: Path,
    *,
    shape=(120, 160),
    frames: int = 30,
    dark_frames: int = 3,
    gains=GAINS,
    dark_exposures_ms=DARK_EXPOSURES_MS,
    shot_gain: float = 12.0,
    shot_exposure_ms: float = 20.0,
    seed: int = 0,
    **camera_options,
) -> SimulatedCamera:
    cam = SimulatedCamera(shape=shape, seed=seed, realtime=False, light=2e5, **camera_options)
    cam.start()
    cam.set("black_level", BLACK_LEVEL)
    cam.covered = True
    lo = cam.limits["exposure_us"][0]
    for g in gains:
        cam.set("gain", g)
        cam.set("exposure_us", lo)
        record(cam, session / "read_noise" / f"{g:g}dB", frames, {"section": "read_noise"})
    for g in gains:
        cam.set("gain", g)
        for ms in dark_exposures_ms:
            us = cam.set("exposure_us", ms * 1e3)
            record(cam, session / "dark" / f"{g:g}dB" / exposure_label(us), dark_frames, {"section": "dark"})
    cam.set("gain", shot_gain)
    cam.set("exposure_us", shot_exposure_ms * 1e3)
    lit = session / "shot" / "light1"
    cam.covered = False
    record(cam, lit, frames, {"section": "shot", "label": "light1", "notes": "simulated"})
    cam.covered = True
    record(cam, session / "shot" / "light1_dark", frames, {"section": "shot_dark", "label": "light1", "pair": str(lit)})
    return cam


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, help="folder to create (must not already hold recordings)")
    p.add_argument("--shape", type=int, nargs=2, default=(300, 480), metavar=("ROWS", "COLS"))
    p.add_argument("--frames", type=int, default=50, help="frames per read noise and shot noise stack")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    cam = simulate_session(args.session, shape=tuple(args.shape), frames=args.frames, seed=args.seed)
    print(f"Wrote {args.session}")
    print(f"Simulated truth: base gain {cam.base_gain} counts/e- at 0 dB, read noise {cam.read_noise_e} e-, "
          f"dark current {cam.dark_e_per_s} e-/s, black level {BLACK_LEVEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

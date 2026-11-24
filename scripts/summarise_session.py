#!/usr/bin/env python
"""Write the numbers the slides show, calculated from a recorded session, to a small JSON file.

The recordings are far too big for git, so the slides read this summary instead, and CI can
render them. Run it again after recording a session the slides should show, and commit the file:

    .venv/bin/python scripts/summarise_session.py data/live/practical results/practical.json

Everything is calculated as in basler_live.py, from every 4th pixel in each direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import noise_analysis as na
from recording import find_recordings, load_metadata, load_recording, write_json

GAINS_DB = (0, 1, 12, 24)
DARK_EXPOSURE_S = 1e-3  # the dynamic range is for this exposure


def dark_fit(path: Path) -> na.DarkFit:
    """Dark counts against exposure, leaving out frames where the camera's offset jumped."""
    metas = [load_metadata(p) for p in find_recordings(path)]
    means = [np.array(m["frame_means"]) for m in metas]
    return na.dark_counts([m["exposure_us"] * 1e-6 for m in metas], [m[na.steady_frames(m)].mean() for m in means])


def summarise(session: Path, step: int = 4) -> dict:
    gains = []
    for gain in GAINS_DB:
        covered, covered_meta = load_recording(session / "read_noise" / f"{gain}dB", step)
        lit, meta = load_recording(session / "shot" / f"{gain}dB", step)
        assert meta["black_level"] == covered_meta["black_level"], "record read noise at the same black level"
        max_value = meta["max_value"]
        read = na.read_noise(covered, max_value)
        dark = dark_fit(session / "dark" / f"{gain}dB")
        dark_count = dark.subtracted[np.isclose(dark.exposures_s, DARK_EXPOSURE_S)][0]
        ratio, db = na.dynamic_range(max_value, read.bias, dark_count, read.read_noise)
        gains.append(
            {
                "gain_db": gain,
                "bias": read.bias,
                "read_noise": read.read_noise,
                "dark_bias": dark.bias,
                "dark_counts_per_s": dark.slope,
                "saturation": max_value - read.bias,
                "dynamic_range": ratio,
                "dynamic_range_db": db,
                "counts_per_e": na.shot_noise(lit, read.bias, max_value).fit.slope,
            }
        )
    return {"camera_model": meta["camera_model"], "recorded": meta["saved"][:10], "gains": gains}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, help="session folder, e.g. data/live/practical")
    p.add_argument("output", type=Path, help="JSON file to write, e.g. results/practical.json")
    args = p.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summarise(args.session))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Record the data for the noise practical from a SiPM on a Rigol DS1000Z.

Run one step at a time, in order, with the same session name. Set Ch1's V/div and
offset first, with the source in place, so the signal stays on screen. They must then
stay fixed, and later steps refuse to run if they have changed.

    .venv/bin/python scripts/sipm_acquire.py read   --session monday                # Exercise 5
    .venv/bin/python scripts/sipm_acquire.py dark   --session monday --bias-step 5  # Exercise 6
    .venv/bin/python scripts/sipm_acquire.py source --session monday --bias-step 5  # Exercise 8
    .venv/bin/python scripts/sipm_analyse.py data/sipm/monday                       # Results

The detector's bias knob can't be read by the computer, so --bias-step records it.

Repeat the source step with --label to compare positions of the vial.

Each step takes two sets of captures, saved as raw 8-bit codes:
  fast  resolves individual pulses, for noise, gain and pulse sizes
  slow  covers exposures up to 1 s, for means only, as it can miss narrow pulses
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from sipm_exposure_sweep import (
    Scope,
    SettingsError,
    _restore_or_warn,
    clipped_fraction,
    estimate_run_seconds,
    plan_capture,
)

STEP_PROMPTS = {
    "read": (
        "Unplug the detector from Ch1, leaving any 50 ohm terminator in place, "
        "then press Enter... "
    ),
    "dark": (
        "Reconnect the detector. Cover the SiPM with no source present, let it settle, "
        "then press Enter... "
    ),
    "source": (
        "Place the tritium vial over the SiPM and cover both to block room light, "
        "then press Enter... "
    ),
}
CAPTURE_SETTINGS = ("sample_rate", "capture_s")


@dataclass(frozen=True)
class CaptureSet:
    name: str
    sample_rate: float
    capture_s: float
    repeats: int


def read_channel(scope: Scope) -> tuple[str, dict]:
    scope.check_channels()
    return (
        scope.query("*IDN?"),
        {
            "scale_v_per_div": float(scope.query(":CHAN1:SCAL?")),
            "offset_v": float(scope.query(":CHAN1:OFFS?")),
            "probe": float(scope.query(":CHAN1:PROB?")),
            "coupling": scope.query(":CHAN1:COUP?"),
            "bandwidth_limit": scope.query(":CHAN1:BWL?"),
        },
    )


def _check_matches_session(
    session: dict, channel: dict, sets: list[CaptureSet], bias_step: int | None
) -> None:
    """Steps are only comparable if taken with the same scope and detector settings."""
    recorded_bias = session.get("bias_step")
    if bias_step is not None and recorded_bias is not None and bias_step != recorded_bias:
        raise SettingsError(
            f"The detector's bias step is {bias_step}, but this session used {recorded_bias}. "
            "Put it back, or start a new session."
        )
    for key, value in session["channel_1"].items():
        if channel[key] != value:
            raise SettingsError(
                f"Ch1 {key} is {channel[key]}, but this session used {value}. "
                "Put it back, or start a new session."
            )
    for capture_set in sets:
        recorded = session["capture_sets"].get(capture_set.name)
        if recorded is None:
            raise SettingsError(f"This session has no {capture_set.name} set.")
        if any(recorded[key] != getattr(capture_set, key) for key in CAPTURE_SETTINGS):
            raise SettingsError(
                f"The {capture_set.name} set differs from this session's: {recorded}."
            )


def run_step(
    scope: Scope,
    session_dir: Path,
    step: str,
    label: str,
    sets: list[CaptureSet],
    prompt=input,
    bias_step: int | None = None,
) -> Path:
    """Record one step of the practical and return its folder."""
    # Checked before touching the scope, so a bad setting changes nothing.
    plans = [plan_capture(s.capture_s, s.sample_rate) for s in sets]
    step_dir = session_dir / (f"{step}_{label}" if label else step)
    if step_dir.exists():
        raise SettingsError(
            f"{step_dir} is already recorded. Delete it to retake it, or use a new --label."
        )

    session_path = session_dir / "session.json"
    session = json.loads(session_path.read_text()) if session_path.exists() else None
    (_, channel), _ = scope.with_recovery(lambda: read_channel(scope))
    if session:
        _check_matches_session(session, channel, sets, bias_step)
    print(f"Ch1: {channel['scale_v_per_div'] * 1e3:g} mV/div")

    prompt(STEP_PROMPTS[step])

    # Read again, because the prompt is when someone might touch the scope.
    (idn, channel), _ = scope.with_recovery(lambda: read_channel(scope))
    if session:
        _check_matches_session(session, channel, sets, bias_step)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
        session = {
            "instrument": idn,
            "channel_1": channel,
            "capture_sets": {s.name: asdict(s) for s in sets},
            "bias_step": None,
        }
    if bias_step is not None:
        session["bias_step"] = bias_step
    session_path.write_text(json.dumps(session, indent=2))

    settings = scope.with_recovery(scope.save_settings)[0]
    try:
        minutes = sum(estimate_run_seconds([p], s.repeats) for p, s in zip(plans, sets)) / 60
        print(f"Expected run time: about {minutes:.1f} min. The screen holds still during it.")
        step_dir.mkdir()
        metadata = {
            "step": step,
            "label": label,
            "started": datetime.now().isoformat(timespec="seconds"),
            "sets": {},
        }
        metadata_path = step_dir / "metadata.json"
        for capture_set, plan in zip(sets, plans):
            scope.prepare(plan)
            entry = {"plan": asdict(plan), "captures": [], "retries": 0}
            metadata["sets"][capture_set.name] = entry
            for repeat in range(capture_set.repeats):
                codes, preamble, retries = scope.measure(plan)
                filename = f"{capture_set.name}_{repeat:03d}.npy"
                np.save(step_dir / filename, codes.astype(np.uint8))
                entry.update(
                    sample_rate=plan.sample_rate,
                    samples=plan.samples,
                    xinc=preamble.xinc,
                    yinc=preamble.yinc,
                )
                entry["captures"].append(filename)
                entry["retries"] += retries
                # Written as it goes, so a crash partway keeps what was measured.
                metadata_path.write_text(json.dumps(metadata, indent=2))
                print(
                    f"  {capture_set.name} {repeat + 1}/{capture_set.repeats}: "
                    f"mean {codes.mean():.2f}, std {codes.std():.2f} codes"
                )
                clipped = clipped_fraction(codes)
                if clipped > 0:
                    print(
                        f"  WARNING: {clipped:.2%} of samples hit the ADC limits. "
                        "Stop, then change the V/div or offset in a new session."
                    )
    finally:
        _restore_or_warn(scope, settings)
    print(f"\nSaved to {step_dir}")
    return step_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=sorted(STEP_PROMPTS))
    parser.add_argument("--session", required=True, help="session folder name")
    parser.add_argument(
        "--label", default="", help="source step only: name this run (default: vial)"
    )
    parser.add_argument(
        "--bias-step", type=int, help="the detector's bias knob; needed for dark and source"
    )
    parser.add_argument(
        "--fast-rate", type=float, default=5e8,
        help="fast set samples per second; pulses must span several samples (default: 5e8)",
    )
    parser.add_argument(
        "--fast-capture", type=float, default=2.4e-3,
        help="fast set seconds per capture (default: 2.4e-3)",
    )
    parser.add_argument("--fast-repeats", type=int, default=40, help="fast captures (default: 40)")
    parser.add_argument("--slow-repeats", type=int, default=10, help="slow captures (default: 10)")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sipm",
        help="where sessions live (default: data/sipm)",
    )
    parser.add_argument("--resource", help="VISA resource (default: first Rigol found)")
    args = parser.parse_args(argv)
    if args.label and args.step != "source":
        parser.error("--label only applies to the source step")
    if args.step != "read" and args.bias_step is None:
        parser.error(f"the {args.step} step needs --bias-step")
    if args.fast_repeats < 2 or args.slow_repeats < 2:
        parser.error("repeats must be at least 2")
    if args.step == "source" and not args.label:
        args.label = "vial"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sets = [
        CaptureSet("fast", args.fast_rate, args.fast_capture, args.fast_repeats),
        CaptureSet("slow", 1e6, 1.2, args.slow_repeats),
    ]
    scope = None
    try:
        # Check the plans first, so a bad setting fails before asking for the scope.
        for capture_set in sets:
            plan_capture(capture_set.capture_s, capture_set.sample_rate)
        scope = Scope(args.resource)
        run_step(
            scope, args.out_dir / args.session, args.step, args.label, sets,
            bias_step=args.bias_step,
        )
        return 0
    except (SettingsError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Captures taken so far are saved.", file=sys.stderr)
        return 130
    finally:
        if scope is not None:
            scope.close()


if __name__ == "__main__":
    sys.exit(main())

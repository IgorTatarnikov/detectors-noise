#!/usr/bin/env python
"""Measure a SiPM's integrated signal against exposure with a Rigol DS1000Z.

Each exposure is a continuous capture on Ch1 at a fixed sample rate, repeated so
the spread between repeats gives the noise. The signal is integrated in the
scope's raw 8-bit units ("counts"), after subtracting a baseline measured with
the detector unplugged:

    counts      = sum(code - baseline_code)
    counts/sec  = counts / exposure

Every repeat's integral is saved, so variance against mean can be used later to
convert counts to photons. Only compare runs taken at the same V/div and sample
rate, because both change what one count means.

Usage:
    .venv/bin/python scripts/sipm_exposure_sweep.py --label covered
    .venv/bin/python scripts/sipm_exposure_sweep.py --exposures 1e-3 0.1 --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

RIGOL_USB_VENDOR = 0x1AB1
RIGOL_VISA_VENDOR = "6833"

DEFAULT_EXPOSURES = (40e-6, 1e-3, 10e-3, 0.1, 1.0)
DEFAULT_SAMPLE_RATE = 1e6
DEFAULT_REPEATS = 10
BASELINE_EXPOSURE = 1.0
BASELINE_REPEATS = 5

# Single-channel memory depths on a DS1074Z with standard memory.
MEMORY_DEPTHS = (12_000, 120_000, 1_200_000, 12_000_000)
SCREEN_DIVISIONS = 12
# Up to this rate, depth / (12 * timebase scale) gives the rate. Above it the memory
# runs past the screen and only certain rates exist, as measured on a DS1074Z with
# one channel on: rate -> (depth, timebase scale).
MAX_FORMULA_RATE = 1e7
MEASURED_HIGH_RATES = {
    2.5e7: ((1_200_000, 2e-3),),
    5e7: ((12_000, 1e-5), (120_000, 1e-4), (1_200_000, 1e-3), (12_000_000, 1e-2)),
    1.25e8: ((1_200_000, 5e-4),),
    5e8: ((1_200_000, 1e-4), (12_000_000, 1e-3)),
}
ADC_MIN, ADC_MAX = 0, 255

# The scope returns empty reads or wedges its USB endpoint without a pause after
# each command and before reading a reply. Over USB, 50 ms was clean for 35 full
# capture-and-read cycles, and 30 ms or less failed every time. Don't go lower.
COMMAND_GAP_S = 0.05
READ_CHUNK_POINTS = 250_000
MAX_CAPTURE_ATTEMPTS = 5
RUN_TIMEOUT_S = 5.0

# Measured on a DS1074Z over USB, used only for the run-time estimate.
READ_S_PER_POINT = 2.3e-6
CAPTURE_OVERHEAD_S = 0.5


class CaptureError(RuntimeError):
    """A capture came back incomplete or inconsistent, so it can be retried."""


class SettingsError(RuntimeError):
    """The scope cannot run this measurement as set up, so retrying won't help."""


@dataclass(frozen=True)
class Preamble:
    format: int
    type: int
    points: int
    count: int
    xinc: float
    xorigin: float
    xref: float
    yinc: float
    yorigin: float
    yref: float

    @property
    def zero_volt_code(self) -> float:
        return self.yorigin + self.yref


@dataclass(frozen=True)
class CapturePlan:
    exposure_s: float
    timebase_scale: float
    memory_depth: int
    samples: int
    sample_rate: float

    @property
    def window_s(self) -> float:
        """How long the memory takes to fill, which can be longer than the screen."""
        return self.memory_depth / self.sample_rate


@dataclass(frozen=True)
class ExposureSummary:
    exposure_s: float
    repeats: int
    mean_counts: float
    var_counts: float
    mean_counts_per_s: float
    std_counts_per_s: float


def parse_preamble(text: str) -> Preamble:
    fields = text.strip().split(",")
    if len(fields) != 10:
        raise ValueError(f"expected 10 fields in :WAV:PRE? reply, got {text!r}")
    v = [float(x) for x in fields]
    return Preamble(
        int(v[0]), int(v[1]), int(v[2]), int(v[3]), v[4], v[5], v[6], v[7], v[8], v[9]
    )


def _is_timebase_step(scale: float) -> bool:
    """The DS1000Z timebase only steps in a 1-2-5 sequence."""
    mantissa = scale / 10 ** math.floor(math.log10(scale) + 1e-9)
    return any(math.isclose(mantissa, step, rel_tol=1e-6) for step in (1, 2, 5))


def plan_capture(exposure_s: float, sample_rate: float) -> CapturePlan:
    """Pick the timebase and depth that give this sample rate and fit the exposure.

    The scope has no command to set its sample rate directly. It follows from the
    depth and timebase scale, so it is set by choosing those two.
    """
    if sample_rate > MAX_FORMULA_RATE:
        usable = list(MEASURED_HIGH_RATES.get(sample_rate, ()))
    else:
        usable = [
            (depth, depth / (SCREEN_DIVISIONS * sample_rate))
            for depth in MEMORY_DEPTHS
            if _is_timebase_step(depth / (SCREEN_DIVISIONS * sample_rate))
        ]
    if not usable:
        raise ValueError(f"sample rate {sample_rate:g} Sa/s is not reachable")

    samples = round(exposure_s * sample_rate)
    if samples < 1:
        raise ValueError(
            f"exposure {exposure_s:g} s is shorter than one sample at {sample_rate:g} Sa/s"
        )
    for depth, scale in usable:
        if samples <= depth:
            return CapturePlan(samples / sample_rate, scale, depth, samples, sample_rate)
    raise ValueError(
        f"exposure {exposure_s:g} s is longer than memory holds at {sample_rate:g} Sa/s"
    )


def integrate_counts(codes: np.ndarray, baseline_code: float) -> float:
    return float(np.sum(codes - baseline_code))


def clipped_fraction(codes: np.ndarray) -> float:
    return float(np.mean((codes <= ADC_MIN) | (codes >= ADC_MAX)))


def summarize(rows: list[dict]) -> list[ExposureSummary]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["exposure_s"], []).append(row["counts"])

    summaries = []
    for exposure in sorted(grouped):
        counts = np.asarray(grouped[exposure], dtype=float)
        var = float(np.var(counts, ddof=1)) if counts.size > 1 else math.nan
        summaries.append(
            ExposureSummary(
                exposure_s=exposure,
                repeats=int(counts.size),
                mean_counts=float(counts.mean()),
                var_counts=var,
                mean_counts_per_s=float(counts.mean()) / exposure,
                std_counts_per_s=math.sqrt(var) / exposure,
            )
        )
    return summaries


def estimate_run_seconds(plans: list[CapturePlan], repeats: int) -> float:
    per_repeat = sum(
        p.window_s + CAPTURE_OVERHEAD_S + p.samples * READ_S_PER_POINT for p in plans
    )
    return per_repeat * repeats


def _usb_errors() -> tuple:
    """What a USB fault can raise. pyvisa-py uses struct.error for an empty read and
    ValueError for a failed write, as well as the USB and VISA errors."""
    import pyvisa
    import usb.core

    return (
        TimeoutError,
        ValueError,
        struct.error,
        usb.core.USBError,
        pyvisa.errors.VisaIOError,
        pyvisa.errors.InvalidSession,
    )


class Scope:
    """Paced access to the DS1000Z over USB, with recovery from a wedged endpoint."""

    SETTINGS = (":ACQ:TYPE", ":ACQ:MDEP", ":TIM:SCAL", ":TIM:OFFS", ":TRIG:SWE")

    def __init__(self, resource: str | None = None):
        import pyvisa

        self._rm = pyvisa.ResourceManager("@py")
        self.resource = resource or self._find_resource()
        self._session = None
        try:
            self._open()
        except _usb_errors():
            # An endpoint left wedged by an earlier run fails on opening. Seen after
            # a clean exit, so it can happen between any two steps of a measurement.
            self.recover()

    def _find_resource(self) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pyvisa-py warns about unused LAN discovery.
            found = [r for r in self._rm.list_resources() if RIGOL_VISA_VENDOR in r]
        if not found:
            raise SettingsError("No Rigol scope found over USB. Is it connected and on?")
        return found[0]

    def _open(self) -> None:
        self._session = self._rm.open_resource(self.resource)
        self._session.timeout = 60_000

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def write(self, command: str) -> None:
        self._session.write(command)
        time.sleep(COMMAND_GAP_S)

    def query(self, command: str) -> str:
        self.write(command)
        return self._session.read().strip()

    def recover(self) -> None:
        """Reset the USB port, which clears a wedged endpoint and keeps settings."""
        import libusb_package
        import usb.util

        try:
            self.close()
        except Exception:
            self._session = None
        device = libusb_package.find(idVendor=RIGOL_USB_VENDOR)
        if device is None:
            raise SettingsError("The scope has disappeared from USB.")
        device.reset()
        usb.util.dispose_resources(device)
        time.sleep(3)
        self._open()
        # The first exchange after a reset can time out, so allow a second try.
        self._session.timeout = 5_000
        try:
            self.query("*IDN?")
        except Exception:
            self.query("*IDN?")
        self._session.timeout = 60_000

    def check_channels(self) -> None:
        if self.query(":CHAN1:DISP?") != "1":
            raise SettingsError("Ch1 is off. Turn it on before measuring.")
        for channel in (2, 3, 4):
            if self.query(f":CHAN{channel}:DISP?") != "0":
                raise SettingsError(
                    f"Ch{channel} is on, which reduces the memory per channel. "
                    "Turn off every channel except Ch1."
                )

    def save_settings(self) -> dict:
        settings = {key: self.query(f"{key}?") for key in self.SETTINGS}
        settings["running"] = self.query(":TRIG:STAT?") != "STOP"
        return settings

    def restore_settings(self, settings: dict) -> list[str]:
        """Put the scope back and return any settings that did not come back."""
        # Sweep first and run before depth: :RUN under single sweep arms one
        # capture and stops, and depth can only be changed while running.
        self.write(":TRIG:SWE AUTO")
        self.write(":RUN")
        time.sleep(0.5)
        self.write(f":ACQ:TYPE {settings[':ACQ:TYPE']}")
        self.write(f":TIM:OFFS {settings[':TIM:OFFS']}")
        self.write(f":TIM:SCAL {settings[':TIM:SCAL']}")
        time.sleep(0.3)
        self.write(f":ACQ:MDEP {settings[':ACQ:MDEP']}")
        time.sleep(0.5)
        # Coming from a slow timebase, the first command stops at 5 us/div.
        self.write(f":TIM:SCAL {settings[':TIM:SCAL']}")
        time.sleep(0.3)
        self.write(f":TRIG:SWE {settings[':TRIG:SWE']}")
        if not settings["running"]:
            self.write(":STOP")

        mismatched = []
        for key in self.SETTINGS:
            now = self.query(f"{key}?")
            if not _same_setting(now, settings[key]):
                mismatched.append(f"{key} is {now}, was {settings[key]}")
        return mismatched

    def configure(self, plan: CapturePlan) -> None:
        self.write(":TRIG:SWE AUTO")
        self.write(":RUN")
        time.sleep(0.2)
        self.write(":ACQ:TYPE NORM")
        self.write(f":TIM:SCAL {plan.timebase_scale}")
        # Trigger at the left edge, so the whole memory fills after the force.
        self.write(f":TIM:OFFS {plan.timebase_scale * SCREEN_DIVISIONS / 2}")
        self.write(f":ACQ:MDEP {plan.memory_depth}")
        time.sleep(0.3)

        rate = float(self.query(":ACQ:SRAT?"))
        depth = self.query(":ACQ:MDEP?")
        if not math.isclose(rate, plan.sample_rate, rel_tol=1e-3):
            raise SettingsError(f"Scope is at {rate:g} Sa/s, wanted {plan.sample_rate:g}.")
        if depth != str(plan.memory_depth):
            raise SettingsError(f"Scope memory depth is {depth}, wanted {plan.memory_depth}.")

    def capture(self, plan: CapturePlan) -> None:
        """Take one forced capture, so the data is not selected by a trigger event."""
        self._start_running()
        self.write(":SING")
        started = time.monotonic()
        deadline = started + plan.window_s * 3 + 20
        last_force = None
        status = "?"
        while time.monotonic() < deadline:
            status = self.query(":TRIG:STAT?")
            now = time.monotonic()
            if status == "STOP":
                return
            if status in ("WAIT", "RUN", "AUTO") and (
                last_force is None or (status == "WAIT" and now - last_force > 1)
            ):
                self.write(":TFOR")
                last_force = now
        raise CaptureError(f"capture did not finish, last status {status}")

    def _start_running(self) -> None:
        """Leave STOP first, so that any STOP after :SING means a finished capture.

        :SING from STOP takes over 100 ms to re-arm, and the status reads STOP until
        then, which can't be told apart from a capture that has finished.
        """
        # Sweep first, because :RUN under single sweep arms one capture and stops.
        self.write(":TRIG:SWE AUTO")
        self.write(":RUN")
        deadline = time.monotonic() + RUN_TIMEOUT_S
        while True:
            status = self.query(":TRIG:STAT?")
            if status != "STOP":
                return
            if time.monotonic() >= deadline:
                raise CaptureError(f"scope did not start running, last status {status}")

    def read_codes(self, plan: CapturePlan) -> tuple[np.ndarray, Preamble]:
        self.write(":WAV:SOUR CHAN1")
        self.write(":WAV:MODE RAW")
        self.write(":WAV:FORM BYTE")
        preamble = parse_preamble(self.query(":WAV:PRE?"))
        if preamble.points != plan.memory_depth:
            raise CaptureError(
                f"scope holds {preamble.points} points, expected {plan.memory_depth}"
            )
        if not math.isclose(preamble.xinc, 1 / plan.sample_rate, rel_tol=1e-3):
            raise CaptureError(f"sample interval is {preamble.xinc:g} s")

        chunks = []
        for start in range(1, plan.samples + 1, READ_CHUNK_POINTS):
            stop = min(start + READ_CHUNK_POINTS - 1, plan.samples)
            self.write(f":WAV:STAR {start}")
            self.write(f":WAV:STOP {stop}")
            chunk = self._session.query_binary_values(
                ":WAV:DATA?", datatype="B", container=np.array, delay=COMMAND_GAP_S
            )
            if chunk.size != stop - start + 1:
                raise CaptureError(f"read {chunk.size} points of {stop - start + 1}")
            chunks.append(chunk)
        return np.concatenate(chunks).astype(float), preamble

    def with_recovery(self, action, before_retry=None) -> tuple[object, int]:
        """Run a hardware step, resetting USB after a fault. Returns (result, retries)."""
        retryable = (CaptureError, *_usb_errors())
        for attempt in range(MAX_CAPTURE_ATTEMPTS):
            try:
                if attempt:
                    self.recover()
                    if before_retry is not None:
                        before_retry()
                return action(), attempt
            except retryable as error:
                kind = "Capture problem" if isinstance(error, CaptureError) else "USB fault"
                print(f"    {kind}, retrying ({type(error).__name__}: {str(error)[:80]})")
                if attempt == MAX_CAPTURE_ATTEMPTS - 1:
                    raise
        raise AssertionError("unreachable")

    def measure(self, plan: CapturePlan) -> tuple[np.ndarray, Preamble, int]:
        """Capture and read once, recovering from USB faults. Returns retries used."""

        def capture_and_read():
            self.capture(plan)
            return self.read_codes(plan)

        (codes, preamble), retries = self.with_recovery(
            capture_and_read, before_retry=lambda: self.configure(plan)
        )
        return codes, preamble, retries

    def prepare(self, plan: CapturePlan) -> None:
        self.with_recovery(lambda: self.configure(plan))


def _same_setting(now: str, saved: str) -> bool:
    try:
        return math.isclose(float(now), float(saved), rel_tol=1e-6, abs_tol=1e-12)
    except ValueError:
        return now == saved


def measure_baseline(scope: Scope, sample_rate: float) -> list[float]:
    """Return the mean code of each baseline capture, with the detector unplugged."""
    input(
        "\nUnplug the detector from Ch1, leaving any 50 ohm terminator in place, "
        "then press Enter to record the baseline... "
    )
    plan = plan_capture(BASELINE_EXPOSURE, sample_rate)
    scope.prepare(plan)
    means = []
    for repeat in range(BASELINE_REPEATS):
        codes, _, _ = scope.measure(plan)
        means.append(float(codes.mean()))
        print(f"  baseline {repeat + 1}/{BASELINE_REPEATS}: {means[-1]:.3f} codes")
    print(f"Baseline: {np.mean(means):.3f} codes (spread {np.std(means, ddof=1):.3f})")
    input("Reconnect the detector, let it settle, then press Enter to start... ")
    return means


def counts_axis_scale(mean_counts: list[float]) -> str:
    """Use a log axis unless a mean is at or below zero, which log cannot show."""
    return "log" if all(mean > 0 for mean in mean_counts) else "symlog"


def _plot_against_exposure(
    summaries: list[ExposureSummary],
    values: list[float],
    spreads: list[float],
    ylabel: str,
    yscale: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series_color = "#2a78d6"
    text_primary = "#0b0b0b"
    text_secondary = "#52514e"
    grid_color = "#e4e3df"

    exposure = [s.exposure_s for s in summaries]
    yerr = [0 if math.isnan(spread) else spread for spread in spreads]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150, layout="constrained")
    ax.errorbar(
        exposure, values, yerr=yerr, color=series_color, marker="o", markersize=6,
        linewidth=2, elinewidth=1.5, capsize=4, markeredgecolor="white", markeredgewidth=1,
    )
    ax.set_xscale("log")
    if yscale == "log":
        # A lower error bar reaching zero is cut at the axis rather than dropped.
        ax.set_yscale("log", nonpositive="clip")
    elif yscale == "symlog":
        smallest = min((abs(v) for v in values if v != 0), default=1.0)
        ax.set_yscale("symlog", linthresh=smallest)
    ax.set_xlabel("Exposure time (s)", color=text_secondary)
    ax.set_ylabel(ylabel, color=text_secondary)
    ax.set_title(title, color=text_primary, loc="left", fontsize=11)
    ax.grid(True, color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=text_secondary, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid_color)
    fig.savefig(path)
    plt.close(fig)


def plot_summary(summaries: list[ExposureSummary], title: str, path: Path) -> None:
    _plot_against_exposure(
        summaries,
        [s.mean_counts_per_s for s in summaries],
        [s.std_counts_per_s for s in summaries],
        "Counts/sec (raw scope units)",
        "linear",
        title,
        path,
    )


def plot_counts(summaries: list[ExposureSummary], title: str, path: Path) -> None:
    means = [s.mean_counts for s in summaries]
    _plot_against_exposure(
        summaries,
        means,
        [math.sqrt(s.var_counts) for s in summaries],
        "Integrated counts (raw scope units)",
        counts_axis_scale(means),
        title,
        path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--exposures", type=float, nargs="+", default=list(DEFAULT_EXPOSURES),
        metavar="SECONDS", help="exposure times, up to 12 s (default: 40 us to 1 s)",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"captures per exposure (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE,
        help=f"samples per second (default: {DEFAULT_SAMPLE_RATE:g})",
    )
    parser.add_argument(
        "--baseline-code", type=float,
        help="reuse a baseline from an earlier run at the same V/div and offset, "
        "instead of being prompted to unplug the detector",
    )
    parser.add_argument("--label", default="", help="added to the output folder name")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sipm",
        help="where to create the run folder (default: data/sipm)",
    )
    parser.add_argument("--resource", help="VISA resource (default: first Rigol found)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def run(args: argparse.Namespace) -> int:
    # Checked before touching the scope, so a bad exposure list changes nothing.
    plans = [plan_capture(t, args.sample_rate) for t in sorted(set(args.exposures))]
    if args.baseline_code is None:
        plan_capture(BASELINE_EXPOSURE, args.sample_rate)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / (f"{stamp}_{args.label}" if args.label else stamp)

    scope = Scope(args.resource)
    settings = None
    rows: list[dict] = []
    clipped_total = 0.0
    try:
        def read_starting_state():
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
                scope.save_settings(),
            )

        (idn, channel, settings), _ = scope.with_recovery(read_starting_state)
        print(f"{idn}\nCh1: {channel['scale_v_per_div'] * 1e3:g} mV/div")

        baseline = args.baseline_code
        baseline_source = "supplied"
        baseline_captures: list[float] = []
        if baseline is None:
            baseline_captures = measure_baseline(scope, args.sample_rate)
            baseline = float(np.mean(baseline_captures))
            baseline_source = "measured"

        minutes = estimate_run_seconds(plans, args.repeats) / 60
        print(f"\nExpected run time: about {minutes:.1f} min. The screen holds still during it.")

        run_dir.mkdir(parents=True)
        metadata = {
            "started": stamp,
            "instrument": idn,
            "channel_1": channel,
            "sample_rate": args.sample_rate,
            "acquisition_type": "NORM",
            "repeats": args.repeats,
            "baseline_code": baseline,
            "baseline_source": baseline_source,
            "baseline_capture_means": baseline_captures,
            "label": args.label,
            "counts_definition": "sum(code - baseline_code) over the exposure",
            "plans": [asdict(p) for p in plans],
        }
        # Written before measuring, because photon conversion needs these settings
        # even if the sweep later aborts.
        metadata_path = run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        fields = [
            "exposure_s", "repeat", "samples", "counts", "counts_per_s",
            "mean_code", "var_code", "clipped_fraction", "retries", "time",
        ]
        # Written as it goes, so a crash partway keeps everything measured so far.
        with (run_dir / "captures.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for plan in plans:
                scope.prepare(plan)
                for repeat in range(args.repeats):
                    codes, preamble, retries = scope.measure(plan)
                    counts = integrate_counts(codes, baseline)
                    row = {
                        "exposure_s": plan.exposure_s,
                        "repeat": repeat,
                        "samples": plan.samples,
                        "counts": counts,
                        "counts_per_s": counts / plan.exposure_s,
                        "mean_code": float(codes.mean()),
                        "var_code": float(codes.var()),
                        "clipped_fraction": clipped_fraction(codes),
                        "retries": retries,
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    clipped_total = max(clipped_total, row["clipped_fraction"])
                    print(
                        f"  {plan.exposure_s:g} s, repeat {repeat + 1}/{args.repeats}: "
                        f"{row['counts_per_s']:.4g} counts/s"
                    )
                metadata["preamble"] = asdict(preamble)
                metadata_path.write_text(json.dumps(metadata, indent=2))
    finally:
        if settings is not None:
            _restore_or_warn(scope, settings)
        scope.close()

    summaries = summarize(rows)
    with (run_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in summaries)
    title = f"SiPM counts/sec against exposure{f' ({args.label})' if args.label else ''}"
    plot_summary(summaries, title, run_dir / "counts_per_s.png")
    suffix = f" ({args.label})" if args.label else ""
    plot_counts(summaries, f"SiPM integrated counts against exposure{suffix}", run_dir / "counts.png")

    print(f"\nExposure (s)   Counts/sec     Spread (std)   Repeats")
    for s in summaries:
        print(
            f"{s.exposure_s:<14g} {s.mean_counts_per_s:<14.5g} "
            f"{s.std_counts_per_s:<14.3g} {s.repeats}"
        )
    if clipped_total > 0:
        print(
            f"\nWARNING: up to {clipped_total:.2%} of samples hit the ADC limits. "
            "The signal is clipping, so these counts are wrong. Change V/div or offset."
        )
    total_retries = sum(r["retries"] for r in rows)
    if total_retries:
        print(f"\n{total_retries} capture(s) were retried after USB faults (see captures.csv).")
    print(f"\nSaved to {run_dir}")
    return 0


def _restore_or_warn(scope: Scope, settings: dict) -> None:
    try:
        try:
            mismatched = scope.restore_settings(settings)
        except Exception:
            scope.recover()
            mismatched = scope.restore_settings(settings)
    except Exception as error:
        mismatched = [f"could not talk to the scope ({type(error).__name__})"]
    if mismatched:
        print("\nWARNING: the scope's settings were not fully restored:", file=sys.stderr)
        for line in mismatched:
            print(f"  {line}", file=sys.stderr)
        print(f"  Original settings: {settings}", file=sys.stderr)
    else:
        print("\nScope settings restored.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (SettingsError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Data measured so far is in captures.csv.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

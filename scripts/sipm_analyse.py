#!/usr/bin/env python
"""Work through the noise practical (Exercises 5-8) on a SiPM session.

Takes a session recorded by sipm_acquire.py and answers each exercise, saving
results.json and plots to <session>/analysis.

The SiPM has no shutter, so an "exposure" here is a window of samples summed in
software, and "counts" are sum(code - baseline) over that window. Three things
differ from the camera derivation in the practical:

- Every sample adds its own read noise, so read noise variance grows with
  exposure. In var = gain * mean + read variance, it moves into the slope and not
  the intercept. The gain fit therefore uses source minus dark at the same window.
- Variance comes from differences of adjacent windows, because the baseline drifts
  slowly enough to dominate a plain variance at long exposures.
- Means at different windows come from the same captures, so their proportionality
  to exposure holds by construction. The spread between captures is the real test
  of stability.

Usage:
    .venv/bin/python scripts/sipm_analyse.py data/sipm/<session>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from sipm_exposure_sweep import ADC_MAX, clipped_fraction

DEFAULT_MIN_WINDOW_S = 10e-6
MIN_PAIRS = 16

# Below this the 8-bit step, not the electronics, sets the measured noise.
MIN_NOISE_CODES = 0.5
# The noise of rounding to whole codes, which is as low as the noise can be taken to be.
ADC_STEP_NOISE_CODES = 1 / math.sqrt(12)

PULSE_THRESHOLD_SIGMA = 3.5
PULSE_SHAPE_SIGMA = 5.0
PULSE_PRE_SAMPLES = 10
PULSE_MIN_POST_SAMPLES = 20
PULSE_WINDOW_S = 100e-9
PULSE_MIN_SEPARATION_S = 50e-9
PULSE_MIN_SIGNIFICANCE = 5.0
MIN_PULSES = 50
MIN_SHAPE_PULSES = 10
# Below this many times the noise, pulses overlap the noise peaks too much for a clean count.
MIN_CLEAR_SNR = 6.0
# Pulse sizes carry about 10-15% uncertainty at low signal-to-noise. Well above this
# ratio, photons are arriving together.
MAX_GAIN_RATIO = 1.5
MAX_REDUCED_CHI2 = 3.0
MIN_FRACTION_EXPLAINED = 0.9

SERIES_COLOR = "#2a78d6"
SECOND_COLOR = "#eb6834"
FIT_COLOR = "#52514e"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


@dataclass(frozen=True)
class WindowStats:
    samples: int
    mean: float
    var: float
    var_se: float
    plain_var: float
    pairs: int


@dataclass(frozen=True)
class LineFit:
    slope: float
    slope_se: float
    intercept: float
    intercept_se: float
    reduced_chi2: float


def window_stats(captures: list[np.ndarray], samples: int, baseline: float) -> WindowStats:
    """Sum each capture over consecutive windows and describe the sums.

    var is half the mean squared difference of adjacent windows within a capture,
    so offsets between captures and slow drift within one are left out.
    """
    sums = []
    differences = []
    for codes in captures:
        windows = codes.size // samples
        if windows == 0:
            continue
        totals = codes[: windows * samples].reshape(windows, samples).sum(axis=1)
        totals = totals - samples * baseline
        sums.append(totals)
        differences.append(np.diff(totals))
    if not sums:
        raise ValueError(f"captures are shorter than a {samples} sample window")

    all_sums = np.concatenate(sums)
    all_differences = np.concatenate(differences)
    pairs = int(all_differences.size)
    var = float(np.mean(all_differences**2) / 2) if pairs else math.nan
    return WindowStats(
        samples=samples,
        mean=float(all_sums.mean()),
        var=var,
        # For adjacent differences of independent windows, var(var) = 3 var^2 / pairs.
        var_se=var * math.sqrt(3 / pairs) if pairs else math.nan,
        plain_var=float(all_sums.var(ddof=1)) if all_sums.size > 1 else math.nan,
        pairs=pairs,
    )


def window_sizes(
    capture_samples: int, sample_rate: float, captures: int, min_window_s: float, min_pairs: int
) -> list[int]:
    """Window lengths in samples, on a 1-2-5 exposure sequence, that the data can support."""
    sizes = []
    start = math.floor(math.log10(min_window_s) - 1e-9)
    for decade in range(start, start + 12):
        for step in (1, 2, 5):
            exposure = step * 10.0**decade
            if exposure < min_window_s * (1 - 1e-9):
                continue
            samples = round(exposure * sample_rate)
            windows = capture_samples // samples if samples else 0
            if samples and windows >= 1 and (windows - 1) * captures >= min_pairs:
                sizes.append(samples)
    return sizes


def fit_line(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> LineFit:
    """Inverse-variance weighted straight line."""
    if x.size < 2:
        raise ValueError("a line needs at least two points")
    weights = 1 / yerr**2
    total = weights.sum()
    x_bar = (weights * x).sum() / total
    y_bar = (weights * y).sum() / total
    spread = (weights * (x - x_bar) ** 2).sum()
    slope = (weights * (x - x_bar) * (y - y_bar)).sum() / spread
    intercept = y_bar - slope * x_bar
    residuals = y - (slope * x + intercept)
    dof = x.size - 2
    return LineFit(
        slope=float(slope),
        slope_se=float(math.sqrt(1 / spread)),
        intercept=float(intercept),
        intercept_se=float(math.sqrt(1 / total + x_bar**2 / spread)),
        reduced_chi2=float((weights * residuals**2).sum() / dof) if dof else math.nan,
    )


def measure_read_noise(captures: list[np.ndarray], yinc: float) -> dict:
    """Exercise 5: the baseline and per-sample spread with the detector unplugged."""
    std = float(np.mean([codes.std(ddof=1) for codes in captures]))
    clipped = float(np.mean([clipped_fraction(codes) for codes in captures]))
    warnings = []
    if std < MIN_NOISE_CODES:
        warnings.append(
            f"Read noise is only {std:.2f} codes, below the ADC step, so it can't be measured at "
            "this V/div. That's fine if the detector's noise is much larger, as Exercise 6 shows."
        )
    if clipped > 0:
        warnings.append(
            f"{clipped:.2%} of samples clip at the ADC limits, which biases the noise low. "
            "Change the channel offset."
        )
    return {
        "baseline_code": float(np.mean([codes.mean() for codes in captures])),
        "std_code": std,
        "std_mv": std * yinc * 1e3,
        "clipped_fraction": clipped,
        "warnings": warnings,
    }


def gain_from_variance(
    source: list[np.ndarray],
    dark: list[np.ndarray],
    baseline: float,
    sample_rate: float,
    min_window_s: float = DEFAULT_MIN_WINDOW_S,
) -> dict:
    """Exercise 8: fit source minus dark variance against source minus dark mean."""
    sizes = window_sizes(source[0].size, sample_rate, len(source), min_window_s, MIN_PAIRS)
    if len(sizes) < 2:
        raise ValueError("captures are too short or too few for a variance against mean fit")

    rows = []
    for samples in sizes:
        lit = window_stats(source, samples, baseline)
        unlit = window_stats(dark, samples, baseline)
        delta_mean = lit.mean - unlit.mean
        delta_var = lit.var - unlit.var
        delta_var_se = math.hypot(lit.var_se, unlit.var_se)
        rows.append(
            {
                "exposure_s": samples / sample_rate,
                "samples": samples,
                "source_mean": lit.mean,
                "source_var": lit.var,
                "source_var_se": lit.var_se,
                "dark_mean": unlit.mean,
                "dark_var": unlit.var,
                "delta_mean": delta_mean,
                "delta_var": delta_var,
                "delta_var_se": delta_var_se,
                "ratio": delta_var / delta_mean if delta_mean > 0 else math.nan,
                "ratio_se": delta_var_se / delta_mean if delta_mean > 0 else math.nan,
                "pairs": lit.pairs,
            }
        )

    def column(name):
        return np.array([row[name] for row in rows])

    warnings = []
    if np.any(column("delta_mean") <= 0):
        warnings.append("The source is no brighter than the dark set, so the fit means nothing.")
    fit = fit_line(column("delta_mean"), column("delta_var"), column("delta_var_se"))
    raw = fit_line(column("source_mean"), column("source_var"), column("source_var_se"))
    if fit.reduced_chi2 > MAX_REDUCED_CHI2:
        warnings.append(
            f"Variance is not a straight line in the mean (reduced chi2 {fit.reduced_chi2:.1f}), "
            "so a single gain does not describe it. See the variance/mean plot."
        )
    return {
        "gain_counts_per_event": fit.slope,
        "gain_se": fit.slope_se,
        "intercept": fit.intercept,
        "intercept_se": fit.intercept_se,
        "reduced_chi2": fit.reduced_chi2,
        "raw_slope": raw.slope,
        "raw_intercept": raw.intercept,
        "windows": rows,
        "warnings": warnings,
    }


def measure_pulses(captures: list[np.ndarray], sample_rate: float) -> dict:
    """Size individual pulses, for a direct measure of counts per detected photon.

    Pulses can be barely above the noise, where any threshold cuts through them and
    both rate and size then depend on it. Noise is symmetric and pulses only go up,
    so downward peaks stand in for the noise among upward peaks. Subtracting them
    leaves the pulses, at a threshold low enough to include nearly all of them.

    The mean area includes crosstalk that fires with the pulse, so it converts counts
    to detected photons. effective_gain_counts = E[a^2] / E[a] is what variance over
    mean should give if pulses arrive independently, which exceeds the mean area by
    the excess noise factor E[a^2] / E[a]^2.
    """
    min_separation = max(1, round(PULSE_MIN_SEPARATION_S * sample_rate))
    post = max(PULSE_MIN_POST_SAMPLES, round(PULSE_WINDOW_S * sample_rate))
    signals, snippets, levels, spreads = [], [], [], []
    for codes in captures:
        centre = float(np.median(codes))
        # Pulses only go up, so samples at or below the median measure the noise.
        rough = math.sqrt(float(np.mean((codes[codes < centre] - centre) ** 2))) if (codes < centre).any() else 0.5
        level = float(codes[np.abs(codes - centre) < 2.5 * max(rough, 0.5)].mean())
        signal = codes - level
        spread = _lower_half_spread(signal)
        # Only peaks too tall to be noise, so the shape is the pulse's.
        tall, _ = find_peaks(signal, height=PULSE_SHAPE_SIGMA * spread, distance=post)
        snippets += [
            signal[peak - PULSE_PRE_SAMPLES : peak + post]
            for peak in tall
            if peak >= PULSE_PRE_SAMPLES and peak + post <= codes.size
        ]
        signals.append(signal)
        levels.append(level)
        spreads.append(spread)

    # The mean shape sets how far apart peaks must be, so a pulse's tail isn't counted
    # as another pulse. Before it is known, assume the shortest separation.
    shape = np.stack(snippets).mean(axis=0) if len(snippets) >= MIN_SHAPE_PULSES else None
    start, stop, decayed = 0, 0, False
    separation = min_separation
    if shape is not None:
        peak_value = shape[PULSE_PRE_SAMPLES]
        below = np.flatnonzero(shape < 0.05 * peak_value)
        before = below[below < PULSE_PRE_SAMPLES]
        after = below[below > PULSE_PRE_SAMPLES]
        start = before[-1] + 1 if before.size else 0
        decayed = bool(after.size)
        stop = after[0] if decayed else shape.size
        separation = max(min_separation, int(stop - PULSE_PRE_SAMPLES))

    heights_up, heights_down = [], []
    for signal, spread in zip(signals, spreads):
        threshold = PULSE_THRESHOLD_SIGMA * spread
        up, _ = find_peaks(signal, height=threshold, distance=separation)
        down, _ = find_peaks(-signal, height=threshold, distance=separation)
        heights_up.append(signal[up])
        heights_down.append(-signal[down])

    up_heights = np.concatenate(heights_up)
    down_heights = np.concatenate(heights_down)
    net = up_heights.size - down_heights.size
    noise = float(np.mean(spreads))
    total_samples = sum(codes.size for codes in captures)
    result = {
        "resolved": False,
        "count": int(net),
        "rate_per_s": net / (total_samples / sample_rate),
        "quiet_code": float(np.mean(levels)),
        "quiet_std_code": noise,
        "height_code": math.nan,
        "height_snr": math.nan,
        "mean_area_counts": math.nan,
        "effective_gain_counts": math.nan,
        "excess_noise_factor": math.nan,
        "fwhm_samples": math.nan,
        "heights_up": up_heights.tolist(),
        "heights_down": down_heights.tolist(),
        "warnings": [],
    }
    significant = net > PULSE_MIN_SIGNIFICANCE * math.sqrt(up_heights.size + down_heights.size)
    if net < MIN_PULSES or not significant or shape is None:
        result["warnings"].append(
            f"Found no clear excess of upward peaks ({up_heights.size} up, {down_heights.size} "
            "down), so pulses could not be sized. They are assumed positive, so check their "
            "polarity, then try a higher bias step."
        )
        return result

    area_per_height = float(shape[start:stop].sum() / peak_value)
    fwhm = int(np.count_nonzero(shape >= peak_value / 2))

    height = float((up_heights.sum() - down_heights.sum()) / net)
    mean_square = float((up_heights @ up_heights - down_heights @ down_heights) / net)
    area = height * area_per_height
    effective_gain = (mean_square - noise**2) / height * area_per_height
    result.update(
        resolved=True,
        height_code=height,
        height_snr=height / noise,
        mean_area_counts=area,
        effective_gain_counts=effective_gain,
        excess_noise_factor=effective_gain / area,
        fwhm_samples=fwhm,
    )
    if fwhm < 2:
        result["warnings"].append(
            f"Pulses span fewer than 2 samples at {sample_rate:g} Sa/s, so their areas and the "
            "gain are wrong. Raise the fast sample rate."
        )
    if not decayed:
        result["warnings"].append(
            f"Pulses have not decayed {post} samples after their peak, so their areas are too "
            "small."
        )
    if result["height_snr"] < MIN_CLEAR_SNR:
        result["warnings"].append(
            f"Pulses are only {result['height_snr']:.1f} times the noise, so many merge into it. "
            "In simulation this reads the rate about 25% low and the area about 10% high. A "
            "higher bias step makes pulses taller against the noise."
        )
    return result


def _lower_half_spread(signal: np.ndarray) -> float:
    """Noise from the samples below the level, which pulses cannot reach.

    Codes are whole numbers, so whichever code sits just below the level would count
    in full while the one just above counts not at all, reading the noise about 7%
    low. Spreading each sample over its code's width removes that, and the width's
    own 1/12 variance is then taken back out.
    """
    spread_out = signal + np.random.default_rng(0).uniform(-0.5, 0.5, signal.size)
    below = spread_out[spread_out < 0]
    return max(math.sqrt(max(float(np.mean(below**2)) - 1 / 12, 0.0)), 0.5)


def dynamic_range(saturation: float, baseline: float, dark: float, noise: float) -> float:
    """Exercise 7: (saturation - dark) / read noise, with bias removed from both."""
    return (saturation - baseline - dark) / noise


def load_set(step_dir: Path, name: str) -> tuple[list[np.ndarray], dict]:
    metadata = json.loads((step_dir / "metadata.json").read_text())
    entry = metadata["sets"].get(name)
    if entry is None or not entry["captures"]:
        raise ValueError(f"{step_dir} has no {name} captures, so the step did not finish")
    captures = [np.load(step_dir / filename).astype(float) for filename in entry["captures"]]
    return captures, entry


def analyse_read(session: Path, min_window_s: float) -> dict:
    captures, entry = load_set(session / "read", "fast")
    result = measure_read_noise(captures, entry["yinc"])
    sizes = window_sizes(entry["samples"], entry["sample_rate"], len(captures), min_window_s, MIN_PAIRS)
    baseline = result["baseline_code"]
    result["windows"] = []
    for samples in sizes:
        stats = window_stats(captures, samples, baseline)
        result["windows"].append(
            {
                "exposure_s": samples / entry["sample_rate"],
                "samples": samples,
                "var": stats.var,
                "var_se": stats.var_se,
            }
        )
    slow, _ = load_set(session / "read", "slow")
    result["slow_baseline_code"] = float(np.mean([codes.mean() for codes in slow]))
    return result


def analyse_dark(session: Path, read: dict) -> dict:
    slow, slow_entry = load_set(session / "dark", "slow")
    rate = slow_entry["sample_rate"]
    baseline = read["slow_baseline_code"]
    per_capture = np.array([codes.mean() - baseline for codes in slow])
    excess = float(per_capture.mean())
    excess_se = float(per_capture.std(ddof=1) / math.sqrt(per_capture.size))

    windows = []
    for samples in window_sizes(slow_entry["samples"], rate, len(slow), 100e-6, 0):
        stats = window_stats(slow, samples, baseline)
        windows.append(
            {
                "exposure_s": samples / rate,
                "samples": samples,
                "mean_counts": stats.mean,
                "plain_var": stats.plain_var,
            }
        )

    fast, fast_entry = load_set(session / "dark", "fast")
    fast_rate = fast_entry["sample_rate"]
    fast_excess = float(np.mean([codes.mean() for codes in fast])) - read["baseline_code"]
    pulses = measure_pulses(fast, fast_rate)
    from_pulses = None
    if pulses["resolved"] and fast_excess > 0:
        from_pulses = pulses["rate_per_s"] * pulses["mean_area_counts"] / fast_rate / fast_excess
    return {
        "counts_per_s": excess * rate,
        "counts_per_s_se": excess_se * rate,
        "sample_rate": rate,
        "mean_excess_code": excess,
        "mean_excess_mv": excess * slow_entry["yinc"] * 1e3,
        "capture_spread_mv": float(per_capture.std(ddof=1)) * slow_entry["yinc"] * 1e3,
        "fast_excess_code": fast_excess,
        "fast_std_code": float(np.mean([codes.std(ddof=1) for codes in fast])),
        "fraction_from_pulses": from_pulses,
        "windows": windows,
        "pulses": pulses,
    }


def analyse_dynamic_range(read: dict, dark: dict | None, saturation_code: float | None) -> dict:
    measured = saturation_code is not None
    saturation = saturation_code if measured else ADC_MAX
    baseline = read["baseline_code"]
    dark_excess = dark["fast_excess_code"] if dark else 0.0
    noise = max(read["std_code"], ADC_STEP_NOISE_CODES)
    per_sample = dynamic_range(saturation, baseline, dark_excess, noise)
    detector_on = None
    if dark and dark["pulses"]["quiet_std_code"] > 0:
        detector_on = dynamic_range(
            saturation, baseline, dark_excess, dark["pulses"]["quiet_std_code"]
        )
    return {
        "saturation_code": saturation,
        "saturation_source": "supplied" if measured else "ADC limit, not measured",
        "dark_included": dark is not None,
        "per_sample": per_sample,
        "per_sample_db": 20 * math.log10(per_sample),
        "per_sample_detector_on": detector_on,
        "limited_by_adc": read["std_code"] < ADC_STEP_NOISE_CODES,
    }


def analyse_source(
    step_dir: Path, dark_captures: list[np.ndarray], read: dict, dark: dict, min_window_s: float
) -> dict:
    captures, entry = load_set(step_dir, "fast")
    result = gain_from_variance(
        captures, dark_captures, read["baseline_code"], entry["sample_rate"], min_window_s
    )
    gain = result["gain_counts_per_event"]
    pulses = dark["pulses"]
    last = result["windows"][-1]
    delta_counts_per_s = last["delta_mean"] / last["exposure_s"]
    yinc, xinc = entry["yinc"], entry["xinc"]
    slow, slow_entry = load_set(step_dir, "slow")
    slow_excess = float(np.mean([codes.mean() for codes in slow])) - read["slow_baseline_code"]
    slow_mean_v = (slow_excess - dark["mean_excess_code"]) * slow_entry["yinc"]
    clipped = float(np.mean([clipped_fraction(codes) for codes in captures]))
    if clipped > 0:
        result["warnings"].append(
            f"{clipped:.2%} of samples clip at the ADC limits, so the variance is wrong. "
            "Change the V/div or offset in a new session."
        )
    result.update(
        gain_v_s=gain * entry["yinc"] * entry["xinc"],
        sample_rate=entry["sample_rate"],
        photons_per_count_variance=1 / gain if gain > 0 else None,
        events_per_s_variance=delta_counts_per_s / gain if gain > 0 else None,
        photons_per_count_pulses=None,
        events_per_s_pulses=None,
        gain_ratio=None,
        conversion_source="variance",
        excess_noise_factor=None,
    )
    counts_per_photon = gain if gain > 0 else None
    if pulses["resolved"]:
        counts_per_photon = pulses["mean_area_counts"]
        result.update(
            photons_per_count_pulses=1 / pulses["mean_area_counts"],
            events_per_s_pulses=delta_counts_per_s / pulses["mean_area_counts"],
            gain_ratio=gain / pulses["effective_gain_counts"],
            conversion_source="pulse areas",
            excess_noise_factor=pulses["excess_noise_factor"],
        )
    if counts_per_photon is None:
        result.update(photons_per_v_s=None, photons_per_count_by_rate={}, events_per_s_slow=None)
    else:
        photons_per_v_s = 1 / (counts_per_photon * yinc * xinc)
        result.update(
            photons_per_v_s=photons_per_v_s,
            photons_per_count_by_rate={
                rate: photons_per_v_s * yinc / rate
                for rate in (entry["sample_rate"], slow_entry["sample_rate"])
            },
            events_per_s_slow=slow_mean_v * photons_per_v_s,
        )
    if pulses["resolved"]:
        if result["gain_ratio"] > MAX_GAIN_RATIO:
            result["warnings"].append(
                f"The variance slope is {result['gain_ratio']:.1f} times what the dark pulse "
                "sizes predict. A tritium decay can emit a burst of photons, which raises the "
                "variance without changing the counts per photon, so trust the pulse value. "
                "Moving the vial back should bring this ratio towards 1 if bursts are the cause."
            )
    else:
        result["warnings"].append(
            "Dark pulses could not be sized, so nothing checks this slope. A burst of photons "
            "from one tritium decay would raise it, so treat it as an upper limit on the gain."
        )
    return result


def analyse_session(
    session: Path,
    min_window_s: float = DEFAULT_MIN_WINDOW_S,
    saturation_code: float | None = None,
) -> dict:
    """Analyse whichever steps have been recorded, and save results and plots."""
    session = Path(session)
    if not (session / "read" / "metadata.json").exists():
        raise ValueError(f"{session} has no read step, which everything else needs for its baseline")

    read = analyse_read(session, min_window_s)
    results: dict = {"session": str(session), "read": read, "dark": None, "source": {}}

    dark_captures = None
    if (session / "dark" / "metadata.json").exists():
        results["dark"] = analyse_dark(session, read)
        dark_captures, _ = load_set(session / "dark", "fast")

    results["dynamic_range"] = analyse_dynamic_range(read, results["dark"], saturation_code)

    if results["dark"] is not None:
        for step_dir in sorted(session.glob("source_*")):
            if (step_dir / "metadata.json").exists():
                label = step_dir.name.removeprefix("source_")
                results["source"][label] = analyse_source(
                    step_dir, dark_captures, read, results["dark"], min_window_s
                )

    out_dir = session / "analysis"
    out_dir.mkdir(exist_ok=True)
    save_plots(results, out_dir)
    saved = _without_nan(results)
    if saved["dark"] is not None:
        del saved["dark"]["pulses"]["heights_up"], saved["dark"]["pulses"]["heights_down"]
    (out_dir / "results.json").write_text(json.dumps(saved, indent=2))
    return results


def _without_nan(value):
    """JSON has no NaN, so an undefined value is saved as null."""
    if isinstance(value, dict):
        return {key: _without_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_nan(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150, layout="constrained")
    return fig, ax


def _finish(fig, ax, xlabel: str, ylabel: str, title: str, path: Path, legend: bool = False) -> None:
    import matplotlib.pyplot as plt

    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=11)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    if legend:
        ax.legend(frameon=False, labelcolor=TEXT_SECONDARY)
    fig.savefig(path)
    plt.close(fig)


POINT_STYLE = {"marker": "o", "markersize": 6, "markeredgecolor": "white", "markeredgewidth": 1}


def plot_read_noise(read: dict, path: Path) -> None:
    windows = read["windows"]
    exposure = np.array([w["exposure_s"] for w in windows])
    samples = np.array([w["samples"] for w in windows])
    fig, ax = _axes()
    ax.errorbar(
        exposure, [w["var"] for w in windows], yerr=[w["var_se"] for w in windows],
        color=SERIES_COLOR, linestyle="none", elinewidth=1.5, capsize=4, label="Measured",
        **POINT_STYLE,
    )
    ax.plot(
        exposure, samples * read["std_code"] ** 2, color=FIT_COLOR, linewidth=2,
        label="Per-sample variance x samples",
    )
    ax.set_xscale("log")
    ax.set_yscale("log", nonpositive="clip")
    _finish(
        fig, ax, "Exposure (s)", "Variance of integrated counts",
        "Read noise variance grows with exposure, detector unplugged", path, legend=True,
    )


def plot_dark_counts(dark: dict, path: Path) -> None:
    windows = dark["windows"]
    exposure = [w["exposure_s"] for w in windows]
    means = [w["mean_counts"] for w in windows]
    fig, ax = _axes()
    ax.plot(exposure, means, color=SERIES_COLOR, linewidth=2, **POINT_STYLE)
    ax.set_xscale("log")
    if all(mean > 0 for mean in means):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=min((abs(m) for m in means if m), default=1.0))
    _finish(
        fig, ax, "Exposure (s)", "Mean counts above baseline",
        f"Dark counts against exposure, {dark['sample_rate']:g} Sa/s", path,
    )


def plot_pulse_heights(pulses: dict, path: Path) -> None:
    fig, ax = _axes()
    top = max(max(pulses["heights_up"]), max(pulses["heights_down"], default=0))
    bins = np.linspace(0, top, 50)
    ax.hist(pulses["heights_up"], bins=bins, histtype="step", color=SERIES_COLOR, linewidth=2, label="Upward peaks")
    ax.hist(pulses["heights_down"], bins=bins, histtype="step", color=SECOND_COLOR, linewidth=2, label="Downward peaks, noise only")
    ax.set_yscale("log", nonpositive="clip")
    _finish(
        fig, ax, "Peak height (codes above the quiet level)", "Peaks",
        f"Dark pulses stand out where the two differ, mean {pulses['height_code']:.1f} codes",
        path, legend=True,
    )


def plot_gain(label: str, source: dict, path: Path) -> None:
    rows = [row for row in source["windows"] if row["delta_mean"] > 0 and row["delta_var"] > 0]
    fig, ax = _axes()
    ax.errorbar(
        [r["source_mean"] for r in source["windows"]],
        [r["source_var"] for r in source["windows"]],
        color=SECOND_COLOR, linestyle="none", label="Source, uncorrected", **POINT_STYLE,
    )
    x = np.array([r["delta_mean"] for r in rows])
    ax.errorbar(
        x, [r["delta_var"] for r in rows], yerr=[r["delta_var_se"] for r in rows],
        color=SERIES_COLOR, linestyle="none", elinewidth=1.5, capsize=4,
        label="Source minus dark", **POINT_STYLE,
    )
    if x.size:
        ax.plot(
            x, source["gain_counts_per_event"] * x + source["intercept"], color=FIT_COLOR,
            linewidth=2, label=f"Fit, slope {source['gain_counts_per_event']:.3g}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log", nonpositive="clip")
    _finish(
        fig, ax, "Mean counts", "Variance of counts",
        f"Variance against mean, {label}", path, legend=True,
    )


def plot_gain_ratio(label: str, source: dict, pulses: dict, path: Path) -> None:
    rows = [row for row in source["windows"] if row["delta_mean"] > 0]
    fig, ax = _axes()
    ax.errorbar(
        [r["exposure_s"] for r in rows], [r["ratio"] for r in rows],
        yerr=[r["ratio_se"] for r in rows], color=SERIES_COLOR, linestyle="none", elinewidth=1.5,
        capsize=4, label="Variance / mean", **POINT_STYLE,
    )
    if pulses["resolved"]:
        ax.axhline(
            pulses["effective_gain_counts"], color=FIT_COLOR, linewidth=2,
            label="Predicted from dark pulses",
        )
    ax.set_xscale("log")
    _finish(
        fig, ax, "Exposure (s)", "Variance / mean (counts per event)",
        f"Flat if photons arrive independently, {label}", path, legend=True,
    )


def save_plots(results: dict, out_dir: Path) -> None:
    plot_read_noise(results["read"], out_dir / "read_noise.png")
    dark = results["dark"]
    if dark is not None:
        plot_dark_counts(dark, out_dir / "dark_counts.png")
        if dark["pulses"]["resolved"]:
            plot_pulse_heights(dark["pulses"], out_dir / "pulse_heights.png")
    for label, source in results["source"].items():
        plot_gain(label, source, out_dir / f"gain_{label}.png")
        plot_gain_ratio(label, source, dark["pulses"], out_dir / f"gain_ratio_{label}.png")


def format_report(results: dict) -> str:
    lines: list[str] = []
    read = results["read"]
    lines += [
        "Exercise 5: read noise (detector unplugged)",
        f"  Baseline {read['baseline_code']:.2f} codes. "
        f"Read noise {read['std_code']:.2f} codes per sample, {read['std_mv']:.2f} mV.",
        "  This is the scope's noise only. The detector's amplifier adds to it, and the",
        "  gain fit below removes both.",
    ]
    if read["windows"]:
        first, last = read["windows"][0], read["windows"][-1]
        lines.append(
            f"  Integrated variance grows with exposure: {first['var']:.3g} at "
            f"{first['exposure_s']:g} s, {last['var']:.3g} at {last['exposure_s']:g} s."
        )
    lines += [f"  WARNING: {w}" for w in read["warnings"]]

    dark = results["dark"]
    lines.append("")
    if dark is None:
        lines.append("Exercise 6: not recorded yet")
    else:
        pulses = dark["pulses"]
        lines += [
            "Exercise 6: dark counts (SiPM covered)",
            f"  {'Exposure (s)':<14}Mean counts",
        ]
        lines += [f"  {w['exposure_s']:<14g}{w['mean_counts']:.4g}" for w in dark["windows"]]
        lines += [
            f"  Dark signal {dark['counts_per_s']:.4g} +/- {dark['counts_per_s_se']:.2g} counts/s "
            f"at {dark['sample_rate']:g} Sa/s, or {dark['mean_excess_mv']:.3f} mV above baseline.",
            f"  It varies by {dark['capture_spread_mv']:.3f} mV between captures. Means at each "
            "exposure share captures, so",
            "  proportionality to exposure is by construction and this spread is the real test.",
            f"  Per-sample noise is {dark['fast_std_code']:.2f} codes with the detector on, "
            f"against {read['std_code']:.2f} unplugged.",
        ]
        if pulses["resolved"]:
            lines.append(
                f"  Counted {pulses['rate_per_s']:.4g} dark pulses/s, height "
                f"{pulses['height_code']:.1f} codes ({pulses['height_snr']:.1f} x noise), area "
                f"{pulses['mean_area_counts']:.3g} counts, width {pulses['fwhm_samples']} samples."
            )
            if dark["fraction_from_pulses"] is not None:
                lines.append(
                    f"  Those pulses explain {dark['fraction_from_pulses']:.0%} of the dark signal."
                )
                if dark["fraction_from_pulses"] < MIN_FRACTION_EXPLAINED:
                    lines.append(
                        "  The unexplained part is a DC offset or pulses too small to find, so "
                        "the dark signal overstates dark counts. Use the counted rate."
                    )
        lines += [f"  WARNING: {w}" for w in pulses["warnings"]]

    dyn = results["dynamic_range"]
    lines += [
        "",
        "Exercise 7: dynamic range",
        f"  Saturation code {dyn['saturation_code']:g} ({dyn['saturation_source']}).",
        f"  (saturation - dark) / read noise = {dyn['per_sample']:.0f}:1 per sample "
        f"({dyn['per_sample_db']:.1f} dB).",
    ]
    if dyn["limited_by_adc"]:
        lines.append(
            "  The scope's noise is below one ADC step, so this uses the step's own noise and "
            "is a ceiling set by 8 bits."
        )
    if dyn["per_sample_detector_on"] is not None:
        lines.append(
            f"  With the detector's own noise included, {dyn['per_sample_detector_on']:.0f}:1 "
            "per sample."
        )
    if dyn["saturation_source"] != "supplied":
        lines.append(
            "  To measure saturation, push the trace off the top of the screen once and pass "
            "the code you see as --saturation-code."
        )
    if not dyn["dark_included"]:
        lines.append("  Dark counts are not included, as the dark step is not recorded yet.")

    lines.append("")
    if not results["source"]:
        lines.append("Exercise 8: not recorded yet (needs the dark and source steps)")
    for label, source in results["source"].items():
        lines += [
            f"Exercise 8: variance against mean ({label})",
            "  Mean and variance below are source minus dark.",
            f"  {'Exposure (s)':<14}{'Mean':<12}{'Variance':<12}{'Dark var':<12}Var/mean",
        ]
        lines += [
            f"  {r['exposure_s']:<14g}{r['delta_mean']:<12.4g}{r['delta_var']:<12.4g}"
            f"{r['dark_var']:<12.4g}{r['ratio']:.3g} +/- {r['ratio_se']:.2g}"
            for r in source["windows"]
        ]
        lines += [
            f"  Gain {source['gain_counts_per_event']:.3g} +/- {source['gain_se']:.2g} counts per "
            f"event at {source['sample_rate']:g} Sa/s, or {source['gain_v_s']:.3g} V.s.",
            f"  Intercept {source['intercept']:.3g} +/- {source['intercept_se']:.2g}. It should be "
            "near 0, not the read noise: dark",
            "  subtraction removes read noise, which grows with exposure here. "
            f"Without it the slope is {source['raw_slope']:.3g}.",
        ]
        if source["photons_per_v_s"] is not None:
            lines += [
                f"  Photon conversion, from {source['conversion_source']}. It holds only at this "
                "V/div's scaling and this bias step:",
                f"    {source['photons_per_v_s']:.4g} photons per V.s",
            ]
            lines += [
                f"    {per_count:.4g} photons per count at {rate:g} Sa/s"
                for rate, per_count in source["photons_per_count_by_rate"].items()
            ]
        if source["events_per_s_pulses"] is not None:
            lines.append(
                f"  Source rate {source['events_per_s_pulses']:.4g} photons/s from the fast set, "
                f"{source['events_per_s_slow']:.4g} from the slow set. These should agree."
            )
            lines.append(
                f"  Excess noise factor {source['excess_noise_factor']:.2f}, from crosstalk. "
                "Variance slope / pulse prediction "
                f"= {source['gain_ratio']:.2f}, which should be near 1."
            )
        if source["events_per_s_variance"] is not None:
            lines.append(
                f"  From the variance alone: {source['events_per_s_variance']:.4g} photons/s. "
                "Crosstalk makes this read low by the excess noise factor."
            )
        lines.append(
            "  These are photons the SiPM detected. Incident photons also need its efficiency."
        )
        lines += [f"  WARNING: {w}" for w in source["warnings"]]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="session folder from sipm_acquire.py")
    parser.add_argument(
        "--min-window", type=float, default=DEFAULT_MIN_WINDOW_S,
        help=f"shortest exposure in the gain fit, in seconds (default: {DEFAULT_MIN_WINDOW_S:g})",
    )
    parser.add_argument(
        "--saturation-code", type=float,
        help="code seen when the signal saturates, if measured (default: the ADC limit)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = analyse_session(args.session, args.min_window, args.saturation_code)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(format_report(results))
    print(f"\nSaved to {args.session / 'analysis'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

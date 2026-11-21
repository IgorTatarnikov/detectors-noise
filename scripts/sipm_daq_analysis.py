"""Noise practical calculations for a SiPM read by the USB-201 DAQ at up to 100 kS/s.

Each DAQ sample is the instantaneous output voltage. SiPM pulses are much shorter than
the 10 us between samples, so a sample either lands on a pulse or doesn't. For
randomly arriving pulses of shape h(t) at rate r (Campbell's theorem):

    mean(v - baseline) = r * integral(h)
    var(v)             = r * integral(h^2) + read variance

so variance against mean is a line, exactly as for the camera, with
    slope     = H_eff = integral(h^2) / integral(h)   the effective single-photon pulse
                height in mV (H for a square pulse, H/2 for an exponential decay)
    intercept = the read noise variance of one sample.

The slope is a gain in mV per photon, at the peak. Turning a mean voltage into a
photon rate also needs the pulse width: r = mean / (H_eff * tau_eff), with
tau_eff = integral(h)^2 / integral(h^2) from a scope capture.

The USB-201's code is about 4.9 mV. When pulses are only one or two codes tall, rounding
to whole codes biases both mean and variance, so single_code_fraction is reported as a
warning. Means and variances are taken within short blocks (default 10 ms), so slow
drift and mains pickup don't inflate the variance.

All voltages here are in mV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ADC_MAX_CODE = 4095
LSB_MV = 20_000 / 4096
DEFAULT_BLOCK_S = 0.01
# Above this fraction of hits that are only one code tall, the ADC step dominates.
MAX_SINGLE_CODE_FRACTION = 0.5


@dataclass(frozen=True)
class LineFit:
    slope: float
    intercept: float


@dataclass(frozen=True)
class BlockStats:
    mean: float  # mean of the block means
    var: float  # mean of the within-block variances
    mean_spread: float  # std of the block means, the stability between blocks
    blocks: int


@dataclass(frozen=True)
class ReadNoise:
    baseline_mv: float
    baseline_code: float
    sd_mv: float  # measured, within blocks
    effective_sd_mv: float  # never below the rounding noise of the ADC
    adc_limited: bool
    clipped_fraction: float


@dataclass(frozen=True)
class SignalPoint:
    mean: float  # baseline-subtracted mean, mV
    var: float  # mV^2
    mean_spread: float
    above_fraction: float  # samples at least one code above the baseline code
    single_code_fraction: float  # of those, how many are exactly one code above
    clipped_fraction: float
    max_mv: float

    @property
    def adc_limited(self) -> bool:
        return self.above_fraction > 0 and self.single_code_fraction > MAX_SINGLE_CODE_FRACTION


def block_samples(rate: float, block_s: float = DEFAULT_BLOCK_S) -> int:
    return max(2, int(round(rate * block_s)))


def block_stats(mv: np.ndarray, block: int) -> BlockStats:
    mv = np.asarray(mv, dtype=np.float64)
    n = mv.size // block * block
    if n < 2 * block:
        x = mv[None, :]
    else:
        x = mv[:n].reshape(-1, block)
    means = x.mean(axis=1)
    return BlockStats(
        mean=float(means.mean()),
        var=float(x.var(axis=1, ddof=1).mean()),
        mean_spread=float(means.std(ddof=1)) if means.size > 1 else math.nan,
        blocks=int(means.size),
    )


def clipped_fraction(codes: np.ndarray) -> float:
    codes = np.asarray(codes)
    return float(np.mean((codes <= 0) | (codes >= ADC_MAX_CODE)))


def read_noise(mv: np.ndarray, codes: np.ndarray, block: int, lsb_mv: float = LSB_MV) -> ReadNoise:
    """Read noise section: detector unplugged, the spread of single samples about the baseline."""
    stats = block_stats(mv, block)
    sd = math.sqrt(stats.var)
    floor = lsb_mv / math.sqrt(12)
    values, counts = np.unique(np.asarray(codes), return_counts=True)
    return ReadNoise(
        baseline_mv=stats.mean,
        baseline_code=float(values[np.argmax(counts)]),
        sd_mv=sd,
        effective_sd_mv=max(sd, floor),
        adc_limited=sd < floor,
        clipped_fraction=clipped_fraction(codes),
    )


def hit_fractions(codes: np.ndarray, baseline_code: float) -> tuple[float, float]:
    """(fraction of samples above the baseline code, fraction of those exactly one code above)."""
    excess = np.round(np.asarray(codes, dtype=np.float64) - baseline_code).astype(np.int64)
    hits = excess[excess > 0]
    if hits.size == 0:
        return 0.0, 0.0
    return float(hits.size / excess.size), float(np.mean(hits == 1))


def signal_point(mv: np.ndarray, codes: np.ndarray, baseline_mv: float, baseline_code: float, block: int) -> SignalPoint:
    """Mean and variance of one recording (dark, or a light level), for the gain fit."""
    stats = block_stats(np.asarray(mv, dtype=np.float64) - baseline_mv, block)
    above, single = hit_fractions(codes, baseline_code)
    return SignalPoint(
        mean=stats.mean,
        var=stats.var,
        mean_spread=stats.mean_spread,
        above_fraction=above,
        single_code_fraction=single,
        clipped_fraction=clipped_fraction(codes),
        max_mv=float(np.max(mv) - baseline_mv),
    )


def fit_line(x, y) -> LineFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return LineFit(math.nan, math.nan)
    slope, intercept = np.polyfit(x, y, 1)
    return LineFit(float(slope), float(intercept))


def gain_fit(points: list[SignalPoint]) -> LineFit:
    """Shot noise section: variance against mean over light levels at one bias step.

    The slope is H_eff, the effective single-photon pulse height in mV.
    """
    return fit_line([p.mean for p in points], [p.var for p in points])


def pulse_rate(mean_mv: float, h_eff_mv: float, pulse_width_s: float) -> float:
    """Pulses per second from a mean excess voltage, given tau_eff from a scope capture."""
    return mean_mv / (h_eff_mv * pulse_width_s)


def dynamic_range(saturation_mv: float, baseline_mv: float, dark_mv: float, noise_mv: float) -> tuple[float, float]:
    """(saturation - dark) / read noise, with the baseline removed. Returns (ratio, dB)."""
    ratio = (saturation_mv - baseline_mv - dark_mv) / noise_mv
    return float(ratio), float(20 * math.log10(ratio)) if ratio > 0 else math.nan


def saturation_level(mv: np.ndarray) -> float:
    """The ceiling the output sits at under bright light, ignoring the odd spike."""
    return float(np.percentile(mv, 99.9))


def window_integrals(excess_mv: np.ndarray, rate: float, windows_s) -> list[tuple[float, float, float]]:
    """The software equivalent of exposure time: sum the signal over windows of each length.

    Returns (window_s, mean integral, std between windows), integrals in mV*s.
    """
    excess_mv = np.asarray(excess_mv, dtype=np.float64)
    out = []
    for t in windows_s:
        n = max(1, int(round(t * rate)))
        k = excess_mv.size // n
        if k < 2:
            continue
        sums = excess_mv[: k * n].reshape(k, n).sum(axis=1) / rate
        out.append((n / rate, float(sums.mean()), float(sums.std(ddof=1))))
    return out

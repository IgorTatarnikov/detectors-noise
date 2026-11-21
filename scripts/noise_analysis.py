"""Calculations for each section of the camera noise practical.

Plain numpy, no Qt and no camera, so basler_live.py, the tests and anyone at a
prompt get the same numbers. All values are in the camera's raw counts (12-bit,
0-4095 for Mono12) unless a name says otherwise.

    stack: array of shape (frames, rows, cols)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Above this fraction of pixels at 0, the standard deviation is too low to trust.
MAX_CLIPPED_FRACTION = 1e-3


@dataclass(frozen=True)
class ReadNoise:
    read_noise: float  # mean over pixels of each pixel's std across the stack
    bias: float  # mean of the whole stack
    frac_zero: float  # fraction of values at 0, where the negative side is clipped
    frac_saturated: float

    @property
    def clipped(self) -> bool:
        return self.frac_zero > MAX_CLIPPED_FRACTION


@dataclass(frozen=True)
class LineFit:
    slope: float
    intercept: float


@dataclass(frozen=True)
class DarkFit:
    exposures_s: np.ndarray
    means: np.ndarray  # mean counts per exposure, bias included
    bias: float  # mean at the shortest exposure
    subtracted: np.ndarray  # means - bias
    slope: float  # counts/pixel/s
    intercept: float


@dataclass(frozen=True)
class ShotNoise:
    mean: np.ndarray  # per pixel, dark subtracted
    var: np.ndarray  # per pixel
    frame_mean_cv: float  # frame-to-frame variation of the illumination, as a fraction
    fit: LineFit  # slope is the gain in counts/e-


def per_pixel_std(stack: np.ndarray) -> np.ndarray:
    """Standard deviation of each pixel across the stack."""
    return np.std(np.asarray(stack, dtype=np.float32), axis=0, ddof=1, dtype=np.float64)


def read_noise(stack: np.ndarray, max_value: int = 4095) -> ReadNoise:
    """Read noise section: average per-pixel std of a dark stack at minimum exposure."""
    stack = np.asarray(stack)
    return ReadNoise(
        read_noise=float(per_pixel_std(stack).mean()),
        bias=float(stack.mean(dtype=np.float64)),
        frac_zero=float(np.mean(stack == 0)),
        frac_saturated=float(np.mean(stack >= max_value)),
    )


def frame_means(stack: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(stack), axis=(1, 2), dtype=np.float64)


def fit_line(x, y) -> LineFit:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    slope, intercept = np.polyfit(x, y, 1)
    return LineFit(float(slope), float(intercept))


def dark_counts(exposures_s, means) -> DarkFit:
    """Dark counts section: bias-subtracted mean counts against exposure.

    The bias is the mean at the shortest exposure, where dark counts are negligible.
    """
    exposures_s = np.asarray(exposures_s, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    order = np.argsort(exposures_s)
    exposures_s, means = exposures_s[order], means[order]
    bias = float(means[0])
    subtracted = means - bias
    fit = fit_line(exposures_s, subtracted) if exposures_s.size >= 2 else LineFit(math.nan, math.nan)
    return DarkFit(exposures_s, means, bias, subtracted, fit.slope, fit.intercept)


def dynamic_range(saturation: float, bias: float, dark: float, read_noise: float) -> tuple[float, float]:
    """(saturation - dark) / read noise, with the bias removed from saturation.

    `dark` is the bias-subtracted dark count at the exposure of interest.
    Returns (ratio, decibels).
    """
    ratio = (saturation - bias - dark) / read_noise
    return float(ratio), float(20 * math.log10(ratio)) if ratio > 0 else math.nan


def shot_noise(stack: np.ndarray, dark_stack: np.ndarray, normalise: bool = True) -> ShotNoise:
    """Shot noise section: per-pixel variance against mean, minus a dark stack at the same settings.

    The dark stack's per-pixel mean removes the bias, fixed pattern and dark counts.
    With normalise, each frame is rescaled to the stack's mean brightness, so slow
    changes in the light are not counted as shot noise.
    """
    frames = np.asarray(stack, dtype=np.float32) - np.mean(dark_stack, axis=0, dtype=np.float64).astype(np.float32)
    means = frame_means(frames)
    cv = float(np.std(means, ddof=1) / np.mean(means)) if means.size > 1 and np.mean(means) else math.nan
    if normalise and np.all(means > 0):
        frames = frames / (means / means.mean()).astype(np.float32)[:, None, None]
    mean = np.mean(frames, axis=0, dtype=np.float64).ravel()
    var = np.var(frames, axis=0, ddof=1, dtype=np.float64).ravel()
    return ShotNoise(mean, var, cv, fit_line(mean, var))


def pooled_fit(points: list[tuple[np.ndarray, np.ndarray]]) -> LineFit:
    """Fit variance against mean over several shot-noise recordings together."""
    return fit_line(np.concatenate([m for m, _ in points]), np.concatenate([v for _, v in points]))


def check_block_ids(block_ids) -> tuple[int, int]:
    """(missing, repeated) frames, from the camera's frame counter."""
    ids = np.asarray(block_ids, dtype=np.int64)
    if ids.size < 2:
        return 0, 0
    step = np.diff(ids)
    return int(np.sum(np.clip(step - 1, 0, None))), int(np.sum(step <= 0))

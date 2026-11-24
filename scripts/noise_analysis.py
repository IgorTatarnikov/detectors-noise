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
# Shot noise is fitted below this fraction of the mean where the variance peaks
# (saturation), as in EMVA 1288, so a pixel whose well is filling doesn't bend the line.
FIT_FRACTION = 0.7
# In a covered stack, a frame-to-frame step in the frame mean larger than this
# (counts) is an offset jump, not noise or dark counts.
MAX_OFFSET_STEP = 0.5


@dataclass(frozen=True)
class ReadNoise:
    read_noise: float  # mean over pixels of each pixel's std across the stack
    bias: float  # mean of the stack
    frac_zero: float  # fraction of values at 0, where the negative side is clipped
    frac_saturated: float
    frames: int  # frames used: the longest run without an offset jump
    pixel_std: np.ndarray  # each pixel's std across those frames

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
    mean: np.ndarray  # per unclipped pixel, bias subtracted
    var: np.ndarray  # per unclipped pixel
    frame_mean_cv: float  # frame-to-frame variation of the illumination, as a fraction
    fit: LineFit  # slope is the gain in counts/e-
    frac_saturated: float  # pixels that reach the largest value in any frame
    frac_zero: float  # pixels that read 0 in any frame
    saturation_mean: float  # mean (minus bias) where the variance peaks
    fit_max: float  # pixels with a mean below this are fitted

    @property
    def pixels(self) -> int:
        """Pixels not clipped in any frame."""
        return int(self.mean.size)

    @property
    def fitted(self) -> int:
        return int(np.count_nonzero(self.mean < self.fit_max))


def per_pixel_std(stack: np.ndarray) -> np.ndarray:
    """Standard deviation of each pixel across the stack."""
    return np.std(np.asarray(stack, dtype=np.float32), axis=0, ddof=1, dtype=np.float64)


def read_noise(stack: np.ndarray, max_value: int = 4095) -> ReadNoise:
    """Read noise section: average per-pixel std of a dark stack at minimum exposure.

    If the camera's offset jumps during the stack, only the longest steady run of frames is
    used. A jump must also be too large to be noise in the frame means, which matters only for
    small images: the std of one frame's difference from the next, over sqrt(pixels), is the
    std of the difference between their means, whatever the offset does.
    """
    stack = np.asarray(stack)
    max_step = MAX_OFFSET_STEP
    if stack.shape[0] > 1:
        difference = stack[1].astype(np.float32) - stack[0].astype(np.float32)
        max_step = max(max_step, 6 * float(difference.std(dtype=np.float64)) / math.sqrt(difference.size))
    steady = steady_frames(frame_means(stack), max_step)
    stack = stack[steady[0] : steady[-1] + 1]
    pixel_std = per_pixel_std(stack)
    return ReadNoise(
        read_noise=float(pixel_std.mean()),
        bias=float(stack.mean(dtype=np.float64)),
        frac_zero=float(np.mean(stack == 0)),
        frac_saturated=float(np.mean(stack >= max_value)),
        frames=int(stack.shape[0]),
        pixel_std=pixel_std,
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


def shot_noise(stack: np.ndarray, bias: float, max_value: int = 4095, normalise: bool = True) -> ShotNoise:
    """Shot noise section: per-pixel variance against mean, from one unevenly lit stack.

    The light falls off across the sensor, so the pixels span every brightness from dark
    to saturated in one recording. A pixel that reads 0 or the largest value in any frame
    is clipped, so its variance is too low and it is left out. A pixel whose well is
    nearly full stops gaining variance before it reaches the largest value, so the line is
    fitted only below FIT_FRACTION of the mean where the variance peaks. `bias` is the mean
    of a covered stack at the same gain and black level (the Read noise section). With
    normalise, each frame is rescaled to the stack's mean brightness, so slow changes in
    the light are not counted as shot noise.
    """
    stack = np.asarray(stack)
    flat = stack.reshape(stack.shape[0], -1)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    usable = (lo > 0) & (hi < max_value)
    frames = flat[:, usable].astype(np.float32) - np.float32(bias)
    means = np.mean(frames, axis=1, dtype=np.float64) if usable.any() else np.zeros(stack.shape[0])
    cv = float(np.std(means, ddof=1) / np.mean(means)) if means.size > 1 and np.mean(means) > 0 else math.nan
    if normalise and np.all(means > 0):
        frames = frames / (means / means.mean()).astype(np.float32)[:, None]
    mean = np.mean(frames, axis=0, dtype=np.float64)
    var = np.var(frames, axis=0, ddof=1, dtype=np.float64)
    saturation = saturation_mean(mean, var)
    fit_max = FIT_FRACTION * saturation
    below = mean < fit_max
    fit = fit_line(mean[below], var[below]) if np.count_nonzero(below) >= 2 else LineFit(math.nan, math.nan)
    return ShotNoise(mean, var, cv, fit, float(np.mean(hi >= max_value)), float(np.mean(lo == 0)), saturation, fit_max)


def saturation_mean(mean: np.ndarray, var: np.ndarray, bins: int = 50, min_pixels: int = 10) -> float:
    """The mean where the binned variance peaks: beyond it, the pixels are saturating."""
    bin_mean, bin_var = binned(mean, var, bins, min_pixels)
    if bin_mean.size == 0:
        return float(mean.max()) if mean.size else math.nan
    return float(bin_mean[np.argmax(bin_var)])


def binned(mean: np.ndarray, var: np.ndarray, bins: int = 60, min_pixels: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Average variance in equal-width bins of mean, for plotting many pixels as one curve."""
    if mean.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(mean.min(), mean.max(), bins + 1)
    idx = np.clip(np.digitize(mean, edges) - 1, 0, bins - 1)
    count = np.bincount(idx, minlength=bins)
    keep = count >= min_pixels
    sum_mean = np.bincount(idx, weights=mean, minlength=bins)
    sum_var = np.bincount(idx, weights=var, minlength=bins)
    return sum_mean[keep] / count[keep], sum_var[keep] / count[keep]


def steady_frames(frame_means, max_step: float = MAX_OFFSET_STEP) -> np.ndarray:
    """Indices of the longest run of frames with no offset jump between them.

    In a covered stack the frame mean should only creep with dark counts. The camera's offset
    sometimes steps by about 2 counts (seen on an acA4112-20um at 12 and 24 dB, e.g. 7 frames
    after a settings change). Averaging across the step mixes two biases, so a covered stack
    is averaged over its longest steady run instead; the later run wins a tie.
    """
    means = np.asarray(frame_means, dtype=np.float64)
    if means.size == 0:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.abs(np.diff(means)) > max_step) + 1
    runs = np.split(np.arange(means.size), breaks)
    return max(reversed(runs), key=len)


def check_block_ids(block_ids) -> tuple[int, int]:
    """(missing, repeated) frames, from the camera's frame counter."""
    ids = np.asarray(block_ids, dtype=np.int64)
    if ids.size < 2:
        return 0, 0
    step = np.diff(ids)
    return int(np.sum(np.clip(step - 1, 0, None))), int(np.sum(step <= 0))

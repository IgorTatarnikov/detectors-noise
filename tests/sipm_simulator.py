"""A simulated SiPM on a Rigol scope, so the pipeline can be checked against known truth."""

from __future__ import annotations

import numpy as np


def simulate_codes(
    samples: int,
    rate: float,
    rng: np.random.Generator,
    *,
    baseline: float = 77.0,
    noise_std: float = 1.0,
    event_rate: float = 0.0,
    height: float = 12.0,
    tau: float = 30e-9,
    burst_size: int = 1,
    burst_tau: float = 20e-6,
    crosstalk: float = 0.0,
    noise_smoothing: int = 1,
) -> np.ndarray:
    """Return 8-bit codes for Poisson events, each an exponential pulse, plus noise.

    A pulse sums to height * tau * rate counts when the sampling resolves it. With
    burst_size > 1, each event emits that many pulses spread over burst_tau, as a
    phosphor excited by one decay would. A crosstalk fraction of pulses are twice as
    large, as when one photon fires two microcells. noise_smoothing averages the noise
    over that many samples at the same std, as a limited bandwidth correlates it.
    """
    duration = samples / rate
    events = rng.poisson(event_rate * duration)
    times = rng.uniform(0, duration, events)
    if burst_size > 1:
        times = np.repeat(times, burst_size) + rng.exponential(burst_tau, events * burst_size)

    heights = height * (1 + (rng.random(times.size) < crosstalk))

    signal = np.zeros(samples)
    first = np.ceil(times * rate).astype(int)
    for offset in range(max(1, int(np.ceil(8 * tau * rate)))):
        index = first + offset
        keep = index < samples
        np.add.at(signal, index[keep], heights[keep] * np.exp(-(index[keep] / rate - times[keep]) / tau))

    noise = rng.normal(0, noise_std * np.sqrt(noise_smoothing), samples)
    noise = np.convolve(noise, np.ones(noise_smoothing) / noise_smoothing, mode="same")
    noisy = baseline + signal + noise
    return np.clip(np.round(noisy), 0, 255)

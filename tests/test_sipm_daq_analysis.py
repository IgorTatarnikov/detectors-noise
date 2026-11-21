"""The per-sample DAQ analysis recovers the simulated SiPM's pulse height, rates and noise."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sipm_daq_analysis as sa  # noqa: E402
from daq_simulator import SimulatedDAQ, SipmModel, mv_to_codes, simulate_mv  # noqa: E402
from usb201 import codes_to_volts  # noqa: E402

RATE = 100_000
N = 400_000
BLOCK = sa.block_samples(RATE)


def record(pulse_rate, height_mv, seed, **kwargs):
    rng = np.random.default_rng(seed)
    codes = mv_to_codes(simulate_mv(N, RATE, pulse_rate, height_mv, rng, **kwargs))
    return codes_to_volts(codes, 1.0, 0.0) * 1e3, codes


def test_codes_round_trip():
    mv = np.array([-1000.0, 0.0, 3.9, 512.3, 9000.0])
    back = codes_to_volts(mv_to_codes(mv), 1.0, 0.0) * 1e3
    np.testing.assert_allclose(back, mv, atol=sa.LSB_MV / 2)


def test_campbell_mean_and_variance_without_quantisation():
    rng = np.random.default_rng(0)
    mv = simulate_mv(N, RATE, 2e5, 500.0, rng, tau_s=50e-9, baseline_mv=0, read_noise_mv=0)
    assert mv.mean() == pytest.approx(2e5 * 500 * 50e-9, rel=0.03)
    assert mv.var() / mv.mean() == pytest.approx(250.0, rel=0.03)  # H/2 for an exponential


def test_read_noise_below_adc_step_is_flagged():
    mv, codes = record(0, 0, seed=1, read_noise_mv=0.16)
    rn = sa.read_noise(mv, codes, BLOCK)
    assert rn.adc_limited
    assert rn.effective_sd_mv == pytest.approx(sa.LSB_MV / math.sqrt(12))
    assert rn.baseline_mv == pytest.approx(3.9, abs=sa.LSB_MV / 2)
    assert rn.clipped_fraction == 0


def test_read_noise_above_adc_step_is_measured():
    mv, codes = record(0, 0, seed=2, read_noise_mv=8.0)
    rn = sa.read_noise(mv, codes, BLOCK)
    assert not rn.adc_limited
    assert rn.sd_mv == pytest.approx(math.sqrt(8.0**2 + sa.LSB_MV**2 / 12), rel=0.03)


def test_gain_fit_recovers_effective_pulse_height_over_light_levels():
    height = 400.0
    rn = sa.read_noise(*record(0, 0, seed=3), BLOCK)
    points = []
    for i, rate in enumerate([0.5e5, 1e5, 2e5, 4e5]):
        mv, codes = record(rate, height, seed=10 + i)
        points.append(sa.signal_point(mv, codes, rn.baseline_mv, rn.baseline_code, BLOCK))
    fit = sa.gain_fit(points)
    assert fit.slope == pytest.approx(height / 2, rel=0.05)
    assert not any(p.adc_limited for p in points)
    # For an exponential pulse, tau_eff = integral(h)^2 / integral(h^2) = 2 * tau.
    rate = sa.pulse_rate(points[2].mean, fit.slope, 2 * 50e-9)
    assert rate == pytest.approx(2e5, rel=0.06)


def test_small_pulses_are_flagged_as_adc_limited():
    rn = sa.read_noise(*record(0, 0, seed=4), BLOCK)
    mv, codes = record(3e5, 9.0, seed=5)
    point = sa.signal_point(mv, codes, rn.baseline_mv, rn.baseline_code, BLOCK)
    assert point.adc_limited
    assert point.single_code_fraction > 0.5


def test_hit_fractions():
    codes = np.array([10, 10, 11, 12, 10, 11])
    above, single = sa.hit_fractions(codes, 10)
    assert above == pytest.approx(0.5)
    assert single == pytest.approx(2 / 3)


def test_block_stats_ignores_slow_drift():
    rng = np.random.default_rng(6)
    drift = np.linspace(0, 50, N)
    noise = rng.normal(0, 2, N)
    stats = sa.block_stats(drift + noise, BLOCK)
    assert stats.var == pytest.approx(4, rel=0.03)
    assert np.var(drift + noise) > 50


def test_window_integrals_scale_with_window():
    excess = np.full(N, 2.0)  # 2 mV for every sample
    rows = sa.window_integrals(excess, RATE, [1e-3, 1e-2, 0.1])
    for t, mean, sd in rows:
        assert mean == pytest.approx(2.0 * t)
        assert sd == pytest.approx(0, abs=1e-12)


def test_dynamic_range_and_saturation():
    ratio, db = sa.dynamic_range(3604.0, 4.0, 0.0, 3.6)
    assert ratio == pytest.approx(1000)
    assert db == pytest.approx(60)
    mv = np.concatenate([np.full(10_000, 3600.0), [9000.0]])
    assert sa.saturation_level(mv) == pytest.approx(3600.0)


def test_simulated_daq_follows_the_bias_knob():
    daq = SimulatedDAQ(seed=7, realtime=False)
    daq.start_continuous(7, RATE)
    daq.bias_step = 2
    low = daq.read_available(40_000)
    daq.bias_step = 8
    high = daq.read_available(40_000)
    assert high.mean() > low.mean()
    daq.connected = False
    assert daq.read_available(40_000).std() < 0.5
    model = SipmModel()
    assert model.h_eff_mv(8) > model.h_eff_mv(5)

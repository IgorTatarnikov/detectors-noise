"""Tests for the SiPM analysis, against hand-worked values and simulated detectors."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sipm_analyse import (  # noqa: E402
    analyse_dynamic_range,
    dynamic_range,
    fit_line,
    gain_from_variance,
    measure_pulses,
    measure_read_noise,
    window_sizes,
    window_stats,
)
from sipm_simulator import simulate_codes  # noqa: E402

RATE = 1e8
SAMPLES = 120_000
# height * tau * rate for the simulator's default pulse.
PULSE_COUNTS = 36.0


def captures(seed, n=20, **kwargs):
    rng = np.random.default_rng(seed)
    return [simulate_codes(SAMPLES, RATE, rng, **kwargs) for _ in range(n)]


def assert_gain_matches(result, expected):
    """The fit must agree within its own error, and that error must be useful."""
    assert abs(result["gain_counts_per_event"] - expected) < 3 * result["gain_se"]
    assert result["gain_se"] < 0.25 * expected


def test_window_stats_sums_each_window_above_the_baseline():
    stats = window_stats([np.array([78.0, 80.0, 76.0, 78.0])], samples=2, baseline=77.0)
    # Windows are (1 + 3) = 4 and (-1 + 1) = 0.
    assert stats.mean == pytest.approx(2.0)
    assert stats.var == pytest.approx(8.0)  # (4 - 0)^2 / 2
    assert stats.pairs == 1


def test_window_stats_drops_samples_that_do_not_fill_a_window():
    stats = window_stats([np.array([78.0, 78.0, 78.0, 78.0, 200.0])], samples=2, baseline=77.0)
    assert stats.mean == pytest.approx(2.0)


def test_window_variance_ignores_offsets_between_captures():
    # Windows are [0, 2] in one capture and [100, 102] in the next, like a slow drift.
    first = np.array([77.0, 77.0, 78.0, 78.0])
    second = np.array([127.0, 127.0, 128.0, 128.0])
    stats = window_stats([first, second], samples=2, baseline=77.0)
    assert stats.var == pytest.approx(2.0)  # Each difference is 2, so 2^2 / 2.
    assert stats.plain_var > 1000
    assert stats.pairs == 2


def test_window_stats_has_no_variance_with_one_window_per_capture():
    stats = window_stats([np.array([78.0, 78.0]), np.array([79.0, 79.0])], samples=2, baseline=77.0)
    assert stats.mean == pytest.approx(3.0)
    assert np.isnan(stats.var)
    assert stats.pairs == 0


def test_window_sizes_follow_one_two_five_and_leave_enough_pairs():
    # 120k samples at 100 MSa/s across 20 captures. A 500 us window gives two per
    # capture, so 20 pairs. A 1 ms window would give none.
    sizes = window_sizes(SAMPLES, RATE, captures=20, min_window_s=10e-6, min_pairs=16)
    assert sizes == [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]


def test_fit_line_recovers_a_line_and_its_slope_error():
    fit = fit_line(np.array([1.0, 2.0, 3.0]), np.array([3.0, 5.0, 7.0]), np.array([1.0, 1.0, 1.0]))
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.slope_se == pytest.approx(1 / np.sqrt(2))
    assert fit.reduced_chi2 == pytest.approx(0.0)


def test_fit_line_gives_little_weight_to_an_uncertain_point():
    fit = fit_line(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.0, 4.0, 6.0, 100.0]),
        np.array([1.0, 1.0, 1.0, 1e6]),
    )
    assert fit.slope == pytest.approx(2.0, rel=1e-3)


def test_read_noise_measures_the_baseline_and_per_sample_spread():
    result = measure_read_noise(captures(1, noise_std=2.5), yinc=0.002)
    assert result["baseline_code"] == pytest.approx(77.0, abs=0.02)
    assert result["std_code"] == pytest.approx(2.5, rel=0.02)
    assert result["std_mv"] == pytest.approx(5.0, rel=0.02)
    assert result["clipped_fraction"] == 0.0


def test_read_noise_warns_when_the_adc_step_hides_the_noise():
    result = measure_read_noise(captures(2, noise_std=0.1), yinc=0.05)
    assert any("V/div" in warning for warning in result["warnings"])


def test_read_noise_warns_about_clipping():
    result = measure_read_noise(captures(3, baseline=2.0, noise_std=2.5), yinc=0.002)
    assert result["clipped_fraction"] > 0
    assert any("clip" in warning.lower() for warning in result["warnings"])


def test_gain_comes_from_shot_noise_not_read_noise():
    # Read noise large enough that it would dominate an uncorrected fit.
    dark = captures(4, noise_std=8.0, event_rate=2e4)
    source = captures(5, noise_std=8.0, event_rate=3e6)
    result = gain_from_variance(source, dark, baseline=77.0, sample_rate=RATE, min_window_s=10e-6)
    assert_gain_matches(result, PULSE_COUNTS)
    assert result["raw_slope"] > 2 * PULSE_COUNTS


def test_gain_fit_passes_near_the_origin_after_subtracting_dark():
    dark = captures(6, event_rate=2e4)
    source = captures(7, event_rate=3e5)
    result = gain_from_variance(source, dark, baseline=77.0, sample_rate=RATE, min_window_s=10e-6)
    assert abs(result["intercept"]) < 3 * result["intercept_se"] + 1e-9


def test_gain_is_not_changed_by_a_constant_amplifier_offset():
    dark = captures(8, baseline=80.0, event_rate=2e4)
    source = captures(9, baseline=80.0, event_rate=3e5)
    result = gain_from_variance(source, dark, baseline=77.0, sample_rate=RATE, min_window_s=10e-6)
    assert_gain_matches(result, PULSE_COUNTS)


def test_gain_slope_includes_every_photon_in_a_burst():
    dark = captures(10, event_rate=2e4)
    source = captures(11, event_rate=1e5, burst_size=5, burst_tau=2e-6)
    result = gain_from_variance(source, dark, baseline=77.0, sample_rate=RATE, min_window_s=10e-6)
    assert result["gain_counts_per_event"] > 2.5 * PULSE_COUNTS


def test_pulses_recover_the_rate_and_area_of_single_events():
    result = measure_pulses(captures(12, event_rate=2e5), RATE)
    assert result["rate_per_s"] == pytest.approx(2e5, rel=0.1)
    assert result["mean_area_counts"] == pytest.approx(PULSE_COUNTS, rel=0.15)
    assert result["effective_gain_counts"] == pytest.approx(PULSE_COUNTS, rel=0.15)


def test_pulses_report_the_quiet_level_rather_than_the_scope_baseline():
    result = measure_pulses(captures(13, baseline=80.0, event_rate=2e5), RATE)
    assert result["quiet_code"] == pytest.approx(80.0, abs=0.1)
    assert result["mean_area_counts"] == pytest.approx(PULSE_COUNTS, rel=0.15)


def test_pulse_prediction_is_not_biased_by_correlated_noise():
    result = measure_pulses(captures(16, event_rate=2e5, noise_smoothing=4), RATE)
    assert result["effective_gain_counts"] == pytest.approx(PULSE_COUNTS, rel=0.15)


def test_pulse_prediction_is_not_biased_by_drift_between_captures():
    rng = np.random.default_rng(17)
    drifting = [
        simulate_codes(SAMPLES, RATE, rng, event_rate=2e5, baseline=77.0 + (i % 2))
        for i in range(20)
    ]
    assert measure_pulses(drifting, RATE)["effective_gain_counts"] == pytest.approx(
        PULSE_COUNTS, rel=0.15
    )


def test_pulses_measure_excess_noise_from_crosstalk():
    # With 30% of pulses doubled: E[a] = 1.3 A and E[a^2] = 1.9 A^2, so the excess
    # noise factor E[a^2] / E[a]^2 is 1.9 / 1.69 = 1.124.
    rng = np.random.default_rng(18)
    tall = [
        simulate_codes(SAMPLES, RATE, rng, event_rate=2e5, height=30.0, crosstalk=0.3)
        for _ in range(20)
    ]
    result = measure_pulses(tall, RATE)
    assert result["mean_area_counts"] == pytest.approx(1.3 * 90, rel=0.1)
    assert result["excess_noise_factor"] == pytest.approx(1.124, abs=0.06)


def bench_like(seed):
    """Pulses about 5 times the noise, as measured on the PDA4x at 500 MSa/s."""
    rng = np.random.default_rng(seed)
    return [
        simulate_codes(
            1_200_000, 5e8, rng, baseline=79.0, noise_std=4.1, noise_smoothing=3,
            event_rate=2e5, height=24.0, tau=3e-9,
        )
        for _ in range(12)
    ]


def test_pulses_barely_above_the_noise_are_still_sized_without_a_threshold_bias():
    # 24 codes * 3 ns * 500 MSa/s = 36 counts. Selecting peaks above 5 sigma, as a
    # plain threshold must to avoid noise, gave areas that rose with the threshold.
    result = measure_pulses(bench_like(19), 5e8)
    assert result["resolved"] is True
    assert result["mean_area_counts"] == pytest.approx(36.0, rel=0.2)
    assert result["rate_per_s"] == pytest.approx(2e5, rel=0.35)


def test_pulses_barely_above_the_noise_come_with_a_warning():
    result = measure_pulses(bench_like(20), 5e8)
    assert result["height_snr"] < 6
    assert any("bias" in warning for warning in result["warnings"])


def test_clearly_separated_pulses_do_not_warn_about_the_noise():
    result = measure_pulses(captures(21, event_rate=2e5), RATE)
    assert not any("bias" in warning for warning in result["warnings"])


def test_negative_pulses_are_not_counted_as_pulses():
    inverted = [154.0 - codes for codes in captures(22, event_rate=2e5)]
    result = measure_pulses(inverted, RATE)
    assert result["resolved"] is False
    assert any("polarity" in warning for warning in result["warnings"])


def test_pulses_report_none_found_in_pure_noise():
    result = measure_pulses(captures(14, noise_std=2.0), RATE)
    assert result["resolved"] is False
    assert any("polarity" in warning for warning in result["warnings"])


def test_pulses_warn_when_the_sampling_cannot_resolve_them():
    rng = np.random.default_rng(15)
    # A 30 ns pulse sampled every 1 us shows up in one sample at most.
    slow = [simulate_codes(1_000_000, 1e6, rng, event_rate=2e5, height=40.0) for _ in range(2)]
    result = measure_pulses(slow, 1e6)
    assert any("sample rate" in warning for warning in result["warnings"])


def test_dynamic_range_divides_the_usable_range_by_the_noise():
    # (255 - 77 - 3) / 2.5
    assert dynamic_range(255, baseline=77.0, dark=3.0, noise=2.5) == pytest.approx(70.0)


def test_dynamic_range_cannot_exceed_what_the_adc_step_allows():
    # Noise far below one code can't be measured, so the step's own 1 / sqrt(12) codes
    # is the floor: (255 - 104) * sqrt(12) = 523, not 151 / 0.008 = 18875.
    result = analyse_dynamic_range({"baseline_code": 104.0, "std_code": 0.008}, None, None)
    assert result["per_sample"] == pytest.approx(523.0, rel=0.01)
    assert result["limited_by_adc"] is True


def test_dynamic_range_uses_measured_noise_when_the_adc_can_resolve_it():
    # 151 / 2.0
    result = analyse_dynamic_range({"baseline_code": 104.0, "std_code": 2.0}, None, None)
    assert result["per_sample"] == pytest.approx(75.5)
    assert result["limited_by_adc"] is False

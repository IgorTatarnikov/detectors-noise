"""Tests for recording a session, and the whole pipeline against a simulated bench."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sipm_acquire import CaptureSet, parse_args, run_step  # noqa: E402
from sipm_analyse import analyse_session, format_report  # noqa: E402
from sipm_exposure_sweep import Preamble, Scope, SettingsError  # noqa: E402
from sipm_simulator import simulate_codes  # noqa: E402

FAST = CaptureSet("fast", sample_rate=5e7, capture_s=2.4e-3, repeats=20)
SLOW = CaptureSet("slow", sample_rate=1e6, capture_s=0.12, repeats=5)
SETS = [FAST, SLOW]
YINC = 0.002
# height * tau * rate = 30 codes * 60 ns * 50 MSa/s, well above the noise.
PULSE_COUNTS = 90.0

BENCH = {
    "unplugged": {"noise_std": 2.0, "event_rate": 0.0},
    "dark": {"noise_std": 2.5, "event_rate": 2e5, "height": 30.0, "tau": 60e-9},
    "vial": {"noise_std": 2.5, "event_rate": 3e6, "height": 30.0, "tau": 60e-9},
}


class BenchScope(Scope):
    """The real Scope logic on a simulated detector, whose state the prompts change."""

    def __init__(self, fail_after=None, scale="0.05"):
        self.state = "unplugged"
        self.rng = np.random.default_rng(42)
        self.fail_after = fail_after
        self.measured = 0
        self.restored = 0
        self.scale = scale

    def query(self, command):
        replies = {
            "*IDN?": "RIGOL TECHNOLOGIES,DS1074Z,SIM,1",
            ":CHAN1:DISP?": "1",
            ":CHAN2:DISP?": "0",
            ":CHAN3:DISP?": "0",
            ":CHAN4:DISP?": "0",
            ":CHAN1:SCAL?": self.scale,
            ":CHAN1:OFFS?": "-0.097",
            ":CHAN1:PROB?": "1",
            ":CHAN1:COUP?": "DC",
            ":CHAN1:BWL?": "OFF",
        }
        return replies[command]

    def save_settings(self):
        return {}

    def restore_settings(self, settings):
        self.restored += 1
        return []

    def prepare(self, plan):
        pass

    def measure(self, plan):
        if self.fail_after is not None and self.measured >= self.fail_after:
            raise SettingsError("simulated failure")
        self.measured += 1
        codes = simulate_codes(plan.samples, plan.sample_rate, self.rng, **BENCH[self.state])
        preamble = Preamble(0, 2, plan.memory_depth, 1, 1 / plan.sample_rate, 0, 0, YINC, -48, 125)
        return codes, preamble, 0

    def close(self):
        pass


def record(scope, session, step, label=""):
    def set_up_bench(message):
        scope.state = {"read": "unplugged", "dark": "dark", "source": "vial"}[step]

    return run_step(scope, session, step, label, SETS, prompt=set_up_bench)


@pytest.fixture(scope="module")
def recorded_session(tmp_path_factory):
    session = tmp_path_factory.mktemp("bench") / "session"
    scope = BenchScope()
    record(scope, session, "read")
    record(scope, session, "dark")
    record(scope, session, "source", "vial")
    return session


def test_each_step_saves_every_capture_as_raw_bytes(recorded_session):
    saved = sorted((recorded_session / "dark").glob("fast_*.npy"))
    assert len(saved) == FAST.repeats
    codes = np.load(saved[0])
    assert codes.dtype == np.uint8
    assert codes.size == 120_000


def test_step_metadata_records_the_scaling_needed_to_convert_to_volts(recorded_session):
    metadata = json.loads((recorded_session / "dark" / "metadata.json").read_text())
    assert metadata["sets"]["fast"]["yinc"] == YINC
    assert metadata["sets"]["fast"]["xinc"] == pytest.approx(2e-8)


def test_a_step_cannot_overwrite_earlier_data(recorded_session):
    with pytest.raises(SettingsError, match="already"):
        record(BenchScope(), recorded_session, "dark")


def test_a_step_is_refused_when_the_channel_settings_have_changed(tmp_path):
    session = tmp_path / "session"
    record(BenchScope(), session, "read")
    with pytest.raises(SettingsError, match="scale_v_per_div"):
        record(BenchScope(scale="0.02"), session, "dark")
    assert not (session / "dark").exists()


def test_a_step_is_refused_when_the_capture_settings_have_changed(tmp_path):
    session = tmp_path / "session"
    record(BenchScope(), session, "read")
    other = CaptureSet("fast", sample_rate=1e7, capture_s=1.2e-3, repeats=20)
    with pytest.raises(SettingsError, match="fast"):
        run_step(BenchScope(), session, "dark", "", [other, SLOW], prompt=lambda message: None)


def test_the_scope_is_restored_when_a_capture_fails(tmp_path):
    scope = BenchScope(fail_after=2)
    with pytest.raises(SettingsError):
        record(scope, tmp_path / "session", "read")
    assert scope.restored == 1


def test_captures_taken_before_a_failure_are_kept(tmp_path):
    session = tmp_path / "session"
    with pytest.raises(SettingsError):
        record(BenchScope(fail_after=2), session, "read")
    assert len(list((session / "read").glob("fast_*.npy"))) == 2


def test_nothing_is_recorded_if_a_capture_set_is_impossible(tmp_path):
    session = tmp_path / "session"
    impossible = CaptureSet("fast", sample_rate=3e6, capture_s=1e-3, repeats=2)
    with pytest.raises(ValueError):
        run_step(BenchScope(), session, "read", "", [impossible], prompt=lambda message: None)
    assert not session.exists()


def test_each_step_asks_for_its_own_bench_setup(tmp_path):
    messages = {}
    for step in ("read", "dark", "source"):
        run_step(
            BenchScope(), tmp_path / "session", step, "vial" if step == "source" else "", SETS,
            prompt=lambda message, step=step: messages.update({step: message}),
        )
    assert "Unplug" in messages["read"]
    assert "Cover" in messages["dark"]
    assert "tritium" in messages["source"]


# The whole pipeline, against the simulated truth above.


@pytest.fixture(scope="module")
def results(recorded_session):
    return analyse_session(recorded_session)


def test_exercise_5_recovers_the_read_noise_and_baseline(results):
    assert results["read"]["std_code"] == pytest.approx(2.0, rel=0.03)
    assert results["read"]["std_mv"] == pytest.approx(4.0, rel=0.03)
    assert results["read"]["baseline_code"] == pytest.approx(77.0, abs=0.05)


def test_exercise_5_shows_read_noise_variance_growing_with_exposure(results):
    windows = results["read"]["windows"]
    per_sample = [w["var"] / w["samples"] for w in windows]
    assert per_sample == pytest.approx([4.0] * len(windows), rel=0.5)
    assert windows[-1]["var"] > 10 * windows[0]["var"]


def test_exercise_6_recovers_the_dark_pulse_rate(results):
    assert results["dark"]["pulses"]["rate_per_s"] == pytest.approx(2e5, rel=0.1)


def test_exercise_6_reports_the_dark_signal_as_a_rate(results):
    # 2e5 events/s * 90 counts at 5e7 Sa/s is 0.36 codes above the baseline per
    # sample, so 3.6e5 counts/s at the slow set's 1e6 Sa/s.
    assert results["dark"]["counts_per_s"] == pytest.approx(3.6e5, rel=0.25)


def test_exercise_7_uses_the_adc_limit_for_dynamic_range(results):
    # (255 - 77 - about 0.36) / 2.02, with rounding to codes adding 1/12 to the variance.
    assert results["dynamic_range"]["per_sample"] == pytest.approx(88.0, rel=0.03)
    assert results["dynamic_range"]["saturation_source"] == "ADC limit, not measured"


def test_exercise_8_recovers_the_gain(results):
    fit = results["source"]["vial"]
    assert abs(fit["gain_counts_per_event"] - PULSE_COUNTS) < 3 * fit["gain_se"]
    assert fit["gain_se"] < 0.25 * PULSE_COUNTS


def test_exercise_8_gain_agrees_with_the_pulse_measurement(results):
    assert results["source"]["vial"]["gain_ratio"] == pytest.approx(1.0, abs=0.4)


def test_photon_conversion_inverts_the_gain(results):
    fit = results["source"]["vial"]
    assert fit["photons_per_count_variance"] == pytest.approx(1 / fit["gain_counts_per_event"])
    assert fit["photons_per_count_pulses"] == pytest.approx(1 / PULSE_COUNTS, rel=0.15)


def test_photon_conversion_is_given_per_volt_second(results):
    # A pulse is 90 counts * 0.002 V * 20 ns = 3.6e-9 V.s.
    fit = results["source"]["vial"]
    assert fit["photons_per_v_s"] == pytest.approx(1 / 3.6e-9, rel=0.15)


def test_photon_conversion_is_given_for_each_sample_rate(results):
    # The same photon is 50 times fewer counts at 1 MSa/s than at 50 MSa/s.
    per_count = results["source"]["vial"]["photons_per_count_by_rate"]
    assert per_count[5e7] == pytest.approx(1 / 90, rel=0.15)
    assert per_count[1e6] == pytest.approx(50 / 90, rel=0.15)


def test_fast_and_slow_captures_give_the_same_photon_rate(results):
    # The slow set misses narrow pulses individually but not on average.
    fit = results["source"]["vial"]
    assert fit["events_per_s_slow"] == pytest.approx(fit["events_per_s_pulses"], rel=0.1)
    # 3e6 from the source minus 2e5 from dark.
    assert fit["events_per_s_pulses"] == pytest.approx(2.8e6, rel=0.15)


def test_crosstalk_does_not_change_the_photon_rate_from_pulses(tmp_path):
    session = tmp_path / "session"
    scope = BenchScope()
    original = dict(BENCH)
    for state in ("dark", "vial"):
        BENCH[state] = dict(BENCH[state], crosstalk=0.3)
    try:
        record(scope, session, "read")
        record(scope, session, "dark")
        record(scope, session, "source", "vial")
    finally:
        BENCH.update(original)
    results = analyse_session(session)
    fit = results["source"]["vial"]
    assert fit["events_per_s_pulses"] == pytest.approx(2.8e6, rel=0.15)
    # The variance route reads low by the excess noise factor of about 1.124.
    assert fit["events_per_s_variance"] < fit["events_per_s_pulses"]
    assert fit["gain_ratio"] == pytest.approx(1.0, abs=0.25)


def test_a_step_is_refused_when_the_bias_step_has_changed(tmp_path):
    session = tmp_path / "session"
    run_step(BenchScope(), session, "dark", "", SETS, prompt=lambda m: None, bias_step=3)
    with pytest.raises(SettingsError, match="bias"):
        run_step(BenchScope(), session, "source", "vial", SETS, prompt=lambda m: None, bias_step=4)
    assert not (session / "source_vial").exists()


def test_the_read_step_does_not_need_a_bias_step(tmp_path):
    session = tmp_path / "session"
    run_step(BenchScope(), session, "read", "", SETS, prompt=lambda m: None)
    run_step(BenchScope(), session, "dark", "", SETS, prompt=lambda m: None, bias_step=3)
    assert json.loads((session / "session.json").read_text())["bias_step"] == 3


@pytest.mark.parametrize("step", ["dark", "source"])
def test_the_command_line_requires_a_bias_step_when_the_detector_is_on(step):
    with pytest.raises(SystemExit):
        parse_args([step, "--session", "x"])
    assert parse_args([step, "--session", "x", "--bias-step", "3"]).bias_step == 3


def test_dynamic_range_is_also_given_with_the_detector_on(results):
    # (255 - 77 - about 0.36) / 2.52
    assert results["dynamic_range"]["per_sample_detector_on"] == pytest.approx(70.6, rel=0.03)


def test_results_and_plots_are_saved(recorded_session, results):
    analysis = recorded_session / "analysis"
    saved = json.loads((analysis / "results.json").read_text())
    assert saved["read"]["std_code"] == results["read"]["std_code"]
    for name in ("read_noise", "dark_counts", "pulse_heights", "gain_vial", "gain_ratio_vial"):
        assert (analysis / f"{name}.png").stat().st_size > 0


def test_saved_results_leave_out_the_individual_pulses(recorded_session, results):
    saved = json.loads((recorded_session / "analysis" / "results.json").read_text())
    assert "heights_up" not in saved["dark"]["pulses"]


def test_the_report_answers_every_exercise(results):
    report = format_report(results)
    for heading in ("Exercise 5", "Exercise 6", "Exercise 7", "Exercise 8"):
        assert heading in report


def test_bursts_of_photons_are_flagged_rather_than_reported_as_gain(tmp_path):
    session = tmp_path / "session"
    scope = BenchScope()
    record(scope, session, "read")
    record(scope, session, "dark")
    bursty = dict(BENCH["vial"], event_rate=3e5, burst_size=10, burst_tau=2e-6)
    original = BENCH["vial"]
    BENCH["vial"] = bursty
    try:
        record(scope, session, "source", "vial")
    finally:
        BENCH["vial"] = original
    fit = analyse_session(session)["source"]["vial"]
    assert fit["gain_ratio"] > 2
    assert any("burst" in warning for warning in fit["warnings"])


def test_a_session_without_dark_data_reports_what_it_can(tmp_path):
    session = tmp_path / "session"
    record(BenchScope(), session, "read")
    results = analyse_session(session)
    assert results["read"]["std_code"] == pytest.approx(2.0, rel=0.03)
    assert results["dark"] is None
    assert "Exercise 6: not recorded yet" in format_report(results)


def test_the_report_only_mentions_unexplained_dark_signal_when_there_is_some(results):
    assert "unexplained" not in format_report(results)
    partly = dict(results, dark=dict(results["dark"], fraction_from_pulses=0.5))
    assert "unexplained" in format_report(partly)

"""Tests for the calculations in the SiPM exposure sweep.

The scope itself is exercised by running the script against real hardware;
these cover the parts where a silent error would corrupt the measurement.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sipm_exposure_sweep import (  # noqa: E402
    CaptureError,
    clipped_fraction,
    counts_axis_scale,
    integrate_counts,
    parse_args,
    parse_preamble,
    plan_capture,
    plot_counts,
    Scope,
    SettingsError,
    summarize,
)

# Captured from the DS1074Z at 1 ms/div, 12k points.
REAL_PREAMBLE = "0,2,12000,1,1.000000e-06,0.000000e+00,0,8.093750e-04,-50,125"


def test_parse_preamble_reads_the_scope_fields():
    pre = parse_preamble(REAL_PREAMBLE)
    assert pre.points == 12000
    assert pre.xinc == pytest.approx(1e-6)
    assert pre.yinc == pytest.approx(8.09375e-4)
    assert pre.yorigin == -50
    assert pre.yref == 125


def test_parse_preamble_rejects_a_wrong_field_count():
    with pytest.raises(ValueError, match="10 fields"):
        parse_preamble("0,2,12000")


@pytest.mark.parametrize(
    "exposure, scale, depth, samples",
    [
        (40e-6, 1e-3, 12_000, 40),
        (1e-3, 1e-3, 12_000, 1_000),
        (10e-3, 1e-3, 12_000, 10_000),
        (0.1, 1e-2, 120_000, 100_000),
        (1.0, 0.1, 1_200_000, 1_000_000),
        (10.0, 1.0, 12_000_000, 10_000_000),
    ],
)
def test_plan_holds_one_megasample_per_second(exposure, scale, depth, samples):
    plan = plan_capture(exposure, sample_rate=1e6)
    assert plan.timebase_scale == pytest.approx(scale)
    assert plan.memory_depth == depth
    assert plan.samples == samples


def test_plan_window_always_covers_the_exposure():
    for exposure in (40e-6, 12e-3, 0.11, 1.2, 12.0):
        plan = plan_capture(exposure, sample_rate=1e6)
        assert plan.timebase_scale * 12 >= exposure


def test_plan_rejects_an_exposure_longer_than_memory_allows():
    with pytest.raises(ValueError, match="longer"):
        plan_capture(13.0, sample_rate=1e6)


def test_plan_rejects_an_exposure_shorter_than_one_sample():
    with pytest.raises(ValueError, match="shorter"):
        plan_capture(1e-7, sample_rate=1e6)


def test_plan_rejects_a_rate_the_timebase_cannot_reach():
    # 12k / (12 * scale) = 3 MSa/s needs a scale outside the 1-2-5 sequence.
    with pytest.raises(ValueError, match="sample rate"):
        plan_capture(1e-3, sample_rate=3e6)


# Measured on the DS1074Z: above 10 MSa/s the memory runs past the screen and only
# certain rates exist, so depth / (12 * scale) no longer gives the rate.
@pytest.mark.parametrize(
    "rate, exposure, scale, depth",
    [
        (5e8, 2.4e-3, 1e-4, 1_200_000),
        (5e8, 24e-3, 1e-3, 12_000_000),
        (5e7, 24e-3, 1e-3, 1_200_000),
        (5e7, 2.4e-3, 1e-4, 120_000),
        (1.25e8, 9.6e-3, 5e-4, 1_200_000),
    ],
)
def test_plan_uses_measured_settings_at_high_rates(rate, exposure, scale, depth):
    plan = plan_capture(exposure, sample_rate=rate)
    assert plan.timebase_scale == pytest.approx(scale)
    assert plan.memory_depth == depth
    assert plan.samples == round(exposure * rate)


def test_plan_rejects_a_high_rate_the_scope_does_not_offer():
    # The formula reaches 100 MSa/s, but the scope gave 50 MSa/s at those settings.
    with pytest.raises(ValueError, match="sample rate"):
        plan_capture(1e-3, sample_rate=1e8)


def test_plan_window_is_how_long_the_memory_takes_to_fill():
    # 1.2M points at 500 MSa/s take 2.4 ms, twice the 1.2 ms on screen at 100 us/div.
    assert plan_capture(2.4e-3, sample_rate=5e8).window_s == pytest.approx(2.4e-3)


def test_integrate_counts_sums_codes_above_the_baseline():
    codes = np.array([80.0, 82.0, 78.0, 80.0])
    assert integrate_counts(codes, baseline_code=75.0) == pytest.approx(20.0)


def test_integrate_counts_can_be_negative_below_the_baseline():
    assert integrate_counts(np.array([70.0, 70.0]), 75.0) == pytest.approx(-10.0)


def test_clipped_fraction_counts_codes_at_the_adc_limits():
    codes = np.array([0.0, 100.0, 255.0, 128.0])
    assert clipped_fraction(codes) == pytest.approx(0.5)


def test_clipped_fraction_is_zero_for_codes_inside_the_range():
    assert clipped_fraction(np.array([1.0, 254.0])) == 0.0


def test_summarize_converts_counts_to_a_rate_per_exposure():
    rows = [
        {"exposure_s": 1.0, "counts": 100.0},
        {"exposure_s": 1.0, "counts": 300.0},
        {"exposure_s": 0.5, "counts": 100.0},
        {"exposure_s": 0.5, "counts": 200.0},
    ]
    by_exposure = {s.exposure_s: s for s in summarize(rows)}
    assert by_exposure[1.0].mean_counts_per_s == pytest.approx(200.0)
    assert by_exposure[0.5].mean_counts_per_s == pytest.approx(300.0)


def test_summarize_reports_sample_variance_of_integrated_counts():
    # The photon conversion needs var(counts), with n-1 in the denominator.
    rows = [
        {"exposure_s": 1.0, "counts": 100.0},
        {"exposure_s": 1.0, "counts": 300.0},
    ]
    summary = summarize(rows)[0]
    assert summary.var_counts == pytest.approx(20_000.0)
    assert summary.std_counts_per_s == pytest.approx(np.sqrt(20_000.0))
    assert summary.repeats == 2


def test_summarize_orders_exposures_from_shortest_to_longest():
    rows = [
        {"exposure_s": 1.0, "counts": 1.0},
        {"exposure_s": 1e-3, "counts": 1.0},
        {"exposure_s": 0.1, "counts": 1.0},
    ]
    assert [s.exposure_s for s in summarize(rows)] == [1e-3, 0.1, 1.0]


def test_summarize_leaves_spread_undefined_for_a_single_repeat():
    summary = summarize([{"exposure_s": 1.0, "counts": 5.0}])[0]
    assert np.isnan(summary.var_counts)
    assert np.isnan(summary.std_counts_per_s)


def test_default_exposures_stop_at_one_second():
    assert max(parse_args([]).exposures) == 1.0


def test_ten_second_exposures_can_still_be_requested():
    assert parse_args(["--exposures", "10"]).exposures == [10.0]


def test_counts_axis_is_log_when_every_mean_is_positive():
    assert counts_axis_scale([120.0, 2_700.0, 2_800_000.0]) == "log"


def test_counts_axis_falls_back_when_a_mean_is_not_positive():
    # A true baseline can leave short exposures at or below zero.
    assert counts_axis_scale([-15.0, 400.0, 28_000.0]) == "symlog"


def test_plot_counts_writes_an_image(tmp_path):
    rows = [
        {"exposure_s": t, "counts": c}
        for t, c in [(1e-3, 2_500.0), (1e-3, 2_900.0), (1.0, 2.7e6), (1.0, 2.8e6)]
    ]
    path = tmp_path / "counts.png"
    plot_counts(summarize(rows), "title", path)
    assert path.stat().st_size > 0


class ScopeWithoutHardware(Scope):
    """Runs the real retry logic, counting USB resets instead of performing them."""

    def __init__(self):
        self.recoveries = 0

    def recover(self):
        self.recoveries += 1


def failing_then_returning(failures, error):
    remaining = [error] * failures

    def action():
        if remaining:
            raise remaining.pop()
        return "done"

    return action


def test_recovery_retries_a_usb_fault_and_reports_the_retries():
    import usb.core

    scope = ScopeWithoutHardware()
    action = failing_then_returning(2, usb.core.USBError("Input/Output Error"))
    assert scope.with_recovery(action) == ("done", 2)
    assert scope.recoveries == 2


def test_recovery_reruns_setup_before_each_retry():
    scope = ScopeWithoutHardware()
    setups = []
    action = failing_then_returning(1, TimeoutError("no reply"))
    scope.with_recovery(action, before_retry=lambda: setups.append("configured"))
    assert setups == ["configured"]


def test_recovery_gives_up_after_the_attempt_limit():
    scope = ScopeWithoutHardware()
    action = failing_then_returning(99, TimeoutError("no reply"))
    with pytest.raises(TimeoutError):
        scope.with_recovery(action)


def test_recovery_does_not_retry_a_settings_problem():
    scope = ScopeWithoutHardware()
    action = failing_then_returning(1, SettingsError("Ch2 is on"))
    with pytest.raises(SettingsError):
        scope.with_recovery(action)
    assert scope.recoveries == 0


def test_opening_a_wedged_usb_endpoint_recovers_instead_of_failing(monkeypatch):
    import pyvisa
    import usb.core

    monkeypatch.setattr(pyvisa, "ResourceManager", lambda backend: object())
    monkeypatch.setattr(Scope, "_find_resource", lambda self: "USB0::6833::SIM::INSTR")

    def wedged(self):
        raise usb.core.USBError("Input/Output Error")

    recoveries = []
    monkeypatch.setattr(Scope, "_open", wedged)
    monkeypatch.setattr(Scope, "recover", lambda self: recoveries.append(True))
    Scope()
    assert recoveries == [True]


class ScopeStates(Scope):
    """The trigger states seen on a DS1074Z with a flat input.

    :SING from STOP takes a while to re-arm, and the status reads STOP until it has.
    From a running scope it arms at once. Under single sweep, :RUN arms one capture
    and stops, so running again needs the sweep set back to AUTO first.
    """

    def __init__(self, rearm_reads=3, triggers_at_once=False, stuck_in_stop=False):
        self.state = "AUTO"
        self.sweep = "AUTO"
        self.armed = False
        self.rearming = 0
        self.rearm_reads = rearm_reads
        self.triggers_at_once = triggers_at_once
        self.stuck_in_stop = stuck_in_stop
        self.captures = 0
        self.commands = []

    def write(self, command):
        self.commands.append(command)
        if command == ":TRIG:SWE AUTO":
            self.sweep = "AUTO"
        elif command == ":RUN" and self.sweep == "AUTO" and not self.stuck_in_stop:
            self.state = "AUTO"
        elif command == ":SING":
            self.sweep = "SING"
            if self.state == "STOP":
                self.rearming = self.rearm_reads
            elif self.triggers_at_once:
                self._finish()
            else:
                self.armed = True
        elif command == ":TFOR" and self.armed:
            self._finish()

    def _finish(self):
        self.armed = False
        self.state = "STOP"
        self.captures += 1

    def query(self, command):
        assert command == ":TRIG:STAT?"
        # Like the real pause between commands, longer than the capture window.
        time.sleep(0.005)
        if self.rearming:
            self.rearming -= 1
            if not self.rearming:
                self.armed = True
                self.state = "AUTO"
            return "STOP"
        return self.state


def short_plan():
    # A window far shorter than the pause between commands.
    return plan_capture(2.4e-3, sample_rate=5e8)


def test_each_capture_waits_for_new_data_rather_than_a_leftover_stop():
    scope = ScopeStates()
    for expected in (1, 2, 3):
        scope.capture(short_plan())
        assert scope.captures == expected


def test_a_capture_triggered_at_once_is_accepted_without_waiting():
    scope = ScopeStates(triggers_at_once=True)
    scope.capture(short_plan())
    scope.capture(short_plan())
    assert scope.captures == 2
    assert ":TFOR" not in scope.commands


def test_a_scope_that_will_not_start_running_is_a_retryable_capture_error(monkeypatch):
    monkeypatch.setattr("sipm_exposure_sweep.RUN_TIMEOUT_S", 0.01)
    scope = ScopeStates(stuck_in_stop=True)
    scope.state = "STOP"
    with pytest.raises(CaptureError, match="STOP"):
        scope.capture(short_plan())
    assert ":SING" not in scope.commands

"""The camera analysis recovers the simulated camera's known bias, read noise, dark current and gain."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import noise_analysis as na  # noqa: E402
from basler_camera import SimulatedCamera, snap  # noqa: E402
from recording import RecordingWriter, find_recordings, load_recording  # noqa: E402

QUANTISATION_VAR = 1 / 12  # flooring to whole counts


def camera(**kwargs):
    kwargs.setdefault("seed", 1)
    cam = SimulatedCamera(shape=(80, 120), realtime=False, **kwargs)
    cam.start()
    return cam


def stack(cam, n):
    return np.stack([cam.grab(1).image for _ in range(n)])


@pytest.mark.parametrize("gain_db", [0.0, 12.0, 24.0])
def test_read_noise_matches_simulated_value_with_an_offset(gain_db):
    cam = camera()
    cam.set("black_level", 64)
    cam.set("gain", gain_db)
    cam.set("exposure_us", 21)
    result = na.read_noise(stack(cam, 60))
    expected = math.sqrt(cam.read_noise_counts() ** 2 + QUANTISATION_VAR)
    assert result.read_noise == pytest.approx(expected, rel=0.05)
    assert result.bias == pytest.approx(64 - 0.5, abs=0.2)
    assert not result.clipped


def test_read_noise_without_offset_is_clipped_and_too_low():
    cam = camera()
    cam.set("black_level", 0)
    cam.set("gain", 12)
    cam.set("exposure_us", 21)
    result = na.read_noise(stack(cam, 60))
    assert result.clipped
    assert result.frac_zero > 0.3
    assert result.read_noise < 0.8 * cam.read_noise_counts()


def test_dark_counts_slope_is_dark_current_times_gain():
    cam = camera(dark_e_per_s=50.0, read_noise_e=2.0)
    cam.set("black_level", 64)
    cam.set("gain", 12)
    exposures = [21e-6, 0.1, 0.5, 1.0, 2.0]
    means = []
    for t in exposures:
        cam.set("exposure_us", t * 1e6)
        means.append(na.frame_means(stack(cam, 3)).mean())
    fit = na.dark_counts(exposures, means)
    assert fit.slope == pytest.approx(50.0 * cam.gain_counts_per_e(), rel=0.03)
    assert fit.intercept == pytest.approx(0, abs=0.3)
    assert fit.bias == pytest.approx(means[0])
    assert fit.subtracted[0] == 0


def test_dark_counts_sorts_exposures():
    fit = na.dark_counts([1.0, 0.0, 0.5], [12.0, 2.0, 7.0])
    np.testing.assert_allclose(fit.exposures_s, [0, 0.5, 1])
    assert fit.bias == 2.0
    assert fit.slope == pytest.approx(10.0)


def test_steady_frames_keeps_the_longest_run_without_a_jump():
    np.testing.assert_array_equal(na.steady_frames([5.0, 5.01, 4.99]), [0, 1, 2])
    # 2 frames high, then 8 low: the low run is used
    means = [129.2, 129.2] + [127.2] * 8
    np.testing.assert_array_equal(na.steady_frames(means), np.arange(2, 10))
    # two jumps: the middle run is longest
    np.testing.assert_array_equal(na.steady_frames([137.0, 135.0, 135.0, 135.0, 132.9]), [1, 2, 3])
    # a tie goes to the later run
    np.testing.assert_array_equal(na.steady_frames([1.0, 1.0, 3.0, 3.0]), [2, 3])
    assert na.steady_frames([]).size == 0
    # a slow creep is not a jump
    assert na.steady_frames(np.linspace(137.72, 137.58, 10)).size == 10


def test_dynamic_range_formula():
    ratio, db = na.dynamic_range(saturation=4095, bias=95, dark=0, read_noise=4)
    assert ratio == 1000
    assert db == pytest.approx(60)


def uneven_stacks(gain_db, frames, exposure_us, **options):
    """A covered stack at the minimum exposure (for the bias) and a lit one at `exposure_us`."""
    cam = camera(light=2e5, **options)
    cam.set("black_level", 64)
    cam.set("gain", gain_db)
    cam.set("exposure_us", 21)
    bias = float(stack(cam, 20).mean())
    cam.set("exposure_us", exposure_us)
    cam.covered = False
    return cam, bias, stack(cam, frames)


@pytest.mark.parametrize("gain_db", [0.0, 12.0, 24.0])
def test_shot_noise_slope_is_gain_from_one_unevenly_lit_stack(gain_db):
    # Exposure scaled with gain, so the brightest pixels just saturate at every gain
    cam, bias, lit = uneven_stacks(gain_db, 40, 200_000 * 10 ** (-gain_db / 20), light_jitter=0.005)
    result = na.shot_noise(lit, bias, cam.max_value, normalise=True)
    k = cam.gain_counts_per_e()
    assert result.fit.slope == pytest.approx(k, rel=0.03)
    assert result.frame_mean_cv == pytest.approx(0.005, rel=0.6)
    assert 0 < result.frac_saturated < 0.05
    assert result.frac_zero == 0
    assert result.pixels == lit[0].size - round(result.frac_saturated * lit[0].size)
    assert result.mean.max() < cam.max_value - bias


def test_shot_noise_leaves_out_clipped_pixels():
    # Far too bright: most pixels saturate. Keeping them would flatten the line.
    cam, bias, lit = uneven_stacks(12.0, 30, 400_000)
    result = na.shot_noise(lit, bias, cam.max_value)
    assert result.frac_saturated > 0.3
    assert result.pixels < 0.7 * lit[0].size
    # So overexposed, the unclipped pixels only span about 2000-2700 counts below the
    # fit limit, so the slope is less precise than with a well exposed image
    assert result.fit.slope == pytest.approx(cam.gain_counts_per_e(), rel=0.1)
    # A pixel at 0 in one frame is clipped too
    lit = lit.copy()
    lit[3].flat[np.flatnonzero(lit[0] < cam.max_value)[0]] = 0
    assert na.shot_noise(lit, bias, cam.max_value).pixels == result.pixels - 1


@pytest.mark.parametrize("light", [2e5, 4e5])
def test_shot_noise_fit_stops_before_a_full_well(light):
    # The well fills at about 2520 counts at 0 dB, below the largest value 4095, so those
    # pixels are not clipped but their variance collapses. Fitting them flips the slope.
    cam, bias, lit = uneven_stacks(0.0, 40, 200_000, full_well=20_000, light_jitter=0.002)
    cam.light = light
    lit = stack(cam, 40)
    result = na.shot_noise(lit, bias, cam.max_value)
    assert result.frac_saturated == 0
    assert result.saturation_mean == pytest.approx(20_000 * cam.gain_counts_per_e(), rel=0.1)
    assert result.fit_max == pytest.approx(na.FIT_FRACTION * result.saturation_mean)
    assert result.mean[result.mean < result.fit_max].max() < result.fit_max
    assert result.fit.slope == pytest.approx(cam.gain_counts_per_e(), rel=0.05)


def test_shot_noise_normalisation_removes_light_flicker():
    cam, bias, lit = uneven_stacks(6.0, 30, 100_000, light_jitter=0.03)
    k = cam.gain_counts_per_e()
    assert na.shot_noise(lit, bias, cam.max_value, normalise=True).fit.slope == pytest.approx(k, rel=0.08)
    assert na.shot_noise(lit, bias, cam.max_value, normalise=False).fit.slope > 1.5 * k


def test_binned_averages_variance_per_bin_of_mean():
    mean = np.repeat(np.arange(10.0), 30)
    var = 2 * mean + 1
    bx, by = na.binned(mean, var, bins=10, min_pixels=20)
    np.testing.assert_allclose(bx, np.arange(10.0))
    np.testing.assert_allclose(by, 2 * bx + 1)
    assert na.binned(mean, var, bins=10, min_pixels=31)[0].size == 0


def test_check_block_ids():
    assert na.check_block_ids([1, 2, 3, 4]) == (0, 0)
    assert na.check_block_ids([1, 2, 5, 6]) == (2, 0)
    assert na.check_block_ids([1, 2, 2, 3]) == (0, 1)


def test_snap_to_camera_increment():
    assert snap(12.34, 0, 24, 0.1) == pytest.approx(12.3)
    assert snap(-5, 0, 24, 0.1) == 0
    assert snap(99, 0, 24, None) == 24


def test_recording_round_trip(tmp_path):
    cam = camera()
    cam.set("black_level", 64)
    writer = RecordingWriter(tmp_path / "read_noise" / "0dB", {"section": "read_noise", **cam.settings()}, analysis_step=2)
    frames = []
    for _ in range(5):
        f = cam.grab(1)
        frames.append(f.image)
        writer.add(f.image, f.block_id, f.timestamp)
    meta = writer.close()
    assert meta["frames"] == 5 and meta["missing_frames"] == 0
    loaded, meta2 = load_recording(tmp_path / "read_noise" / "0dB")
    np.testing.assert_array_equal(loaded, np.stack(frames))
    assert loaded.dtype == np.uint16
    assert meta2["black_level"] == 64
    np.testing.assert_array_equal(writer.stack(), np.stack(frames)[:, ::2, ::2])
    assert find_recordings(tmp_path) == [tmp_path / "read_noise" / "0dB"]
    with pytest.raises(FileExistsError):
        RecordingWriter(tmp_path / "read_noise" / "0dB", {})


def test_read_noise_leaves_out_frames_before_an_offset_jump():
    rng = np.random.default_rng(0)
    steady = np.round(rng.normal(64, 2, (100, 80, 80))).astype(np.uint16)
    jumped = steady.copy()
    jumped[:30] += 2
    result = na.read_noise(jumped)
    expected = na.read_noise(steady[30:])
    assert result.frames == 70
    assert result.read_noise == pytest.approx(expected.read_noise) and result.bias == pytest.approx(expected.bias)
    # frame means of a small, noisy image wander by more than a jump does; that isn't one
    noisy = np.round(rng.normal(100, 14, (100, 20, 20))).astype(np.uint16)
    assert na.read_noise(noisy).frames == 100

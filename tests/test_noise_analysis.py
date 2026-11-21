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


def test_dynamic_range_formula():
    ratio, db = na.dynamic_range(saturation=4095, bias=95, dark=0, read_noise=4)
    assert ratio == 1000
    assert db == pytest.approx(60)


@pytest.mark.parametrize("gain_db", [0.0, 12.0])
def test_shot_noise_slope_is_gain_and_intercept_read_variance(gain_db):
    cam = camera(light=2e5, light_jitter=0.005)
    cam.set("black_level", 64)
    cam.set("gain", gain_db)
    cam.set("exposure_us", 20_000)
    dark = stack(cam, 40)
    cam.covered = False
    lit = stack(cam, 40)
    result = na.shot_noise(lit, dark, normalise=True)
    k = cam.gain_counts_per_e()
    assert result.fit.slope == pytest.approx(k, rel=0.05)
    assert result.frame_mean_cv == pytest.approx(0.005, rel=0.6)
    read_var = cam.read_noise_counts() ** 2 + QUANTISATION_VAR
    assert result.fit.intercept == pytest.approx(read_var, rel=0.25)


def test_shot_noise_normalisation_removes_light_flicker():
    cam = camera(light=2e5, light_jitter=0.03)
    cam.set("black_level", 64)
    cam.set("gain", 6)
    cam.set("exposure_us", 20_000)
    dark = stack(cam, 30)
    cam.covered = False
    lit = stack(cam, 30)
    k = cam.gain_counts_per_e()
    assert na.shot_noise(lit, dark, normalise=True).fit.slope == pytest.approx(k, rel=0.08)
    assert na.shot_noise(lit, dark, normalise=False).fit.slope > 1.5 * k


def test_pooled_fit_combines_recordings():
    x1, x2 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    fit = na.pooled_fit([(x1, 2 * x1 + 1), (x2, 2 * x2 + 1)])
    assert fit.slope == pytest.approx(2)
    assert fit.intercept == pytest.approx(1)


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

"""Drive the camera GUI offscreen with the simulated camera, through every tab, then reopen the session."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pg = pytest.importorskip("pyqtgraph")

from basler_camera import SimulatedCamera  # noqa: E402
from basler_live import SETTLE_S, MainWindow, Recording, Step, estimate_seconds, visit_order  # noqa: E402
from live_widgets import close_window, format_duration, parse_numbers  # noqa: E402
from pyqtgraph.Qt import QtWidgets  # noqa: E402


def wait_idle(win, app, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        app.processEvents()
        if not win.busy and not win.worker.commands.qsize():
            # let queued signals land
            for _ in range(20):
                app.processEvents()
                time.sleep(0.005)
            if not win.busy:
                return
        if time.monotonic() > deadline:
            raise TimeoutError("GUI did not finish")
        time.sleep(0.01)


def test_helpers():
    assert parse_numbers("0, 1 ,12;24") == [0, 1, 12, 24]
    assert format_duration(75) == "1 min 15 s"
    steps = [Step(Path("x"), {"exposure_us": 1e6}, 10)]
    assert estimate_seconds(steps) == pytest.approx(10 + max(1, SETTLE_S) + 0.3)
    # A 10 s exposure discards one frame, not ten
    assert estimate_seconds([Step(Path("x"), {"exposure_us": 10e6}, 1)]) == pytest.approx(10 + 10 + 0.3)


def test_visit_order_ramps_up_and_back_within_each_gain():
    steps = [Step(Path(f"{g}/{e}"), {"gain": g, "exposure_us": e}, 3) for g in (0, 12) for e in (1e6, 40, 1e5)]
    # within a gain, by exposure: 40 us is step 1, 100 ms step 2, 1 s step 0
    assert visit_order(steps) == [
        (1, 0), (2, 0), (0, 0), (0, 1), (2, 1), (1, 1), (1, 2), (2, 2), (0, 2),
        (4, 0), (5, 0), (3, 0), (3, 1), (5, 1), (4, 1), (4, 2), (5, 2), (3, 2),
    ]
    # every frame of every step is visited once
    assert sorted(visit_order(steps)) == sorted((i, p) for i in range(6) for p in range(3))


def test_interleaved_estimate_counts_a_thrown_away_frame_per_change():
    steps = [Step(Path(str(e)), {"gain": 0, "exposure_us": e}, 2) for e in (1e6, 10e6)]
    # 1 s, 10 s | 10 s, 1 s: settle at the start, one frame thrown away per change, none at the turn
    expected = (max(1, SETTLE_S) + 0.3 + 1) + (10 + 0.05 + 10) + 10 + (1 + 0.05 + 1)
    assert estimate_seconds(steps, interleave=True) == pytest.approx(expected)


def test_interleaved_dark_recording(tmp_path):
    app = pg.mkQApp()
    cam = SimulatedCamera(shape=(40, 60), seed=6, max_fps=500)
    win = MainWindow(cam, tmp_path / "s", analysis_step=1, simulate=True)
    try:
        win.dark_tab.current_only.setChecked(True)
        win.dark_tab.exposure_edit.setText("0.05, 5, 20")
        win.dark_tab.frames.setValue(4)
        win.dark_tab.order.setCurrentIndex(1)
        assert win.dark_tab.interleaved()
        win.dark_tab.record()
        wait_idle(win, app)
        gain = win.current["gain"]
        metas = {}
        for folder in (tmp_path / "s" / "dark" / f"{gain:g}dB").iterdir():
            m = json.loads((folder / "metadata.json").read_text())
            metas[m["exposure_us"]] = m
        assert sorted(metas) == [50, 5000, 20000]
        for m in metas.values():
            assert m["order"] == "interleaved" and m["passes"] == [0, 1, 2, 3]
            assert m["frames"] == 4 and m["missing_frames"] == 0 and m["repeated_frames"] == 0
        # The frames of different exposures alternate in time, up and back down
        times = sorted((t, e) for e, m in metas.items() for t in m["timestamps"])
        assert [e for _, e in times] == [50, 5000, 20000, 20000, 5000, 50] * 2
        assert win.dark_tab.lookup(gain, win.current["black_level"]) is not None
        assert win.dark_tab.series_choice.count() == 1
        assert len(win.dark_tab.time_plot.getPlotItem().listDataItems()) == 3
    finally:
        close_window(win)


def test_offset_jumps_are_corrected_without_a_warning(tmp_path, monkeypatch):
    app = pg.mkQApp()
    win = MainWindow(SimulatedCamera(shape=(40, 60), seed=9, max_fps=500), tmp_path / "s", analysis_step=1, simulate=True)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: pytest.fail("a warning was shown"))
    (tmp_path / "s").mkdir()
    try:
        stack = np.round(np.random.default_rng(1).normal(64, 2, (50, 40, 60))).astype(np.uint16)
        stack[0] += 2  # as seen on the real camera at 24 dB
        meta = {
            "section": "read_noise", "gain": 24.0, "black_level": 64.0, "max_value": 4095, "frames": 50,
            "missing_frames": 0, "repeated_frames": 0, "frame_means": stack.mean(axis=(1, 2)).tolist(),
        }
        win.callbacks["read_noise"] = win.read_tab.on_recorded
        win.on_recorded("read_noise", [Recording(Path("read_noise/24dB"), meta, stack)])
        row = win.read_tab.lookup(24, 64)
        assert row["bias"] == pytest.approx(stack[1:].mean()) and row["frames"] == 50 and row["frames_used"] == 49

        # flat within a run, with a 2 count jump after the first frame
        dark = [
            Recording(Path(f"dark/24dB/{e}us"), {**meta, "section": "dark", "exposure_us": e, "frame_means": [66.0 + s] + [64.0 + s] * 9}, None)
            for e, s in ((40, 0.0), (1_000_000, 10.0))
        ]
        win.callbacks["dark"] = win.dark_tab.on_recorded
        win.on_recorded("dark", dark)
        assert win.dark_tab.lookup(24, 64).bias == pytest.approx(64)
        assert "jump" not in win.dark_tab.message.text() and not win.dark_tab.message.styleSheet()
        assert win.statusBar().currentMessage().startswith("Saved")
    finally:
        close_window(win)


def test_shot_noise_checks_read_noise_when_the_camera_reports_no_gain_increment(tmp_path):
    # A real camera's float gain node may have no increment: limits["gain"] is (lo, hi, None)
    app = pg.mkQApp()
    cam = SimulatedCamera(shape=(60, 90), seed=5, max_fps=500)
    win = MainWindow(cam, tmp_path / "s", analysis_step=1, simulate=True)
    try:
        cam.limits["gain"] = (*cam.limits["gain"][:2], None)
        win.shot_tab.gain_edit.setText("0")
        win.shot_tab.record()  # no read noise yet: refused with a message, not an exception
        assert not win.busy and "Record read noise" in win.shot_tab.message.text()
        app.processEvents()
    finally:
        close_window(win)


def test_full_session_with_simulator(tmp_path):
    app = pg.mkQApp()
    cam = SimulatedCamera(shape=(60, 90), seed=3, max_fps=500)
    win = MainWindow(cam, tmp_path / "s1", analysis_step=1, simulate=True)
    try:
        win.set_setting("black_level", 64)
        win.set_setting("gain", 12)
        win.set_setting("exposure_us", 5000)
        deadline = time.monotonic() + 5
        while win.current["black_level"] != 64 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

        win.read_tab.gain_edit.setText("0, 12")
        win.read_tab.frames.setValue(20)
        win.read_tab.record()
        wait_idle(win, app)
        assert len(win.read_tab.by_key) == 2
        rn = win.read_tab.lookup(12, 64)
        assert rn["read_noise"] == pytest.approx((cam.read_noise_counts(12) ** 2 + 1 / 12) ** 0.5, rel=0.1)
        assert rn["frac_zero"] == 0

        win.dark_tab.current_only.setChecked(True)
        win.dark_tab.exposure_edit.setText("0.05, 10, 40")
        win.dark_tab.frames.setValue(2)
        win.dark_tab.record()
        wait_idle(win, app)
        assert win.dark_tab.lookup(12, 64) is not None

        win.dr_tab.refresh_exposures()
        win.dr_tab.redraw()
        assert any(r["gain"] == 12 and r["ratio"] > 100 for r in win.dr_tab.rows)

        # Read noise is only at 0 and 12 dB, so 24 dB is refused until it is recorded
        win.shot_tab.gain_edit.setText("0, 12, 24")
        win.shot_tab.record()
        assert not win.busy and "24" in win.shot_tab.message.text()

        # Bright enough at 12 dB, 5 ms that the brightest pixels saturate
        cam.light = 2e6
        win.shot_tab.gain_edit.setText("0, 12")
        win.shot_tab.frames.setValue(30)
        assert [s.settings["exposure_us"] for s in win.shot_tab.steps()] == pytest.approx([5000 * 10 ** (12 / 20), 5000], rel=1e-3)
        win.shot_tab.record()
        wait_idle(win, app)
        for g in (0, 12):
            row = win.shot_tab.by_key[(g, 64)]
            assert row["gain_counts_per_e"] == pytest.approx(cam.gain_counts_per_e(g), rel=0.05)
            assert 0 < row["frac_saturated"] < 0.5
            assert win.shot_tab.gain_for(g, 64) == row["gain_counts_per_e"]

        assert win.shot_tab.vs_gain.getPlotItem().ctrl.logYCheck.isChecked()
        win.shot_tab.log_y.setChecked(False)
        assert not win.shot_tab.vs_gain.getPlotItem().ctrl.logYCheck.isChecked()

        results = json.loads((tmp_path / "s1" / "results.json").read_text())
        assert {r["gain"] for r in results["shot_noise"]} == {0, 12}
        assert all(r["read_noise_e"] > 0 for r in results["shot_noise"])
        meta = json.loads((tmp_path / "s1" / "shot" / "12dB" / "metadata.json").read_text())
        assert meta["gain"] == 12 and meta["black_level"] == 64 and meta["missing_frames"] == 0
        assert meta["exposure_us"] == pytest.approx(5000)
    finally:
        close_window(win)

    cam2 = SimulatedCamera(shape=(60, 90), seed=4, max_fps=500)
    win2 = MainWindow(cam2, tmp_path / "s1", analysis_step=1, simulate=True)
    try:
        win2.open_session(tmp_path / "s1")
        assert len(win2.read_tab.by_key) == 2
        assert win2.dark_tab.lookup(12, 64) is not None
        assert set(win2.shot_tab.by_key) == {(0, 64), (12, 64)}
    finally:
        close_window(win2)
    QtWidgets.QApplication.processEvents()


def red(widget) -> bool:
    return bool(widget.styleSheet())


def test_clipped_read_noise_happens_without_comment(tmp_path):
    # Students are meant to run into this and work out why: the app shows the numbers and no verdict
    app = pg.mkQApp()
    cam = SimulatedCamera(shape=(40, 60), seed=5, max_fps=500)
    assert cam.get("black_level") == 0
    win = MainWindow(cam, tmp_path / "s", analysis_step=1, simulate=True)
    try:
        win.read_tab.gain_edit.setText("24")
        win.read_tab.frames.setValue(20)
        win.read_tab.record()
        wait_idle(win, app)
        assert win.current["black_level"] == 0  # not raised for them
        row = win.read_tab.lookup(24, 0)
        assert row["frac_zero"] > 0.3
        t = win.read_tab.table
        assert t.item(0, 4).text() == f"{100 * row['frac_zero']:.4g}"
        normal = QtWidgets.QTableWidgetItem().foreground()
        assert all(t.item(0, c).foreground() == normal for c in range(t.columnCount()))
        assert not red(win.read_tab.message) and not red(win.stats)
        text = win.read_tab.message.text() + win.read_tab.findChildren(QtWidgets.QLabel)[0].text()
        assert "Raise" not in text and "too low" not in text and "black level high" not in text
    finally:
        close_window(win)


def test_recording_again_replaces_the_take_the_notebook_reads(tmp_path):
    from recording import find_recordings, load_metadata

    app = pg.mkQApp()
    cam = SimulatedCamera(shape=(40, 60), seed=6, max_fps=500)
    session = tmp_path / "s"
    win = MainWindow(cam, session, analysis_step=1, simulate=True)
    try:
        win.read_tab.current_only.setChecked(True)
        win.read_tab.frames.setValue(5)
        for black_level in (20, 80):
            win.set_setting("black_level", black_level)
            deadline = time.monotonic() + 5
            while win.current["black_level"] != black_level and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            win.read_tab.record()
            wait_idle(win, app)
        gain = f"{win.current['gain']:g}dB"
        assert load_metadata(session / "read_noise" / gain)["black_level"] == 80
        assert not (session / "read_noise" / f"{gain}_2").exists()
        assert list((session / "superseded").rglob("metadata.json"))
        assert find_recordings(session) == [session / "read_noise" / gain]
    finally:
        close_window(win)

    win2 = MainWindow(SimulatedCamera(shape=(40, 60), seed=7, max_fps=500), session, analysis_step=1, simulate=True)
    try:
        win2.open_session(session)
        assert [k[1] for k in win2.read_tab.by_key] == [80.0]
    finally:
        close_window(win2)


def test_starts_with_only_the_live_view_and_its_controls(tmp_path):
    app = pg.mkQApp()
    win = MainWindow(SimulatedCamera(shape=(40, 60), seed=8, max_fps=500), tmp_path / "s", analysis_step=1, simulate=True)
    try:
        win.show()
        app.processEvents()
        assert win.view.titles() == ["Live", "Read noise", "Dark counts", "Dynamic range", "Shot noise"]
        assert win.image.isVisible() and win.exposure.isVisible()
        hidden = [win.read_tab, win.progress, win.cancel_button, *win.session_buttons]
        assert not any(w.isVisible() for w in hidden)

        win.view.select("Dark counts")
        app.processEvents()
        assert win.dark_tab.isVisible() and not win.read_tab.isVisible()
        assert win.image.isVisible() and win.exposure.isVisible()
        assert all(w.isVisible() for w in hidden[1:])
        # both sides get a usable share of the window
        assert min(win.view.splitter.sizes()) > 0.25 * win.width()

        win.view.select("Live")  # and back again
        app.processEvents()
        assert win.image.isVisible()
        assert not any(w.isVisible() for w in [win.dark_tab, *hidden])
    finally:
        close_window(win)

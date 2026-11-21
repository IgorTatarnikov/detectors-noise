"""Drive the camera GUI offscreen with the simulated camera, through every tab, then reopen the session."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pg = pytest.importorskip("pyqtgraph")

from basler_camera import SimulatedCamera  # noqa: E402
from basler_live import MainWindow, Step, estimate_seconds  # noqa: E402
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
    assert estimate_seconds(steps) == pytest.approx(11.3)


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

        win.shot_tab.frames.setValue(30)
        win.shot_tab.record()
        wait_idle(win, app)  # lit, then the dark stack is submitted from the callback
        wait_idle(win, app)
        assert win.shot_tab.fit is not None
        assert win.shot_tab.fit["gain_counts_per_e"] == pytest.approx(cam.gain_counts_per_e(12), rel=0.1)

        results = json.loads((tmp_path / "s1" / "results.json").read_text())
        assert results["shot_noise"]["fit"]["read_noise_e"] > 0
        meta = json.loads((tmp_path / "s1" / "shot" / "light1" / "metadata.json").read_text())
        assert meta["gain"] == 12 and meta["black_level"] == 64 and meta["missing_frames"] == 0
        assert (tmp_path / "s1" / "shot" / "light1_dark" / "metadata.json").exists()
    finally:
        close_window(win)

    cam2 = SimulatedCamera(shape=(60, 90), seed=4, max_fps=500)
    win2 = MainWindow(cam2, tmp_path / "s1", analysis_step=1, simulate=True)
    try:
        win2.open_session(tmp_path / "s1")
        assert len(win2.read_tab.by_key) == 2
        assert win2.dark_tab.lookup(12, 64) is not None
        assert len(win2.shot_tab.recordings) == 1
    finally:
        close_window(win2)
    QtWidgets.QApplication.processEvents()


def test_fresh_window_starts_with_a_black_level_above_zero(tmp_path):
    app = pg.mkQApp()
    cam = SimulatedCamera(shape=(40, 60), seed=5, max_fps=500)
    assert cam.get("black_level") == 0
    win = MainWindow(cam, tmp_path / "s", analysis_step=1, simulate=True)
    try:
        deadline = time.monotonic() + 5
        while win.current["black_level"] == 0 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert win.current["black_level"] == 64
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

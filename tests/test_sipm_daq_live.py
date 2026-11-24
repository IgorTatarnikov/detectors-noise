"""Drive the SiPM DAQ GUI offscreen with the simulated detector through every tab, then reopen the session."""

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

from daq_simulator import SimulatedDAQ, SipmModel  # noqa: E402
from live_widgets import close_window  # noqa: E402
from sipm_daq_live import MainWindow  # noqa: E402


def wait_idle(win, app, timeout=60):
    deadline = time.monotonic() + timeout
    while win.current is not None or win.pending:
        app.processEvents()
        time.sleep(0.005)
        if time.monotonic() > deadline:
            raise TimeoutError("recording did not finish")
    for _ in range(5):
        app.processEvents()


def test_full_session_with_simulator(tmp_path):
    app = pg.mkQApp()
    model = SipmModel(height_per_step_mv=80.0)
    daq = SimulatedDAQ(model=model, seed=1)
    win = MainWindow(daq, 7, 100_000, tmp_path / "s1", simulate=True)
    try:
        win.read_tab.seconds.setValue(0.5)
        win.read_tab.record()
        wait_idle(win, app)
        rn = win.read_tab.result
        assert rn is not None and rn.adc_limited
        assert rn.baseline_mv == pytest.approx(model.baseline_mv, abs=3)

        win.dark_tab.bias_edit.setText("4, 8")
        win.dark_tab.seconds.setValue(0.5)
        win.dark_tab.record()
        wait_idle(win, app)
        assert set(win.dark_tab.points) == {4.0, 8.0}
        assert win.dark_tab.points[8.0]["point"].mean > win.dark_tab.points[4.0]["point"].mean

        win.bias.setValue(8)
        for i, light in enumerate([1e6, 3e6]):
            win.sim_light.setValue(light)
            win.gain_tab.label.setText(f"led{i}")
            win.gain_tab.seconds.setValue(1.0)
            win.gain_tab.record()
            wait_idle(win, app)
        fit = win.gain_tab.fit
        assert fit is not None and fit["bias_step"] == 8
        assert fit["h_eff_mv"] == pytest.approx(model.h_eff_mv(8), rel=0.15)

        win.dr_tab.seconds.setValue(0.3)
        win.dr_tab.record()
        wait_idle(win, app)
        assert win.dr_tab.measured == pytest.approx(model.ceiling_mv, abs=10)
        assert all(r["ratio"] > 0 for r in win.dr_tab.rows)

        results = json.loads((tmp_path / "s1" / "results.json").read_text())
        assert results["gain"]["fit"]["h_eff_mv"] == pytest.approx(fit["h_eff_mv"])
        meta = json.loads((tmp_path / "s1" / "dark" / "bias8" / "metadata.json").read_text())
        assert meta["bias_step"] == 8 and meta["samples"] == 50_000
        assert np.load(tmp_path / "s1" / "dark" / "bias8" / "codes.npy").dtype == np.uint16
    finally:
        close_window(win)

    win2 = MainWindow(SimulatedDAQ(seed=2), 7, 100_000, tmp_path / "s1", simulate=True)
    try:
        win2.open_session(tmp_path / "s1")
        assert win2.read_tab.result is not None
        assert set(win2.dark_tab.points) == {4.0, 8.0}
        assert win2.gain_tab.fit["h_eff_mv"] == pytest.approx(fit["h_eff_mv"])
    finally:
        close_window(win2)


def test_starts_with_only_the_live_view_and_its_controls(tmp_path):
    app = pg.mkQApp()
    win = MainWindow(SimulatedDAQ(seed=3), 7, 100_000, tmp_path / "s", simulate=True)
    try:
        win.show()
        app.processEvents()
        assert win.view.titles() == ["Live", "Read noise", "Dark", "Dynamic range", "Gain"]
        assert win.plots.isVisible() and win.bias.isVisible()
        hidden = [win.read_tab, win.progress, win.cancel_button, *win.session_buttons]
        assert not any(w.isVisible() for w in hidden)

        win.view.select("Dark")
        app.processEvents()
        assert win.dark_tab.isVisible() and not win.read_tab.isVisible()
        assert win.plots.isVisible() and win.bias.isVisible()
        assert all(w.isVisible() for w in hidden[1:])
        # both sides get a usable share of the window
        assert min(win.view.splitter.sizes()) > 0.25 * win.width()

        win.view.select("Live")  # and back again
        app.processEvents()
        assert win.plots.isVisible()
        assert not any(w.isVisible() for w in [win.dark_tab, *hidden])
    finally:
        close_window(win)

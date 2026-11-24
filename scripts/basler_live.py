#!/usr/bin/env python
"""Live view and data collection for the camera noise practical, with a Basler USB camera.

    .venv/bin/python scripts/basler_live.py              # first Basler camera found
    .venv/bin/python scripts/basler_live.py --simulate   # no hardware: a simulated camera
    .venv/bin/python scripts/basler_live.py --open data/live/practical   # record into (or re-analyse) this folder

The first tab, Live, is just the live image with black level (bias), gain and exposure
controls, to try the camera out. It records nothing.

The other tabs keep that on the left, with the fraction of pixels at 0 or saturated. The
app reports these numbers but doesn't judge them: a recording that goes wrong, such as read
noise clipped at 0, is left for the students to notice and discuss.

There is one tab per section of the practical. Each tab, on the right, records what it needs into
the session folder (TIFFs plus metadata.json, see recording.py) and shows the result:
  Read noise     covered, minimum exposure, a stack per gain
  Dark counts    covered, a stack per exposure and gain, fit of counts against time
  Dynamic range  from the other two tabs, and optionally a saturated frame
  Shot noise     an unevenly lit stack per gain, bias from Read noise; variance against mean
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import noise_analysis as na
from basler_camera import SETTINGS, camera_metadata, open_camera
from live_widgets import (
    PracticalView,
    color,
    fill_table,
    format_duration,
    parse_numbers,
    plot_widget,
    table,
    unique_path,
)
from recording import (
    RecordingWriter,
    exposure_label,
    find_recordings,
    load_metadata,
    load_recording,
    new_session,
    supersede,
    write_json,
)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

LIVE_EMIT_S = 0.05
MAX_SCATTER = 20_000
# Seconds of frames thrown away after each settings change, and always at least one
# frame. At short exposures an acA4112-20um's offset sat 2 counts away for about 0.5 s
# after a change. Counting frames instead would waste 10 exposures at 10 s each.
SETTLE_S = 1.0


# --------------------------------------------------------------------------- jobs


@dataclass
class Step:
    path: Path
    settings: dict  # any of black_level, gain, exposure_us
    frames: int
    metadata: dict = field(default_factory=dict)
    replace: bool = False  # move an earlier take at this path to superseded/ first


@dataclass
class Job:
    name: str
    steps: list[Step]
    keep_stack: bool = True
    # Record one frame per visit, ramping through the exposures and back (see visit_order)
    interleave: bool = False


@dataclass
class Recording:
    path: Path
    metadata: dict
    stack: np.ndarray | None  # subsampled frames, uint16


def live_stats(image: np.ndarray, max_value: int) -> dict:
    sub = image[::4, ::4]
    return {
        "mean": float(sub.mean()),
        "frac_zero": float(np.mean(sub == 0)),
        "frac_sat": float(np.mean(sub >= max_value)),
    }


def _group(step: Step) -> tuple:
    """Settings other than the exposure: steps that share them are interleaved together."""
    return tuple(sorted((k, v) for k, v in step.settings.items() if k != "exposure_us"))


def visit_order(steps: list[Step]) -> list[tuple[int, int]]:
    """(step index, pass) for each frame of an interleaved recording.

    Steps with the same settings apart from exposure form a group. Each pass records one
    frame at every exposure, shortest first on even passes and longest first on odd ones,
    so the exposures ramp up and back down and a steady drift affects each one equally.
    The exposure at each turn is visited twice in a row, without changing settings.
    """
    order = []
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(steps):
        groups.setdefault(_group(s), []).append(i)
    for members in groups.values():
        members = sorted(members, key=lambda i: steps[i].settings.get("exposure_us", 0))
        for p in range(max(steps[i].frames for i in members)):
            for i in members if p % 2 == 0 else reversed(members):
                if p < steps[i].frames:
                    order.append((i, p))
    return order


def estimate_seconds(steps: list[Step], max_fps: float = 40.0, interleave: bool = False) -> float:
    def period(s: Step) -> float:
        return max(s.settings.get("exposure_us", 0) * 1e-6, 1 / max_fps)

    if not interleave:
        return sum(s.frames * period(s) + max(period(s), SETTLE_S) + 0.3 for s in steps)
    total, last = 0.0, None
    for i, _ in visit_order(steps):
        s = steps[i]
        if last is None or _group(steps[last]) != _group(s):
            total += max(period(s), SETTLE_S) + 0.3  # a new gain: settle fully
        elif last != i:
            total += period(s) + 0.05  # a new exposure: one frame thrown away
        total += period(s)
        last = i
    return total


def setting_key(gain, black_level) -> tuple[float, float]:
    """Results are grouped by gain and black level, as read back from the camera."""
    return round(float(gain), 3), round(float(black_level), 3)


class CameraWorker(QtCore.QThread):
    """Owns the camera. Streams frames for the live view and runs recording jobs."""

    frame = QtCore.pyqtSignal(object, dict)
    settings_changed = QtCore.pyqtSignal(dict)
    progress = QtCore.pyqtSignal(str, int, int)
    recorded = QtCore.pyqtSignal(str, list)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, cam, analysis_step: int):
        super().__init__()
        self.cam = cam
        self.analysis_step = analysis_step
        self.commands: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.running = True
        self.live_missing = 0
        self._last_emit = 0.0
        self._last_id = None
        self._fps_times: list[float] = []

    # called from the GUI thread
    def request(self, *command):
        self.commands.put(command)

    def stop(self):
        self.running = False
        self.cancel.set()
        self.wait(15_000)

    # worker thread
    def _timeout(self) -> float:
        return self.cam.get("exposure_us") * 1e-6 + 2.0

    def _emit_live(self, image, force=False):
        now = time.monotonic()
        self._fps_times = [t for t in self._fps_times if now - t < 2.0] + [now]
        if force or now - self._last_emit >= LIVE_EMIT_S:
            self._last_emit = now
            stats = live_stats(image, self.cam.max_value)
            span = self._fps_times[-1] - self._fps_times[0]
            stats["fps"] = (len(self._fps_times) - 1) / span if span > 0 else math.nan
            stats["missing"] = self.live_missing
            self.frame.emit(image, stats)

    def _handle(self, command):
        kind = command[0]
        if kind == "set":
            _, key, value = command
            self.cam.set(key, value)
            self.settings_changed.emit(self.cam.settings())
        elif kind == "sim":
            _, attr, value = command
            setattr(self.cam, attr, value)
        elif kind == "job":
            self._run_job(command[1])

    def run(self):
        try:
            self.cam.start()
            self.settings_changed.emit(self.cam.settings())
            while self.running:
                try:
                    while True:
                        self._handle(self.commands.get_nowait())
                except queue.Empty:
                    pass
                try:
                    f = self.cam.grab(self._timeout())
                except Exception:  # timeouts and incomplete frames: try again
                    continue
                if self._last_id is not None and f.block_id > self._last_id + 1:
                    self.live_missing += f.block_id - self._last_id - 1
                self._last_id = f.block_id
                self._emit_live(f.image)
        except Exception as e:
            self.failed.emit("", f"Camera error: {e!r}")
        finally:
            try:
                self.cam.close()
            except Exception:
                pass

    def _apply(self, settings: dict):
        self.cam.stop()  # empty the buffers, so no frame from the old settings is kept
        for key in SETTINGS:
            if key in settings:
                self.cam.set(key, settings[key])
        self.settings_changed.emit(self.cam.settings())
        self.cam.start()

    def _settle(self, seconds: float):
        """Throw away frames for `seconds`, and always at least one."""
        start = time.monotonic()
        settled = 0
        while settled == 0 or time.monotonic() - start < seconds:
            if self.cancel.is_set():
                raise InterruptedError
            self.cam.grab(self._timeout())
            settled += 1

    def _grab_one(self):
        while True:
            if self.cancel.is_set():
                raise InterruptedError
            try:
                return self.cam.grab(self._timeout())
            except InterruptedError:
                raise
            except Exception:
                continue

    def _writer(self, step: Step, **options) -> RecordingWriter:
        meta = {**step.metadata, **camera_metadata(self.cam), "analysis_step": self.analysis_step}
        return RecordingWriter(step.path, meta, analysis_step=self.analysis_step, **options)

    def _run_grouped(self, job: Job) -> list[Recording]:
        results = []
        total = sum(s.frames for s in job.steps)
        done = 0
        for step in job.steps:
            self._apply(step.settings)
            self._settle(SETTLE_S)
            writer = self._writer(step)
            try:
                for _ in range(step.frames):
                    f = self._grab_one()
                    writer.add(f.image, f.block_id, f.timestamp)
                    done += 1
                    self._emit_live(f.image)
                    self.progress.emit(f"{job.name}: {step.path.name}", done, total)
            except InterruptedError:
                writer.close(cancelled=True)
                raise
            meta = writer.close(cancelled=False)
            results.append(Recording(step.path, meta, writer.stack() if job.keep_stack else None))
        return results

    def _run_interleaved(self, job: Job) -> list[Recording]:
        order = visit_order(job.steps)
        writers: dict[int, RecordingWriter] = {}
        passes: dict[int, list[int]] = {}
        last = None
        try:
            for done, (i, p) in enumerate(order, start=1):
                step = job.steps[i]
                if last != i:
                    self._apply(step.settings)
                    new_group = last is None or _group(job.steps[last]) != _group(step)
                    self._settle(SETTLE_S if new_group else 0.0)
                if i not in writers:
                    writers[i] = self._writer(step, continuous=False)
                f = self._grab_one()
                writers[i].add(f.image, f.block_id, f.timestamp)
                passes.setdefault(i, []).append(p)
                last = i
                self._emit_live(f.image)
                self.progress.emit(f"{job.name}: {step.path.name}, pass {p + 1}", done, len(order))
        except InterruptedError:
            for i, w in writers.items():
                w.close(cancelled=True, order="interleaved", passes=passes.get(i, []))
            raise
        results = []
        for i, step in enumerate(job.steps):
            meta = writers[i].close(cancelled=False, order="interleaved", passes=passes[i])
            results.append(Recording(step.path, meta, writers[i].stack() if job.keep_stack else None))
        return results

    def _run_job(self, job: Job):
        self.cancel.clear()
        restore = self.cam.settings()
        try:
            results = self._run_interleaved(job) if job.interleave else self._run_grouped(job)
            self.recorded.emit(job.name, results)
        except InterruptedError:
            self.failed.emit(job.name, "Recording cancelled. The partial recording is marked cancelled in its metadata.")
        except Exception as e:
            self.failed.emit(job.name, f"Recording failed: {e!r}")
        finally:
            try:
                self.cam.stop()
                for key in SETTINGS:
                    self.cam.set(key, restore[key])
                self.cam.start()
                self._last_id = None
                self.settings_changed.emit(self.cam.settings())
            except Exception as e:
                self.failed.emit("", f"Could not restore the camera settings: {e!r}")


# --------------------------------------------------------------------------- widgets


class SectionTab(QtWidgets.QWidget):
    """Common layout: a form of inputs, a record button with an estimated time, plots, a table."""

    def __init__(self, win: "MainWindow", intro: str):
        super().__init__()
        self.win = win
        outer = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(intro)
        label.setWordWrap(True)
        outer.addWidget(label)
        self.form = QtWidgets.QFormLayout()
        outer.addLayout(self.form)
        buttons = QtWidgets.QHBoxLayout()
        self.record_button = QtWidgets.QPushButton("Record")
        self.record_button.clicked.connect(self.record)
        self.estimate = QtWidgets.QLabel()
        buttons.addWidget(self.record_button)
        buttons.addWidget(self.estimate, 1)
        outer.addLayout(buttons)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        self.plots = QtWidgets.QHBoxLayout()
        outer.addLayout(self.plots, 3)
        self.body = outer

    def say(self, text: str, warn: bool = False):
        self.message.setText(text)
        self.message.setStyleSheet("color: #c92a2a;" if warn else "")

    def gains(self, edit: QtWidgets.QLineEdit, current_only: QtWidgets.QCheckBox) -> list[float]:
        if current_only.isChecked():
            return [self.win.current["gain"]]
        return parse_numbers(edit.text())

    def update_estimate(self):
        try:
            steps = self.steps()
        except (ValueError, AttributeError):
            self.estimate.setText("")
            return
        self.estimate.setText(f"{len(steps)} stack(s), about {format_duration(estimate_seconds(steps, interleave=self.interleaved()))}")

    def steps(self) -> list[Step]:
        return []

    def interleaved(self) -> bool:
        return False

    def record(self):
        raise NotImplementedError

    def results(self) -> dict:
        return {}


class ReadNoiseTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Cover the camera. Records a stack at the minimum exposure for each gain, at the current "
            "black level. Read noise is the average over pixels of each pixel's standard deviation "
            "across the stack.",
        )
        self.gain_edit = QtWidgets.QLineEdit("0, 1, 12, 24")
        self.current_only = QtWidgets.QCheckBox("current gain only")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.gain_edit)
        row.addWidget(self.current_only)
        self.form.addRow(f"Gains ({win.cam.gain_unit})", row)
        self.frames = QtWidgets.QSpinBox(minimum=2, maximum=10_000, value=100)
        self.form.addRow("Frames per stack", self.frames)
        for w in (self.gain_edit,):
            w.textChanged.connect(self.update_estimate)
        self.current_only.toggled.connect(self.update_estimate)
        self.frames.valueChanged.connect(self.update_estimate)

        self.hist = plot_widget("Per-pixel standard deviation", "standard deviation (counts)", "pixels")
        self.vs_gain = plot_widget("Read noise against gain", f"gain ({win.cam.gain_unit})", "read noise (counts)")
        self.plots.addWidget(self.hist)
        self.plots.addWidget(self.vs_gain)
        self.table = table(["Gain", "Black level", "Bias", "Read noise", "% at 0", "Frames used", "Missing", "Folder"])
        self.body.addWidget(self.table, 2)
        self.by_key: dict[tuple, dict] = {}
        self.update_estimate()

    def steps(self):
        exposure = self.win.cam.limits["exposure_us"][0]
        return [
            Step(
                self.win.session / "read_noise" / f"{g:g}{self.win.cam.gain_unit}",
                {"gain": g, "exposure_us": exposure},
                self.frames.value(),
                {"section": "read_noise"},
                replace=True,
            )
            for g in self.gains(self.gain_edit, self.current_only)
        ]

    def record(self):
        if self.win.prompt("cover", "Cover the camera so no light reaches the sensor."):
            self.win.submit(Job("read_noise", self.steps()), self.on_recorded)

    def on_recorded(self, recordings: list[Recording]):
        for rec in recordings:
            m = rec.metadata
            result = na.read_noise(rec.stack, m["max_value"])
            key = setting_key(m["gain"], m["black_level"])
            self.by_key[key] = {
                "gain": m["gain"],
                "black_level": m["black_level"],
                "bias": result.bias,
                "read_noise": result.read_noise,
                "frac_zero": result.frac_zero,
                "frames": m["frames"],
                "frames_used": result.frames,
                "missing": m["missing_frames"] + m["repeated_frames"],
                "path": str(rec.path),
                "std": result.pixel_std.ravel(),
            }
        self.redraw()
        last = self.by_key[setting_key(recordings[-1].metadata["gain"], recordings[-1].metadata["black_level"])]
        self.say(f"Read noise at {last['gain']:g} {self.win.cam.gain_unit}: {last['read_noise']:.3f} counts.")
        self.win.results_changed()

    def redraw(self):
        rows = sorted(self.by_key.values(), key=lambda r: (r["black_level"], r["gain"]))
        fill_table(
            self.table,
            [
                [r["gain"], r["black_level"], r["bias"], r["read_noise"], 100 * r["frac_zero"], r["frames_used"], r["missing"], Path(r["path"]).name]
                for r in rows
            ],
        )
        self.hist.clear()
        for i, r in enumerate(rows[-4:]):
            counts, edges = np.histogram(r["std"], bins=100)
            self.hist.plot(edges, counts, stepMode="center", pen=pg.mkPen(color(i), width=1.5), name=f"{r['gain']:g} {self.win.cam.gain_unit}")
            self.hist.addItem(pg.InfiniteLine(r["read_noise"], angle=90, pen=pg.mkPen(color(i), style=QtCore.Qt.PenStyle.DashLine)))
        self.vs_gain.clear()
        for i, bl in enumerate(sorted({r["black_level"] for r in rows})):
            sel = sorted((r for r in rows if r["black_level"] == bl), key=lambda r: r["gain"])
            self.vs_gain.plot([r["gain"] for r in sel], [r["read_noise"] for r in sel], pen=color(i), symbol="o", symbolBrush=color(i), name=f"black level {bl:g}")

    def lookup(self, gain, black_level) -> dict | None:
        return self.by_key.get(setting_key(gain, black_level))

    def lookup_nearest(self, gain, black_level) -> dict | None:
        """The result at a requested gain, which the camera may read back as e.g. 11.99 for 12."""
        tol = max(0.1, self.win.cam.limits["gain"][2] or 0.0)  # a camera may report no increment
        near = [r for r in self.by_key.values() if math.isclose(r["black_level"], black_level) and abs(r["gain"] - gain) <= tol]
        return min(near, key=lambda r: abs(r["gain"] - gain)) if near else None

    def results(self):
        return [{k: v for k, v in r.items() if k != "std"} for r in self.by_key.values()]


class DarkTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Cover the camera. Records a stack at each exposure for each gain. The bias is the mean at "
            "the shortest exposure; the slope of bias-subtracted counts against exposure is the dark "
            "count rate in counts/pixel/s.",
        )
        self.gain_edit = QtWidgets.QLineEdit("0, 1, 12, 24")
        self.current_only = QtWidgets.QCheckBox("current gain only")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.gain_edit)
        row.addWidget(self.current_only)
        self.form.addRow(f"Gains ({win.cam.gain_unit})", row)
        self.exposure_edit = QtWidgets.QLineEdit("0.04, 1, 10, 100, 1000")
        self.form.addRow("Exposures (ms)", self.exposure_edit)
        self.frames = QtWidgets.QSpinBox(minimum=1, maximum=10_000, value=10)
        self.form.addRow("Frames per exposure", self.frames)
        self.order = QtWidgets.QComboBox()
        self.order.addItems(
            ["grouped: all frames at one exposure, then the next", "interleaved: one frame per exposure, ramping up and back down"]
        )
        self.order.setToolTip(
            "Interleaved spreads each exposure's frames over the whole recording. If the offset drifts with "
            "time, every exposure sees the same drift; if it depends on the exposure, each exposure reads the "
            "same on every pass. The plot of frame means over time shows which."
        )
        self.order.currentIndexChanged.connect(self.update_estimate)
        self.form.addRow("Order", self.order)
        self.log_x = QtWidgets.QCheckBox("log exposure axis")
        self.form.addRow("", self.log_x)
        for w in (self.gain_edit, self.exposure_edit):
            w.textChanged.connect(self.update_estimate)
        self.current_only.toggled.connect(self.update_estimate)
        self.frames.valueChanged.connect(self.update_estimate)
        self.log_x.toggled.connect(self.redraw)

        self.plot = plot_widget("Dark counts against exposure", "exposure (s)", "mean counts − bias")
        self.plots.addWidget(self.plot)
        over_time = QtWidgets.QVBoxLayout()
        self.series_choice = QtWidgets.QComboBox()
        self.series_choice.currentIndexChanged.connect(self.redraw_series)
        over_time.addWidget(self.series_choice)
        self.time_plot = plot_widget("Frame means over time", "time since first frame (s)", "frame mean (counts)")
        over_time.addWidget(self.time_plot)
        self.plots.addLayout(over_time)
        self.table = table(["Gain", "Black level", "Bias", "Dark (counts/px/s)", "Intercept", "Dark (e⁻/px/s)", "Stacks"])
        self.body.addWidget(self.table, 2)
        # (gain, black_level) -> {exposure_s: (mean, path)}
        self.points: dict[tuple, dict[float, tuple[float, str]]] = {}
        # (gain, black_level) -> {exposure_s: (timestamps in s, frame means)}
        self.series: dict[tuple, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
        self.fits: dict[tuple, na.DarkFit] = {}
        self.update_estimate()

    def exposures_us(self) -> list[float]:
        lo, hi, _ = self.win.cam.limits["exposure_us"]
        return [min(max(ms * 1e3, lo), hi) for ms in parse_numbers(self.exposure_edit.text())]

    def steps(self):
        steps = []
        for g in self.gains(self.gain_edit, self.current_only):
            for us in self.exposures_us():
                steps.append(
                    Step(
                        self.win.session / "dark" / f"{g:g}{self.win.cam.gain_unit}" / exposure_label(us),
                        {"gain": g, "exposure_us": us},
                        self.frames.value(),
                        {"section": "dark"},
                        replace=True,
                    )
                )
        return steps

    def record(self):
        if self.win.prompt("cover", "Cover the camera so no light reaches the sensor."):
            self.win.submit(Job("dark", self.steps(), keep_stack=False, interleave=self.interleaved()), self.on_recorded)

    def interleaved(self) -> bool:
        return self.order.currentIndex() == 1

    def on_recorded(self, recordings: list[Recording]):
        for rec in recordings:
            m = rec.metadata
            key = setting_key(m["gain"], m["black_level"])
            # The mean of the longest run of frames without an offset jump
            means = np.asarray(m["frame_means"])
            steady = na.steady_frames(means)
            self.points.setdefault(key, {})[m["exposure_us"] * 1e-6] = (float(means[steady].mean()), str(rec.path))
            stamps = np.asarray(m.get("timestamps", []), dtype=np.float64) * 1e-9  # camera clock, ns
            if stamps.size == means.size:
                self.series.setdefault(key, {})[m["exposure_us"] * 1e-6] = (stamps, means)
        self.fits = {
            key: na.dark_counts(list(pts), [v[0] for v in pts.values()]) for key, pts in self.points.items() if len(pts) >= 2
        }
        self.refresh_series_choice(setting_key(recordings[-1].metadata["gain"], recordings[-1].metadata["black_level"]))
        self.redraw()
        self.say("Dark count fits updated." if self.fits else "Record at least two exposures to fit a line.")
        self.win.results_changed()

    def redraw(self):
        self.plot.clear()
        self.plot.setLogMode(x=self.log_x.isChecked(), y=False)
        rows = []
        for i, (key, fit) in enumerate(sorted(self.fits.items())):
            gain, bl = key
            label = f"{gain:g} {self.win.cam.gain_unit}, black level {bl:g}"
            x = fit.exposures_s
            self.plot.plot(x, fit.subtracted, pen=None, symbol="o", symbolBrush=color(i), name=label)
            xs = np.geomspace(max(x.min(), 1e-6), x.max(), 50) if self.log_x.isChecked() else np.linspace(0, x.max(), 2)
            self.plot.plot(xs, fit.slope * xs + fit.intercept, pen=pg.mkPen(color(i), style=QtCore.Qt.PenStyle.DashLine))
            k = self.win.shot_tab.gain_for(gain, bl)
            rows.append([gain, bl, fit.bias, fit.slope, fit.intercept, fit.slope / k if k else math.nan, len(x)])
        fill_table(self.table, rows)

    def refresh_series_choice(self, select: tuple | None = None):
        current = select or self.series_choice.currentData()
        keys = sorted(self.series)
        self.series_choice.blockSignals(True)
        self.series_choice.clear()
        for gain, bl in keys:
            self.series_choice.addItem(f"{gain:g} {self.win.cam.gain_unit}, black level {bl:g}", (gain, bl))
        if current in keys:
            self.series_choice.setCurrentIndex(keys.index(current))
        self.series_choice.blockSignals(False)
        self.redraw_series()

    def redraw_series(self):
        """Every frame's mean against when it was taken, coloured by exposure.

        Interleaved: if the offset drifts with time, all exposures move together; if it
        depends on the exposure, each one stays at its own level on every pass.
        """
        self.time_plot.clear()
        series = self.series.get(self.series_choice.currentData(), {})
        if not series:
            return
        t0 = min(t.min() for t, _ in series.values())
        for i, (exposure, (t, means)) in enumerate(sorted(series.items())):
            order = np.argsort(t)
            self.time_plot.plot(
                t[order] - t0, means[order], pen=pg.mkPen(color(i), width=1), symbol="o", symbolSize=5,
                symbolBrush=color(i), symbolPen=None, name=f"{exposure * 1e3:g} ms",
            )

    def lookup(self, gain, black_level) -> na.DarkFit | None:
        return self.fits.get(setting_key(gain, black_level))

    def results(self):
        return [
            {
                "gain": g,
                "black_level": bl,
                "exposures_s": fit.exposures_s,
                "means": fit.means,
                "bias": fit.bias,
                "slope_counts_per_s": fit.slope,
                "intercept": fit.intercept,
            }
            for (g, bl), fit in self.fits.items()
        ]


class DynamicRangeTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Dynamic range = (saturation − dark counts) / read noise, with the bias removed from both. "
            "Uses the Read noise and Dark counts results at the same gain and black level. Saturation is "
            "the largest value the camera can output, or the value measured from a saturated frame.",
        )
        self.record_button.setText("Record saturated frames")
        self.exposure = QtWidgets.QComboBox()
        self.exposure.currentIndexChanged.connect(self.redraw)
        self.form.addRow("Dark counts at exposure", self.exposure)
        self.use_measured = QtWidgets.QCheckBox("use the measured saturation value")
        self.use_measured.toggled.connect(self.redraw)
        self.form.addRow("", self.use_measured)
        self.saturation_label = QtWidgets.QLabel(f"Largest output value: {win.cam.max_value}")
        self.form.addRow("Saturation", self.saturation_label)
        self.measured: float | None = None
        self.plot = plot_widget("Dynamic range against gain", f"gain ({win.cam.gain_unit})", "dynamic range (dB)")
        self.plots.addWidget(self.plot)
        self.table = table(["Gain", "Black level", "Saturation − bias", "Dark", "Read noise", "Dynamic range", "dB"])
        self.body.addWidget(self.table, 2)
        self.rows: list[dict] = []

    def steps(self):
        return [Step(unique_path(self.win.session / "saturation" / "bright"), {}, 5, {"section": "saturation"})]

    def record(self):
        if self.win.prompt("bright", "Shine a bright light on the camera so part of the image saturates."):
            self.win.submit(Job("saturation", self.steps()), self.on_recorded)

    def on_recorded(self, recordings):
        self.measured = float(recordings[-1].stack.max())
        self.saturation_label.setText(f"Largest output value: {self.win.cam.max_value}, measured: {self.measured:g}")
        self.say(f"The brightest pixel read {self.measured:g}.")
        self.use_measured.setChecked(True)
        self.redraw()
        self.win.results_changed()

    def refresh_exposures(self):
        exposures = sorted({t for fit in self.win.dark_tab.fits.values() for t in fit.exposures_s})
        current = self.exposure.currentData()
        self.exposure.blockSignals(True)
        self.exposure.clear()
        for t in exposures:
            self.exposure.addItem(f"{t * 1e3:g} ms", t)
        if exposures:
            target = current if current in exposures else min(exposures, key=lambda t: abs(t - 1e-3))
            self.exposure.setCurrentIndex(exposures.index(target))
        self.exposure.blockSignals(False)

    def redraw(self):
        saturation = self.measured if self.use_measured.isChecked() and self.measured else self.win.cam.max_value
        t = self.exposure.currentData()
        self.rows = []
        for (gain, bl), rn in sorted(self.win.read_tab.by_key.items()):
            fit = self.win.dark_tab.lookup(gain, bl)
            dark = math.nan
            if fit is not None and t is not None:
                idx = np.flatnonzero(np.isclose(fit.exposures_s, t))
                dark = float(fit.subtracted[idx[0]]) if idx.size else math.nan
            ratio, db = na.dynamic_range(saturation, rn["bias"], 0.0 if math.isnan(dark) else dark, rn["read_noise"])
            self.rows.append(
                {"gain": gain, "black_level": bl, "saturation_minus_bias": saturation - rn["bias"], "dark": dark, "read_noise": rn["read_noise"], "ratio": ratio, "db": db, "exposure_s": t}
            )
        fill_table(
            self.table,
            [[r["gain"], r["black_level"], r["saturation_minus_bias"], r["dark"], r["read_noise"], f"{r['ratio']:.0f}:1", r["db"]] for r in self.rows],
        )
        self.plot.clear()
        for i, bl in enumerate(sorted({r["black_level"] for r in self.rows})):
            sel = sorted((r for r in self.rows if r["black_level"] == bl), key=lambda r: r["gain"])
            self.plot.plot([r["gain"] for r in sel], [r["db"] for r in sel], pen=color(i), symbol="o", symbolBrush=color(i), name=f"black level {bl:g}")
        if not self.rows:
            self.say("Record read noise (and dark counts) first.")

    def results(self):
        return {"measured_saturation": self.measured, "rows": self.rows}


class ShotTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Light the camera unevenly, so the image runs from dark to saturated. Records one stack per "
            "gain, with the exposure scaled so the image stays as bright as in the live view at the "
            "current gain. Pixels clipped at 0 or the largest value in any frame are left out, and the "
            f"line is fitted below {na.FIT_FRACTION:.0%} of the mean where the variance peaks (dotted), where wells start to fill. The bias "
            "comes from the Read noise tab, so record read noise at the same gains and black level first. "
            "The slope of variance against mean is the gain in counts/e⁻.",
        )
        self.gain_edit = QtWidgets.QLineEdit("0, 1, 12, 24")
        self.current_only = QtWidgets.QCheckBox("current gain only")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.gain_edit)
        row.addWidget(self.current_only)
        self.form.addRow(f"Gains ({win.cam.gain_unit})", row)
        self.frames = QtWidgets.QSpinBox(minimum=2, maximum=10_000, value=100)
        self.form.addRow("Frames per stack", self.frames)
        self.scale_exposure = QtWidgets.QCheckBox("scale the exposure with gain to keep the same brightness")
        self.scale_exposure.setChecked(True)
        self.form.addRow("", self.scale_exposure)
        self.normalise = QtWidgets.QCheckBox("rescale frames to a common mean (removes light flicker)")
        self.normalise.setChecked(True)
        self.form.addRow("", self.normalise)
        self.log_y = QtWidgets.QCheckBox("log axis for the measured gain plot")
        self.log_y.setChecked(True)
        self.log_y.toggled.connect(self.redraw)
        self.form.addRow("", self.log_y)
        self.notes = QtWidgets.QLineEdit()
        self.notes.setPlaceholderText("e.g. light source, LED current")
        self.form.addRow("Notes", self.notes)
        for w in (self.gain_edit,):
            w.textChanged.connect(self.update_estimate)
        for w in (self.current_only, self.scale_exposure):
            w.toggled.connect(self.update_estimate)
        self.frames.valueChanged.connect(self.update_estimate)

        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(self.redraw)
        self.plot = plot_widget("Variance against mean", "mean counts − bias", "variance (counts²)")
        self.vs_gain = plot_widget("Measured gain", f"gain setting ({win.cam.gain_unit})", "counts/e⁻")
        self.plots.addWidget(self.list, 1)
        self.plots.addWidget(self.plot, 3)
        self.plots.addWidget(self.vs_gain, 2)
        self.table = table(
            ["Gain", "Black level", "Exposure (ms)", "Counts/e⁻", "e⁻/count", "Intercept", "Read noise (e⁻)", "Saturation (e⁻)", "Pixels fitted", "Saturated", "Flicker"]
        )
        self.body.addWidget(self.table, 2)
        # (gain, black_level) -> analysis of the latest stack at that setting
        self.by_key: dict[tuple, dict] = {}
        self.update_estimate()

    def exposure_for(self, gain: float) -> float:
        exposure = self.win.current["exposure_us"]
        if self.scale_exposure.isChecked():
            exposure *= 10 ** ((self.win.current["gain"] - gain) / 20)
        lo, hi, _ = self.win.cam.limits["exposure_us"]
        return min(max(exposure, lo), hi)

    def steps(self):
        return [
            Step(
                self.win.session / "shot" / f"{g:g}{self.win.cam.gain_unit}",
                {"gain": g, "exposure_us": self.exposure_for(g)},
                self.frames.value(),
                {"section": "shot", "notes": self.notes.text()},
                replace=True,
            )
            for g in self.gains(self.gain_edit, self.current_only)
        ]

    def record(self):
        bl = self.win.current["black_level"]
        missing = [g for g in self.gains(self.gain_edit, self.current_only) if self.win.read_tab.lookup_nearest(g, bl) is None]
        if missing:
            gains = ", ".join(f"{g:g}" for g in missing)
            self.say(f"Record read noise at {gains} {self.win.cam.gain_unit} and black level {bl:g} first: it gives the bias.", warn=True)
            return
        if self.win.prompt("light", "Uncover the camera and switch on the light source. The image should run from dark to saturated."):
            self.win.submit(Job("shot", self.steps()), self.on_recorded)

    def on_recorded(self, recordings: list[Recording]):
        skipped = []
        for rec in recordings:
            m = rec.metadata
            rn = self.win.read_tab.lookup_nearest(m["gain"], m["black_level"])
            if rn is None:
                skipped.append(f"{m['gain']:g}")
                continue
            result = na.shot_noise(rec.stack, rn["bias"], m["max_value"], normalise=self.normalise.isChecked())
            k = result.fit.slope
            self.by_key[setting_key(m["gain"], m["black_level"])] = {
                "gain": m["gain"],
                "black_level": m["black_level"],
                "exposure_us": m["exposure_us"],
                "notes": m.get("notes", ""),
                "path": str(rec.path),
                "bias": rn["bias"],
                "gain_counts_per_e": k,
                "intercept": result.fit.intercept,
                "read_noise_e": rn["read_noise"] / k if k > 0 else math.nan,
                "saturation_e": (m["max_value"] - rn["bias"]) / k if k > 0 else math.nan,
                "saturation_mean": result.saturation_mean,
                "fit_max": result.fit_max,
                "pixels": result.fitted,
                "frac_used": result.fitted / (rec.stack[0].size or 1),
                "frac_saturated": result.frac_saturated,
                "frac_zero": result.frac_zero,
                "frame_mean_cv": result.frame_mean_cv,
                "binned": na.binned(result.mean, result.var),
                "sample": self._sample(result),
            }
        self.refresh_list()
        self.redraw()
        if skipped:
            self.say(f"No read noise at {', '.join(skipped)} {self.win.cam.gain_unit} and this black level, so those stacks can't be analysed. Record read noise and open the session again.", warn=True)
        else:
            rows = [self.by_key[setting_key(r.metadata["gain"], r.metadata["black_level"])] for r in recordings]
            worst = max(rows, key=lambda r: r["frame_mean_cv"])
            self.say(f"Frame means varied by up to {100 * worst['frame_mean_cv']:.2f}%.")
        self.win.results_changed()

    @staticmethod
    def _sample(result: na.ShotNoise) -> tuple[np.ndarray, np.ndarray]:
        n = min(MAX_SCATTER, result.pixels)
        idx = np.random.default_rng(0).choice(result.pixels, n, replace=False) if n else np.array([], dtype=int)
        return result.mean[idx], result.var[idx]

    def rows(self) -> list[dict]:
        return sorted(self.by_key.values(), key=lambda r: (r["black_level"], r["gain"]))

    def refresh_list(self):
        hidden = {self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.list.count()) if self.list.item(i).checkState() != QtCore.Qt.CheckState.Checked}
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.rows():
            key = setting_key(r["gain"], r["black_level"])
            item = QtWidgets.QListWidgetItem(f"{r['gain']:g} {self.win.cam.gain_unit} (bl {r['black_level']:g})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked if key in hidden else QtCore.Qt.CheckState.Checked)
            self.list.addItem(item)
        self.list.blockSignals(False)

    def shown(self) -> set:
        return {self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.list.count()) if self.list.item(i).checkState() == QtCore.Qt.CheckState.Checked}

    def redraw(self):
        rows = self.rows()
        fill_table(
            self.table,
            [
                [
                    r["gain"], r["black_level"], r["exposure_us"] / 1e3, r["gain_counts_per_e"], 1 / r["gain_counts_per_e"], r["intercept"],
                    r["read_noise_e"], r["saturation_e"], f"{100 * r['frac_used']:.1f}%", f"{100 * r['frac_saturated']:.1f}%", f"{100 * r['frame_mean_cv']:.2f}%",
                ]
                for r in rows
            ],
        )
        self.plot.clear()
        shown = self.shown()
        for i, r in enumerate(rows):
            if setting_key(r["gain"], r["black_level"]) not in shown:
                continue
            c = color(i)
            mean, var = r["sample"]
            self.plot.plot(mean, var, pen=None, symbol="o", symbolSize=2, symbolPen=None, symbolBrush=pg.mkBrush(c + "30"))
            bx, by = r["binned"]
            self.plot.plot(bx, by, pen=None, symbol="o", symbolSize=6, symbolBrush=c, name=f"{r['gain']:g} {self.win.cam.gain_unit}")
            xmax = r["fit_max"]
            if math.isfinite(xmax) and math.isfinite(r["gain_counts_per_e"]):
                self.plot.plot([0, xmax], [r["intercept"], r["intercept"] + r["gain_counts_per_e"] * xmax], pen=pg.mkPen(c, width=2, style=QtCore.Qt.PenStyle.DashLine))
                self.plot.addItem(pg.InfiniteLine(xmax, angle=90, pen=pg.mkPen(c, style=QtCore.Qt.PenStyle.DotLine)))
        self.vs_gain.clear()
        self.vs_gain.setLogMode(x=False, y=self.log_y.isChecked())
        for i, bl in enumerate(sorted({r["black_level"] for r in rows})):
            sel = [r for r in rows if r["black_level"] == bl and r["gain_counts_per_e"] > 0]
            if not sel:
                continue
            self.vs_gain.plot([r["gain"] for r in sel], [r["gain_counts_per_e"] for r in sel], pen=None, symbol="o", symbolBrush=color(i), name=f"black level {bl:g}")
            # The nominal gain, 10^(dB/20) times the measured gain at the lowest setting
            g0, k0 = sel[0]["gain"], sel[0]["gain_counts_per_e"]
            gs = np.linspace(g0, sel[-1]["gain"], 50)
            self.vs_gain.plot(gs, k0 * 10 ** ((gs - g0) / 20), pen=pg.mkPen(color(i), style=QtCore.Qt.PenStyle.DashLine))
        self.win.dark_tab.redraw()

    def gain_for(self, gain, black_level) -> float | None:
        r = self.by_key.get(setting_key(gain, black_level))
        return r["gain_counts_per_e"] if r and r["gain_counts_per_e"] > 0 else None

    def results(self):
        return [{k: v for k, v in r.items() if k not in ("binned", "sample")} for r in self.rows()]


# --------------------------------------------------------------------------- main window


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cam, session: Path, analysis_step: int, simulate: bool):
        super().__init__()
        self.cam = cam
        self.session = session
        self.simulate = simulate
        self.current = cam.settings()
        self.live_stats: dict = {}
        self._shape = None
        self.callbacks: dict[str, callable] = {}
        self.busy = False
        self.setWindowTitle(f"Camera noise practical: {cam.model} {cam.serial}")
        # Delete the widgets when the window closes, while Python still holds everything they
        # call back into. Left to the garbage collector, Qt can fire signals into half-freed
        # Python objects and segfault.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.worker = CameraWorker(cam, analysis_step)
        self.worker.frame.connect(self.on_frame)
        self.worker.settings_changed.connect(self.on_settings)
        self.worker.progress.connect(self.on_progress)
        self.worker.recorded.connect(self.on_recorded)
        self.worker.failed.connect(self.on_failed)

        live = self._live_panel()
        self.read_tab = ReadNoiseTab(self)
        self.dark_tab = DarkTab(self)
        self.shot_tab = ShotTab(self)
        self.dr_tab = DynamicRangeTab(self)
        bar = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFormat("%v / %m frames")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.worker.cancel.set)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.cancel_button)
        self.view = PracticalView(
            live,
            [
                ("Read noise", self.read_tab),
                ("Dark counts", self.dark_tab),
                ("Dynamic range", self.dr_tab),
                ("Shot noise", self.shot_tab),
            ],
            bar,
            self.session_buttons,
        )
        self.setCentralWidget(self.view)
        self.statusBar().showMessage(f"Session: {session}")
        self.worker.start()

    def _live_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.image = pg.ImageView()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        layout.addWidget(self.image, 1)
        self.auto_levels = QtWidgets.QCheckBox("auto contrast")
        self.auto_levels.setChecked(True)
        layout.addWidget(self.auto_levels)
        self.stats = QtWidgets.QLabel()
        self.stats.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(self.stats)

        form = QtWidgets.QFormLayout()
        lo, hi, inc = self.cam.limits["black_level"]
        self.black = QtWidgets.QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=inc or 1, decimals=0 if (inc or 1) >= 1 else 2)
        lo, hi, inc = self.cam.limits["gain"]
        self.gain = QtWidgets.QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=inc or 0.1, decimals=2)
        lo, hi, _ = self.cam.limits["exposure_us"]
        self.exposure = QtWidgets.QDoubleSpinBox(minimum=lo / 1e3, maximum=hi / 1e3, decimals=3, singleStep=1.0)
        for w, key, scale in ((self.black, "black_level", 1), (self.gain, "gain", 1), (self.exposure, "exposure_us", 1e3)):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, key=key, scale=scale: self.set_setting(key, v * scale))
        form.addRow("Black level (bias)", self.black)
        form.addRow(f"Gain ({self.cam.gain_unit})", self.gain)
        form.addRow("Exposure (ms)", self.exposure)
        layout.addLayout(form)

        if self.simulate:
            box = QtWidgets.QGroupBox("Simulator")
            bl = QtWidgets.QFormLayout(box)
            self.sim_covered = QtWidgets.QCheckBox("camera covered")
            self.sim_covered.setChecked(True)
            self.sim_covered.toggled.connect(lambda v: self.worker.request("sim", "covered", v))
            self.sim_light = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e9, decimals=0, value=2e5, singleStep=1e4)
            self.sim_light.setKeyboardTracking(False)
            self.sim_light.valueChanged.connect(lambda v: self.worker.request("sim", "light", v))
            self.worker.request("sim", "light", 2e5)
            bl.addRow(self.sim_covered)
            bl.addRow("Light (e⁻/s, brightest pixel)", self.sim_light)
            layout.addWidget(box)

        row = QtWidgets.QHBoxLayout()
        new = QtWidgets.QPushButton("New session")
        new.clicked.connect(self.new_session)
        opened = QtWidgets.QPushButton("Open session…")
        opened.clicked.connect(self.open_session_dialog)
        row.addWidget(new)
        row.addWidget(opened)
        layout.addLayout(row)
        self.session_buttons = [new, opened]
        self.controls = [self.black, self.gain, self.exposure, *self.session_buttons]
        return panel

    # ---- live
    def set_setting(self, key, value):
        if not self.busy:
            self.worker.request("set", key, value)

    def on_settings(self, settings: dict):
        self.current = settings
        for w, key, scale in ((self.black, "black_level", 1), (self.gain, "gain", 1), (self.exposure, "exposure_us", 1e3)):
            w.blockSignals(True)
            w.setValue(settings[key] / scale)
            w.blockSignals(False)
        for tab in (self.read_tab, self.dark_tab, self.shot_tab):
            tab.update_estimate()

    def on_frame(self, image, stats):
        self.live_stats = stats
        if self.auto_levels.isChecked():
            lo, hi = np.percentile(image[::8, ::8], (0.5, 99.5))
            levels = (float(lo), float(max(hi, lo + 1)))
        else:
            levels = (0, self.cam.max_value)
        new_shape = image.shape != self._shape
        self._shape = image.shape
        self.image.setImage(image, autoRange=new_shape, autoLevels=False, levels=levels, autoHistogramRange=new_shape)
        self.stats.setText(
            f"mean {stats['mean']:8.1f}   at 0: {100 * stats['frac_zero']:5.2f}%   "
            f"saturated: {100 * stats['frac_sat']:5.2f}%   {stats['fps']:5.1f} fps   missed {stats['missing']}"
        )

    # ---- jobs
    def submit(self, job: Job, callback):
        if self.busy:
            QtWidgets.QMessageBox.information(self, "Busy", "Wait for the current recording to finish.")
            return
        self.busy = True
        for w in self.controls + [t.record_button for t in (self.read_tab, self.dark_tab, self.shot_tab, self.dr_tab)]:
            w.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setMaximum(sum(s.frames for s in job.steps))
        self.progress.setValue(0)
        self.callbacks[job.name] = callback
        for step in job.steps:
            if step.replace:
                supersede(step.path, self.session)
        self.statusBar().showMessage(f"Recording {job.name}, about {format_duration(estimate_seconds(job.steps, interleave=job.interleave))}")
        self.worker.request("job", job)

    def _idle(self):
        self.busy = False
        for w in self.controls + [t.record_button for t in (self.read_tab, self.dark_tab, self.shot_tab, self.dr_tab)]:
            w.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def on_progress(self, text, done, total):
        self.progress.setValue(done)
        self.statusBar().showMessage(text)

    def on_recorded(self, name, recordings):
        self._idle()
        self.statusBar().showMessage(f"Saved {len(recordings)} recording(s) to {self.session}")
        callback = self.callbacks.pop(name, None)
        if callback:
            callback(recordings)

    def on_failed(self, name, message):
        self.callbacks.pop(name, None)
        if name:
            self._idle()
        self.statusBar().showMessage(message)
        QtWidgets.QMessageBox.warning(self, "Camera", message)

    def prompt(self, kind: str, text: str) -> bool:
        if self.simulate:
            covered = kind == "cover"
            self.sim_covered.setChecked(covered)
            if kind == "bright":
                self.sim_light.setValue(1e8)
            elif kind == "light" and self.sim_light.value() >= 1e8:
                self.sim_light.setValue(2e5)
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "Before recording",
            text + "\n\nPress OK when ready.",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    # ---- results and sessions
    def results_changed(self):
        self.dr_tab.refresh_exposures()
        self.dr_tab.redraw()
        self.dark_tab.redraw()
        write_json(
            self.session / "results.json",
            {
                "camera": {"model": self.cam.model, "serial": self.cam.serial, "max_value": self.cam.max_value},
                "read_noise": self.read_tab.results(),
                "dark": self.dark_tab.results(),
                "dynamic_range": self.dr_tab.results(),
                "shot_noise": self.shot_tab.results(),
            },
        )

    def new_session(self):
        self.session = new_session(self.session.parent)
        self.statusBar().showMessage(f"Session: {self.session}")

    def open_session_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open session", str(self.session.parent))
        if path:
            self.open_session(Path(path))

    def open_session(self, session: Path):
        """Load every recording in a session and re-run the analysis. New recordings go there too."""
        self.session = session
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            step = self.worker.analysis_step
            groups: dict[str, list[Recording]] = {}
            for path in find_recordings(session):
                meta = load_metadata(path)
                if meta.get("cancelled"):
                    continue
                section = meta.get("section")
                # Dark counts only need the frame means in metadata.json. Older sessions also
                # hold shot_dark stacks, which the shot noise analysis no longer uses.
                stack = load_recording(path, step)[0] if section in ("read_noise", "shot", "saturation") else None
                groups.setdefault(section, []).append(Recording(path, meta, stack))
            if groups.get("read_noise"):
                self.read_tab.on_recorded(groups["read_noise"])
            if groups.get("dark"):
                self.dark_tab.on_recorded(groups["dark"])
            if groups.get("shot"):
                self.shot_tab.on_recorded(groups["shot"])
            if groups.get("saturation"):
                self.dr_tab.on_recorded(groups["saturation"][-1:])
            self.results_changed()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Session: {session}")

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulate", action="store_true", help="use a simulated camera")
    p.add_argument("--serial", help="camera serial number (default: the first Basler camera)")
    p.add_argument("--root", type=Path, default=Path("data/live"), help="folder for sessions (default: data/live)")
    p.add_argument("--open", type=Path, help="record into this session folder, re-analysing anything already there")
    p.add_argument("--analysis-step", type=int, default=4, help="analyse every Nth pixel in each direction (default: 4)")
    args = p.parse_args(argv)

    app = pg.mkQApp("Camera noise practical")
    try:
        cam = open_camera(args.simulate, args.serial)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Camera", str(e))
        return 1
    if args.open:
        args.open.mkdir(parents=True, exist_ok=True)
    session = args.open or new_session(args.root)
    win = MainWindow(cam, session, args.analysis_step, args.simulate)
    if args.open:
        win.open_session(args.open)
    win.resize(1600, 950)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Live view and data collection for the camera noise practical, with a Basler USB camera.

    .venv/bin/python scripts/basler_live.py              # first Basler camera found
    .venv/bin/python scripts/basler_live.py --simulate   # no hardware: a simulated camera
    .venv/bin/python scripts/basler_live.py --open data/live/practical   # record into (or re-analyse) this folder

Left: the live image with black level (bias), gain and exposure controls, and the
fraction of pixels at 0 or saturated. Keep both at 0% for every recording except
the saturation one.

Right: one tab per section of the practical. Each tab records what it needs into
the session folder (TIFFs plus metadata.json, see recording.py) and shows the result:
  Read noise     covered, minimum exposure, a stack per gain
  Dark counts    covered, a stack per exposure and gain, fit of counts against time
  Dynamic range  from the other two tabs, and optionally a saturated frame
  Shot noise     lit stack plus a covered stack at the same settings; variance against mean
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
# A black level of 0 clips the negative half of the read noise, so start above it.
DEFAULT_BLACK_LEVEL = 64
MAX_SCATTER = 20_000
WARN_FRACTION = 1e-3


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


def estimate_seconds(steps: list[Step], max_fps: float = 40.0) -> float:
    total = 0.0
    for s in steps:
        period = max(s.settings.get("exposure_us", 0) * 1e-6, 1 / max_fps)
        total += (s.frames + 1) * period + 0.3
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

    def _run_job(self, job: Job):
        self.cancel.clear()
        restore = self.cam.settings()
        results = []
        total = sum(s.frames for s in job.steps)
        done = 0
        try:
            for step in job.steps:
                self.cam.stop()  # empty the buffers, so no frame from the old settings is kept
                for key in SETTINGS:
                    if key in step.settings:
                        self.cam.set(key, step.settings[key])
                actual = self.cam.settings()
                self.settings_changed.emit(actual)
                self.cam.start()
                self.cam.grab(self._timeout())  # discard the first frame after a change
                meta = {**step.metadata, **camera_metadata(self.cam), "analysis_step": self.analysis_step}
                writer = RecordingWriter(step.path, meta, analysis_step=self.analysis_step)
                grabbed = 0
                while grabbed < step.frames:
                    if self.cancel.is_set():
                        writer.close(cancelled=True)
                        raise InterruptedError
                    try:
                        f = self.cam.grab(self._timeout())
                    except InterruptedError:
                        raise
                    except Exception:
                        continue
                    writer.add(f.image, f.block_id, f.timestamp)
                    grabbed += 1
                    done += 1
                    self._emit_live(f.image)
                    self.progress.emit(f"{job.name}: {step.path.name}", done, total)
                meta = writer.close(cancelled=False)
                results.append(Recording(step.path, meta, writer.stack() if job.keep_stack else None))
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
        self.estimate.setText(f"{len(steps)} stack(s), about {format_duration(estimate_seconds(steps))}")

    def steps(self) -> list[Step]:
        return []

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
            "across the stack. Keep the black level high enough that no pixel reads 0.",
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
        self.table = table(["Gain", "Black level", "Bias", "Read noise", "% at 0", "Frames", "Missing", "Folder"])
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
                "missing": m["missing_frames"] + m["repeated_frames"],
                "path": str(rec.path),
                "std": na.per_pixel_std(rec.stack).ravel(),
            }
        self.redraw()
        last = self.by_key[setting_key(recordings[-1].metadata["gain"], recordings[-1].metadata["black_level"])]
        clipped = [r for r in self.by_key.values() if r["frac_zero"] > WARN_FRACTION]
        if clipped:
            self.say(
                f"{len(clipped)} stack(s) have pixels at 0, so their read noise is too low. "
                "Raise the black level and record again.",
                warn=True,
            )
        else:
            self.say(f"Read noise at {last['gain']:g} {self.win.cam.gain_unit}: {last['read_noise']:.3f} counts.")
        self.win.results_changed()

    def redraw(self):
        rows = sorted(self.by_key.values(), key=lambda r: (r["black_level"], r["gain"]))
        fill_table(
            self.table,
            [
                [r["gain"], r["black_level"], r["bias"], r["read_noise"], 100 * r["frac_zero"], r["frames"], r["missing"], Path(r["path"]).name]
                for r in rows
            ],
            {i for i, r in enumerate(rows) if r["frac_zero"] > WARN_FRACTION or r["missing"]},
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
        self.exposure_edit = QtWidgets.QLineEdit("0.04, 1, 10, 100, 1000, 10000")
        self.form.addRow("Exposures (ms)", self.exposure_edit)
        self.frames = QtWidgets.QSpinBox(minimum=1, maximum=10_000, value=10)
        self.form.addRow("Frames per exposure", self.frames)
        self.log_x = QtWidgets.QCheckBox("log exposure axis")
        self.form.addRow("", self.log_x)
        for w in (self.gain_edit, self.exposure_edit):
            w.textChanged.connect(self.update_estimate)
        self.current_only.toggled.connect(self.update_estimate)
        self.frames.valueChanged.connect(self.update_estimate)
        self.log_x.toggled.connect(self.redraw)

        self.plot = plot_widget("Dark counts against exposure", "exposure (s)", "mean counts − bias")
        self.plots.addWidget(self.plot)
        self.table = table(["Gain", "Black level", "Bias", "Dark (counts/px/s)", "Intercept", "Dark (e⁻/px/s)", "Stacks"])
        self.body.addWidget(self.table, 2)
        # (gain, black_level) -> {exposure_s: (mean, path)}
        self.points: dict[tuple, dict[float, tuple[float, str]]] = {}
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
            self.win.submit(Job("dark", self.steps(), keep_stack=False), self.on_recorded)

    def on_recorded(self, recordings: list[Recording]):
        for rec in recordings:
            m = rec.metadata
            key = setting_key(m["gain"], m["black_level"])
            self.points.setdefault(key, {})[m["exposure_us"] * 1e-6] = (float(np.mean(m["frame_means"])), str(rec.path))
        self.fits = {
            key: na.dark_counts(list(pts), [v[0] for v in pts.values()]) for key, pts in self.points.items() if len(pts) >= 2
        }
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
        if self.measured < self.win.cam.max_value:
            self.say("No pixel reached the largest output value. Use a brighter light or a longer exposure.", warn=True)
        else:
            self.say(f"Saturated pixels read {self.measured:g}.")
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
            "Light the camera with a steady source at the current settings, with no pixel saturated. "
            "Records the lit stack, then asks you to cover the camera for a dark stack at identical "
            "settings, which removes the bias and fixed pattern. The slope of variance against mean is the "
            "gain in counts/e⁻. Tick several recordings at the same gain to fit them together.",
        )
        self.record_button.setText("Record lit + dark")
        self.label = QtWidgets.QLineEdit("light1")
        self.form.addRow("Label", self.label)
        self.notes = QtWidgets.QLineEdit()
        self.notes.setPlaceholderText("e.g. LED current, sample")
        self.form.addRow("Notes", self.notes)
        self.frames = QtWidgets.QSpinBox(minimum=2, maximum=10_000, value=100)
        self.frames.valueChanged.connect(self.update_estimate)
        self.form.addRow("Frames per stack", self.frames)
        self.normalise = QtWidgets.QCheckBox("rescale frames to a common mean (removes light flicker)")
        self.normalise.setChecked(True)
        self.form.addRow("", self.normalise)

        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(self.redraw)
        self.plot = plot_widget("Variance against mean", "mean counts (dark subtracted)", "variance (counts²)")
        self.plots.addWidget(self.list, 1)
        self.plots.addWidget(self.plot, 3)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.addWidget(self.summary)
        self.recordings: list[dict] = []
        self.fit: dict | None = None
        self._lit: Recording | None = None
        self.update_estimate()

    def steps(self):
        label = self.label.text().strip() or "light"
        return [Step(unique_path(self.win.session / "shot" / label), {}, self.frames.value(), {"section": "shot", "label": label, "notes": self.notes.text()})]

    def record(self):
        if self.win.live_stats.get("frac_sat", 0) > 0:
            self.say("Some pixels are saturated. Dim the light or shorten the exposure first.", warn=True)
            return
        if self.win.prompt("light", "Uncover the camera and switch on the light source."):
            self.win.submit(Job("shot_lit", self.steps()), self.on_lit)

    def on_lit(self, recordings):
        self._lit = recordings[-1]
        lit_path = self._lit.path
        if not self.win.prompt("cover", "Now cover the camera, keeping every setting the same."):
            self.say("Dark stack skipped, so this recording can't be analysed.", warn=True)
            return
        dark = Step(
            lit_path.with_name(lit_path.name + "_dark"),
            {},
            self.frames.value(),
            {"section": "shot_dark", "label": self._lit.metadata["label"], "pair": str(lit_path)},
        )
        self.win.submit(Job("shot_dark", [dark]), self.on_dark)

    def on_dark(self, recordings):
        dark = recordings[-1]
        lit, self._lit = self._lit, None
        keys = ("gain", "black_level", "exposure_us")
        if any(not math.isclose(lit.metadata[k], dark.metadata[k]) for k in keys):
            self.say("The settings changed between the lit and dark stacks, so they can't be paired.", warn=True)
            return
        self.add_pair(lit, dark)

    def add_pair(self, lit: Recording, dark: Recording):
        result = na.shot_noise(lit.stack, dark.stack, normalise=self.normalise.isChecked())
        m = lit.metadata
        entry = {
            "label": m.get("label", lit.path.name),
            "path": str(lit.path),
            "dark_path": str(dark.path),
            "gain": m["gain"],
            "black_level": m["black_level"],
            "exposure_us": m["exposure_us"],
            "notes": m.get("notes", ""),
            "slope": result.fit.slope,
            "intercept": result.fit.intercept,
            "frame_mean_cv": result.frame_mean_cv,
            "max_mean": float(result.mean.max()),
            "saturated": float(np.mean(lit.stack >= m["max_value"])),
            "mean": result.mean,
            "var": result.var,
        }
        self.recordings.append(entry)
        item = QtWidgets.QListWidgetItem(
            f"{entry['label']}  ({entry['gain']:g} {self.win.cam.gain_unit}, {entry['exposure_us'] / 1e3:g} ms): slope {entry['slope']:.4g}"
        )
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.CheckState.Checked)
        self.list.blockSignals(True)
        self.list.addItem(item)
        self.list.blockSignals(False)
        msg = f"Frame means varied by {100 * entry['frame_mean_cv']:.2f}%."
        if entry["saturated"] > 0:
            self.say(msg + " Some pixels saturated, which bends the line at high means.", warn=True)
        else:
            self.say(msg)
        self.redraw()
        self.win.results_changed()

    def selected(self) -> list[dict]:
        return [r for i, r in enumerate(self.recordings) if self.list.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def redraw(self):
        self.plot.clear()
        chosen = self.selected()
        self.fit = None
        if not chosen:
            self.summary.setText("")
            return
        gain, bl = chosen[0]["gain"], chosen[0]["black_level"]
        same = [r for r in chosen if setting_key(r["gain"], r["black_level"]) == setting_key(gain, bl)]
        rng = np.random.default_rng(0)
        for i, r in enumerate(same):
            idx = rng.choice(r["mean"].size, min(MAX_SCATTER, r["mean"].size), replace=False)
            self.plot.plot(r["mean"][idx], r["var"][idx], pen=None, symbol="o", symbolSize=3, symbolPen=None, symbolBrush=pg.mkBrush(color(i) + "60"), name=r["label"])
        fit = na.pooled_fit([(r["mean"], r["var"]) for r in same])
        xmax = max(r["max_mean"] for r in same)
        self.plot.plot([0, xmax], [fit.intercept, fit.intercept + fit.slope * xmax], pen=pg.mkPen("#0b0b0b", width=2, style=QtCore.Qt.PenStyle.DashLine), name="fit")
        rn = self.win.read_tab.lookup(gain, bl)
        dark = self.win.dark_tab.lookup(gain, bl)
        k = fit.slope
        self.fit = {"gain": gain, "black_level": bl, "gain_counts_per_e": k, "intercept": fit.intercept, "recordings": [r["label"] for r in same]}
        lines = [
            f"<b>Gain at {gain:g} {self.win.cam.gain_unit}: {k:.4g} counts/e⁻</b> ({1 / k:.3g} e⁻/count), from {len(same)} recording(s).",
            f"Intercept {fit.intercept:.4g} counts², implying read noise {math.sqrt(fit.intercept):.3g} counts" if fit.intercept > 0 else f"Intercept {fit.intercept:.4g} counts² (negative: the extrapolation to 0 is unreliable)",
        ]
        if rn:
            lines.append(f"Read noise measured directly: {rn['read_noise']:.3g} counts = <b>{rn['read_noise'] / k:.3g} e⁻</b>; saturation {(self.win.cam.max_value - rn['bias']) / k:.4g} e⁻.")
            self.fit.update(read_noise_e=rn["read_noise"] / k, saturation_e=(self.win.cam.max_value - rn["bias"]) / k)
        if dark:
            lines.append(f"Dark counts: {dark.slope:.3g} counts/px/s = <b>{dark.slope / k:.3g} e⁻/px/s</b>.")
            self.fit.update(dark_e_per_s=dark.slope / k)
        if len(same) < len(chosen):
            lines.append("Recordings at a different gain or black level are left out of the fit.")
        self.summary.setText("<br>".join(lines))
        self.win.dark_tab.redraw()

    def gain_for(self, gain, black_level) -> float | None:
        if self.fit and setting_key(self.fit["gain"], self.fit["black_level"]) == setting_key(gain, black_level):
            return self.fit["gain_counts_per_e"]
        return None

    def results(self):
        return {
            "fit": self.fit,
            "recordings": [{k: v for k, v in r.items() if k not in ("mean", "var")} for r in self.recordings],
        }


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

        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        splitter.addWidget(self._live_panel())
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        self.tabs = QtWidgets.QTabWidget()
        self.read_tab = ReadNoiseTab(self)
        self.dark_tab = DarkTab(self)
        self.shot_tab = ShotTab(self)
        self.dr_tab = DynamicRangeTab(self)
        self.tabs.addTab(self.read_tab, "Read noise")
        self.tabs.addTab(self.dark_tab, "Dark counts")
        self.tabs.addTab(self.dr_tab, "Dynamic range")
        self.tabs.addTab(self.shot_tab, "Shot noise")
        rl.addWidget(self.tabs)
        bar = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFormat("%v / %m frames")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.worker.cancel.set)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.cancel_button)
        rl.addLayout(bar)
        splitter.addWidget(right)
        splitter.setSizes([700, 900])
        self.statusBar().showMessage(f"Session: {session}")
        if self.current["black_level"] <= 0:
            lo, hi, _ = cam.limits["black_level"]
            self.worker.request("set", "black_level", min(max(DEFAULT_BLACK_LEVEL, lo), hi))
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
        self.controls = [self.black, self.gain, self.exposure, new, opened]
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
        warn = stats["frac_zero"] > WARN_FRACTION or stats["frac_sat"] > WARN_FRACTION
        self.stats.setStyleSheet("color: #c92a2a;" if warn else "")
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
        self.statusBar().showMessage(f"Recording {job.name}, about {format_duration(estimate_seconds(job.steps))}")
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
                stack = None if section == "dark" else load_recording(path, step)[0]
                groups.setdefault(section, []).append(Recording(path, meta, stack))
            if groups.get("read_noise"):
                self.read_tab.on_recorded(groups["read_noise"])
            if groups.get("dark"):
                self.dark_tab.on_recorded(groups["dark"])
            darks = {r.metadata.get("pair"): r for r in groups.get("shot_dark", [])}
            for lit in groups.get("shot", []):
                if str(lit.path) in darks:
                    self.shot_tab.add_pair(lit, darks[str(lit.path)])
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

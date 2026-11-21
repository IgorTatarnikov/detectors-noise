#!/usr/bin/env python
"""Live view and data collection for the noise practical with a SiPM on the USB-201 DAQ.

    .venv/bin/python scripts/sipm_daq_live.py                 # Ch7, 100 kS/s
    .venv/bin/python scripts/sipm_daq_live.py --channel 0
    .venv/bin/python scripts/sipm_daq_live.py --simulate      # no hardware
    .venv/bin/python scripts/sipm_daq_live.py --open data/sipm_live/practical

Left: the live trace and 10 ms block means, with the fraction of samples above the
baseline code, how many of those are only one code tall, and the fraction clipped.
The bias knob can't be read by the computer, so set "Bias step" to match it; every
recording stores it.

Right: one tab per section of the practical. Each recording is saved as codes.npy
plus metadata.json under the session folder (see sipm_daq_analysis.py for the method):
  Read noise     detector unplugged: the baseline and the spread of single samples
  Dark           SiPM covered, at each bias step: mean and variance of the samples
  Dynamic range  from the other tabs, and optionally the output's ceiling in bright light
  Gain           several light levels at one bias step: the slope of variance against
                 mean is the effective single-photon pulse height in mV
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import sipm_daq_analysis as sa
from daq_live import BlockAverager, Reader, RingBuffer
from live_widgets import color, fill_table, format_duration, parse_numbers, plot_widget, table, unique_path
from recording import new_session, write_json
from usb201 import MAX_RATE

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

DEFAULT_ROOT = Path("data/sipm_live")
SETTLE_S = 0.3  # samples dropped after each prompt, while hands leave the bench
TRACE_S = 0.5
HISTORY_S = 30.0
WARN_FRACTION = 1e-3


# --------------------------------------------------------------------------- recordings


@dataclass
class Step:
    path: Path
    seconds: float
    metadata: dict
    prompt: tuple[str, str] | None = None  # (kind, text) asked before this step
    bias_step: float | None = None  # ask for the knob to be set to this first


@dataclass
class DaqRecording:
    path: Path
    metadata: dict
    codes: np.ndarray
    mv: np.ndarray


def save_recording(path: Path, codes: np.ndarray, metadata: dict) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "codes.npy", codes.astype(np.uint16))
    metadata = {**metadata, "samples": int(codes.size), "saved": datetime.now().isoformat(timespec="seconds")}
    write_json(path / "metadata.json", metadata)
    return metadata


def load_daq_recording(path: Path, to_mv) -> DaqRecording:
    meta = json.loads((path / "metadata.json").read_text())
    codes = np.load(path / "codes.npy")
    return DaqRecording(path, meta, codes, to_mv(codes, meta))


def live_stats(codes: np.ndarray, mv: np.ndarray, baseline_code: float | None) -> dict:
    if codes.size == 0:
        return {}
    if baseline_code is None:
        values, counts = np.unique(codes, return_counts=True)
        baseline_code = float(values[np.argmax(counts)])
    above, single = sa.hit_fractions(codes, baseline_code)
    return {
        "mean": float(mv.mean()),
        "sd": float(mv.std()),
        "max": float(mv.max()),
        "above": above,
        "single": single,
        "clipped": sa.clipped_fraction(codes),
    }


# --------------------------------------------------------------------------- tabs


class SectionTab(QtWidgets.QWidget):
    def __init__(self, win: "MainWindow", intro: str, button: str = "Record"):
        super().__init__()
        self.win = win
        outer = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(intro)
        label.setWordWrap(True)
        outer.addWidget(label)
        self.form = QtWidgets.QFormLayout()
        outer.addLayout(self.form)
        row = QtWidgets.QHBoxLayout()
        self.record_button = QtWidgets.QPushButton(button)
        self.record_button.clicked.connect(self.record)
        self.estimate = QtWidgets.QLabel()
        row.addWidget(self.record_button)
        row.addWidget(self.estimate, 1)
        outer.addLayout(row)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        self.plots = QtWidgets.QHBoxLayout()
        outer.addLayout(self.plots, 3)
        self.body = outer

    def say(self, text: str, warn: bool = False):
        self.message.setText(text)
        self.message.setStyleSheet("color: #c92a2a;" if warn else "")

    def seconds_box(self, value: float) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox(minimum=0.1, maximum=3600, decimals=1, value=value, suffix=" s")
        box.valueChanged.connect(self.update_estimate)
        return box

    def update_estimate(self):
        steps = self.steps()
        if steps:
            total = sum(s.seconds + SETTLE_S for s in steps)
            self.estimate.setText(f"{len(steps)} recording(s), about {format_duration(total)}")

    def steps(self) -> list[Step]:
        return []

    def record(self):
        raise NotImplementedError

    def need_read(self) -> bool:
        if self.win.read_tab.result is None:
            self.say("Record read noise first: the baseline comes from it.", warn=True)
            return False
        return True


class ReadTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Disconnect the detector from the DAQ input, leaving any terminator in place. The baseline is the "
            "mean, and read noise is the spread of single samples within 10 ms blocks. If that spread is below "
            "the rounding noise of one ADC code (code/√12), the DAQ can't resolve it and the rounding noise is "
            "used instead.",
        )
        self.seconds = self.seconds_box(5.0)
        self.form.addRow("Duration", self.seconds)
        self.hist = plot_widget("Sample values", "mV", "samples")
        self.plots.addWidget(self.hist)
        self.table = table(["Baseline (mV)", "Baseline code", "SD (mV)", "Read noise used (mV)", "Below ADC step", "% clipped", "Folder"])
        self.body.addWidget(self.table, 1)
        self.result: sa.ReadNoise | None = None
        self.path: str | None = None
        self.update_estimate()

    def steps(self):
        return [
            Step(
                unique_path(self.win.session / "read" / "unplugged"),
                self.seconds.value(),
                {"section": "read"},
                prompt=("unplug", "Disconnect the detector from the DAQ input, leaving any terminator in place."),
            )
        ]

    def record(self):
        self.win.run_steps(self.steps(), self.on_recorded)

    def on_recorded(self, recordings: list[DaqRecording]):
        rec = recordings[-1]
        self.result = sa.read_noise(rec.mv, rec.codes, self.win.block(rec.metadata["rate"]), rec.metadata["lsb_mv"])
        self.path = str(rec.path)
        r = self.result
        fill_table(self.table, [[r.baseline_mv, r.baseline_code, r.sd_mv, r.effective_sd_mv, "yes" if r.adc_limited else "no", 100 * r.clipped_fraction, rec.path.name]])
        self.hist.clear()
        lsb = rec.metadata["lsb_mv"]
        edges = np.arange(rec.mv.min() - lsb / 2, rec.mv.max() + lsb, lsb)
        counts, edges = np.histogram(rec.mv, bins=edges)
        self.hist.plot(edges, counts, stepMode="center", fillLevel=0, brush=pg.mkBrush(color(0) + "80"), pen=color(0))
        if r.adc_limited:
            self.say(
                f"The noise ({r.sd_mv:.3g} mV) is below the ADC step ({lsb:.2f} mV per code), so nearly every sample "
                f"reads the same code. Read noise is taken as code/√12 = {r.effective_sd_mv:.3g} mV."
            )
        else:
            self.say(f"Read noise {r.sd_mv:.3g} mV about a baseline of {r.baseline_mv:.2f} mV.")
        self.win.results_changed()

    def results(self):
        if self.result is None:
            return None
        return {**self.result.__dict__, "path": self.path}


class DarkTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Reconnect the detector and cover the SiPM, with no source. Records at each bias step, asking you to "
            "turn the knob between them. The mean is the dark signal in mV (the integrated signal per second); "
            "variance/mean estimates the pulse height, but only the Gain tab's fit over light levels separates it "
            "from the read noise.",
        )
        self.bias_edit = QtWidgets.QLineEdit("3, 5, 7, 8, 9, 10")
        self.bias_edit.textChanged.connect(self.update_estimate)
        self.form.addRow("Bias steps", self.bias_edit)
        self.seconds = self.seconds_box(10.0)
        self.form.addRow("Duration per step", self.seconds)
        self.vs_bias = plot_widget("Dark signal against bias step", "bias step", "mean − baseline (mV)")
        self.vs_bias.setLogMode(y=True)
        self.windows = plot_widget("Integrated dark signal against exposure window", "window (s)", "integral (mV·s)")
        self.windows.setLogMode(x=True, y=True)
        self.plots.addWidget(self.vs_bias)
        self.plots.addWidget(self.windows)
        self.table = table(["Bias", "Mean (mV)", "Var (mV²)", "Var/mean (mV)", "% above baseline", "% 1-code hits", "% clipped", "Pulses/s"])
        self.body.addWidget(self.table, 2)
        self.points: dict[float, dict] = {}
        self.update_estimate()

    def steps(self):
        try:
            biases = parse_numbers(self.bias_edit.text())
        except ValueError:
            return []
        steps = []
        for i, b in enumerate(biases):
            steps.append(
                Step(
                    unique_path(self.win.session / "dark" / f"bias{b:g}"),
                    self.seconds.value(),
                    {"section": "dark"},
                    prompt=("cover", "Reconnect the detector and cover the SiPM, with no source.") if i == 0 else None,
                    bias_step=b,
                )
            )
        return steps

    def record(self):
        if self.need_read():
            self.win.run_steps(self.steps(), self.on_recorded)

    def on_recorded(self, recordings: list[DaqRecording]):
        rn = self.win.read_tab.result
        for rec in recordings:
            m = rec.metadata
            point = sa.signal_point(rec.mv, rec.codes, rn.baseline_mv, rn.baseline_code, self.win.block(m["rate"]))
            windows = sa.window_integrals(rec.mv - rn.baseline_mv, m["rate"], [1e-3, 1e-2, 0.1, 1.0])
            self.points[float(m["bias_step"])] = {"point": point, "windows": windows, "path": str(rec.path)}
            self.win.gain_tab.add_point(rec, point, "dark")
        self.redraw()
        warn = [b for b, d in self.points.items() if d["point"].adc_limited or d["point"].clipped_fraction > WARN_FRACTION]
        if warn:
            self.say(
                "At bias step(s) " + ", ".join(f"{b:g}" for b in sorted(warn)) + " most pulses are only one ADC code "
                "tall, or the output clipped, so the mean and variance there are not reliable.",
                warn=True,
            )
        else:
            self.say("Dark recordings analysed.")
        self.win.results_changed()

    def redraw(self):
        self.vs_bias.clear()
        self.windows.clear()
        rows = []
        biases = sorted(self.points)
        pos = [(b, self.points[b]["point"].mean) for b in biases if self.points[b]["point"].mean > 0]
        if pos:
            self.vs_bias.plot([b for b, _ in pos], [m for _, m in pos], pen=color(0), symbol="o", symbolBrush=color(0))
        for i, b in enumerate(biases):
            d = self.points[b]
            p = d["point"]
            w = [(t, mean) for t, mean, _ in d["windows"] if mean > 0]
            if w:
                self.windows.plot([t for t, _ in w], [m for _, m in w], pen=color(i), symbol="o", symbolBrush=color(i), name=f"bias {b:g}")
            rate = self.win.gain_tab.rate_for(b, p.mean)
            rows.append([b, p.mean, p.var, p.var / p.mean if p.mean > 0 else math.nan, 100 * p.above_fraction, 100 * p.single_code_fraction, 100 * p.clipped_fraction, rate])
        fill_table(self.table, rows, {i for i, b in enumerate(biases) if self.points[b]["point"].adc_limited})

    def results(self):
        return [{"bias_step": b, **d["point"].__dict__, "windows": d["windows"], "path": d["path"]} for b, d in sorted(self.points.items())]


class DynamicRangeTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Dynamic range = (saturation − dark) / read noise, per sample, with the baseline removed. Saturation is "
            "the ADC's full scale (+10 V) unless you record the output's ceiling: shine a bright light so the "
            "detector's output stops rising.",
            button="Record saturation",
        )
        self.seconds = self.seconds_box(2.0)
        self.form.addRow("Duration", self.seconds)
        self.use_measured = QtWidgets.QCheckBox("use the measured ceiling")
        self.use_measured.toggled.connect(self.redraw)
        self.form.addRow("", self.use_measured)
        self.sat_label = QtWidgets.QLabel("ADC full scale: 10000 mV")
        self.form.addRow("Saturation", self.sat_label)
        self.plot = plot_widget("Dynamic range against bias step", "bias step", "dynamic range (dB)")
        self.plots.addWidget(self.plot)
        self.table = table(["Bias", "Saturation − baseline (mV)", "Dark (mV)", "Read noise (mV)", "Dynamic range", "dB"])
        self.body.addWidget(self.table, 2)
        self.measured: float | None = None
        self.rows: list[dict] = []
        self.update_estimate()

    def steps(self):
        return [
            Step(
                unique_path(self.win.session / "saturation" / f"bias{self.win.bias.value():g}"),
                self.seconds.value(),
                {"section": "saturation"},
                prompt=("bright", "Shine a bright light on the SiPM so its output stops rising."),
            )
        ]

    def record(self):
        self.win.run_steps(self.steps(), self.on_recorded)

    def on_recorded(self, recordings):
        rec = recordings[-1]
        self.measured = sa.saturation_level(rec.mv)
        self.sat_label.setText(f"ADC full scale: 10000 mV, measured ceiling: {self.measured:.0f} mV")
        self.use_measured.setChecked(True)
        self.say(f"The output sat at {self.measured:.0f} mV (99.9th percentile of {rec.mv.size} samples).")
        self.redraw()
        self.win.results_changed()

    def redraw(self):
        rn = self.win.read_tab.result
        self.rows = []
        self.plot.clear()
        if rn is None:
            fill_table(self.table, [])
            return
        saturation = self.measured if self.use_measured.isChecked() and self.measured else 10_000.0
        for b, d in sorted(self.win.dark_tab.points.items()):
            ratio, db = sa.dynamic_range(saturation, rn.baseline_mv, d["point"].mean, rn.effective_sd_mv)
            self.rows.append({"bias_step": b, "saturation_minus_baseline": saturation - rn.baseline_mv, "dark_mv": d["point"].mean, "read_noise_mv": rn.effective_sd_mv, "ratio": ratio, "db": db})
        if not self.rows:
            self.rows.append({"bias_step": math.nan, "saturation_minus_baseline": saturation - rn.baseline_mv, "dark_mv": 0.0, "read_noise_mv": rn.effective_sd_mv, **dict(zip(("ratio", "db"), sa.dynamic_range(saturation, rn.baseline_mv, 0.0, rn.effective_sd_mv)))})
        fill_table(self.table, [[r["bias_step"], r["saturation_minus_baseline"], r["dark_mv"], r["read_noise_mv"], f"{r['ratio']:.0f}:1", r["db"]] for r in self.rows])
        pts = [r for r in self.rows if not math.isnan(r["bias_step"])]
        if pts:
            self.plot.plot([r["bias_step"] for r in pts], [r["db"] for r in pts], pen=color(0), symbol="o", symbolBrush=color(0))

    def results(self):
        return {"measured_ceiling_mv": self.measured, "rows": self.rows}


class GainTab(SectionTab):
    def __init__(self, win):
        super().__init__(
            win,
            "Keep the bias knob fixed and record several light levels (LED current, or the source at different "
            "distances). Each recording gives a mean and variance of the samples; dark recordings at the same "
            "bias are added automatically. The slope of variance against mean is the effective single-photon "
            "pulse height H_eff in mV, and the intercept should match the read noise variance. With the pulse "
            "width from one scope capture, mean ÷ (H_eff × width) gives pulses per second.",
            button="Record light level",
        )
        self.label = QtWidgets.QLineEdit("light1")
        self.form.addRow("Label", self.label)
        self.notes = QtWidgets.QLineEdit()
        self.notes.setPlaceholderText("e.g. LED current, source distance")
        self.form.addRow("Notes", self.notes)
        self.seconds = self.seconds_box(10.0)
        self.form.addRow("Duration", self.seconds)
        self.width_ns = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e6, decimals=1, value=0, suffix=" ns", specialValueText="unknown")
        self.width_ns.valueChanged.connect(self.redraw)
        self.form.addRow("Pulse width τ_eff (from scope)", self.width_ns)
        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(self.redraw)
        self.plot = plot_widget("Variance against mean", "mean − baseline (mV)", "variance (mV²)")
        self.plots.addWidget(self.list, 1)
        self.plots.addWidget(self.plot, 3)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.addWidget(self.summary)
        self.entries: list[dict] = []
        self.fit: dict | None = None
        self.update_estimate()

    def steps(self):
        label = self.label.text().strip() or "light"
        return [
            Step(
                unique_path(self.win.session / "light" / f"bias{self.win.bias.value():g}_{label}"),
                self.seconds.value(),
                {"section": "light", "label": label, "notes": self.notes.text()},
                prompt=("light", f"Set the light level for '{label}', keeping the bias knob at {self.win.bias.value():g}."),
            )
        ]

    def record(self):
        if self.need_read():
            self.win.run_steps(self.steps(), self.on_recorded)

    def on_recorded(self, recordings):
        rn = self.win.read_tab.result
        for rec in recordings:
            point = sa.signal_point(rec.mv, rec.codes, rn.baseline_mv, rn.baseline_code, self.win.block(rec.metadata["rate"]))
            self.add_point(rec, point, rec.metadata.get("label", rec.path.name))
        self.win.results_changed()

    def add_point(self, rec: DaqRecording, point: sa.SignalPoint, label: str):
        bias = float(rec.metadata["bias_step"])
        self.entries = [e for e in self.entries if e["path"] != str(rec.path)]
        self.entries.append({"label": label, "bias_step": bias, "point": point, "path": str(rec.path), "notes": rec.metadata.get("notes", "")})
        self.list.blockSignals(True)
        self.list.clear()
        for e in self.entries:
            p = e["point"]
            item = QtWidgets.QListWidgetItem(f"bias {e['bias_step']:g}  {e['label']}: mean {p.mean:.3g} mV, var {p.var:.3g} mV²" + ("  ⚠ ADC" if p.adc_limited else ""))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if e["bias_step"] == bias else QtCore.Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.redraw()

    def selected(self):
        return [e for i, e in enumerate(self.entries) if self.list.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def redraw(self):
        self.plot.clear()
        self.fit = None
        chosen = self.selected()
        if not chosen:
            self.summary.setText("")
            return
        bias = chosen[-1]["bias_step"]
        same = [e for e in chosen if e["bias_step"] == bias]
        xs = [e["point"].mean for e in same]
        ys = [e["point"].var for e in same]
        self.plot.plot(xs, ys, pen=None, symbol="o", symbolBrush=color(0), name=f"bias {bias:g}")
        for e in same:
            text = pg.TextItem(e["label"], color="#52514e", anchor=(0, 1))
            text.setPos(e["point"].mean, e["point"].var)
            self.plot.addItem(text)
        lines = []
        if len(same) >= 2:
            fit = sa.gain_fit([e["point"] for e in same])
            x = np.array([0, max(xs)])
            self.plot.plot(x, fit.intercept + fit.slope * x, pen=pg.mkPen("#0b0b0b", width=2, style=QtCore.Qt.PenStyle.DashLine), name="fit")
            rn = self.win.read_tab.result
            self.fit = {"bias_step": bias, "h_eff_mv": fit.slope, "intercept_mv2": fit.intercept, "points": [e["label"] for e in same]}
            lines.append(f"<b>Bias step {bias:g}: H_eff = {fit.slope:.4g} mV per photon</b> (slope), from {len(same)} recordings.")
            if rn is not None:
                lines.append(f"Intercept {fit.intercept:.3g} mV², compared with read noise² = {rn.effective_sd_mv ** 2:.3g} mV².")
            width = self.width_ns.value() * 1e-9
            if width > 0 and fit.slope > 0:
                rates = ", ".join(f"{e['label']} {sa.pulse_rate(e['point'].mean, fit.slope, width):.3g}/s" for e in same)
                lines.append(f"Pulse rates with τ_eff = {width * 1e9:g} ns: {rates}.")
                self.fit["pulse_width_s"] = width
        else:
            lines.append("Record at least two light levels (or dark plus one) at this bias step to fit a line.")
        bad = [e["label"] for e in same if e["point"].adc_limited]
        if bad:
            lines.append(f"<span style='color:#c92a2a'>Most pulses are one ADC code tall in: {', '.join(bad)}. Rounding biases these points; use a higher bias step.</span>")
        clipped = [e["label"] for e in same if e["point"].clipped_fraction > WARN_FRACTION]
        if clipped:
            lines.append(f"<span style='color:#c92a2a'>The output clipped in: {', '.join(clipped)}.</span>")
        if len(same) < len(chosen):
            lines.append("Recordings at other bias steps are left out of the fit.")
        self.summary.setText("<br>".join(lines))
        self.win.dark_tab.redraw()

    def rate_for(self, bias: float, mean_mv: float) -> float:
        if self.fit and self.fit["bias_step"] == bias and self.fit.get("pulse_width_s") and self.fit["h_eff_mv"] > 0:
            return sa.pulse_rate(mean_mv, self.fit["h_eff_mv"], self.fit["pulse_width_s"])
        return math.nan

    def results(self):
        return {"fit": self.fit, "points": [{"label": e["label"], "bias_step": e["bias_step"], "path": e["path"], "notes": e["notes"], **e["point"].__dict__} for e in self.entries]}


# --------------------------------------------------------------------------- main window


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, daq, channel: int, rate: float, session: Path, simulate: bool):
        super().__init__()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.daq = daq
        self.channel = channel
        self.simulate = simulate
        self.session = session
        self.reader = Reader(daq, channel, rate)
        self.rate = self.reader.actual_rate
        self.serial = daq.serial()
        # One code in mV, from this channel's factory calibration (nominally 20 V / 4096).
        self.lsb_mv = daq.cal[channel][0] * 20_000 / 4096
        self.trace = RingBuffer(int(TRACE_S * self.rate))
        self.trace_codes = RingBuffer(int(TRACE_S * self.rate))
        self.averager = BlockAverager(self.block(self.rate))
        self.history = RingBuffer(int(HISTORY_S / (self.block(self.rate) / self.rate)))
        # recording state
        self.pending: list[Step] = []
        self.done: list[DaqRecording] = []
        self.callback = None
        self.current: Step | None = None
        self.buffer: list[np.ndarray] = []
        self.collected = 0
        self.skip = 0
        self.restarts = 0
        self.setWindowTitle(f"SiPM noise practical: USB-201 {self.serial} Ch{channel} @ {self.rate:.0f} S/s")

        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        splitter.addWidget(self._live_panel())
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        self.tabs = QtWidgets.QTabWidget()
        self.read_tab = ReadTab(self)
        self.gain_tab = GainTab(self)
        self.dark_tab = DarkTab(self)
        self.dr_tab = DynamicRangeTab(self)
        self.tabs.addTab(self.read_tab, "Read noise")
        self.tabs.addTab(self.dark_tab, "Dark")
        self.tabs.addTab(self.dr_tab, "Dynamic range")
        self.tabs.addTab(self.gain_tab, "Gain")
        rl.addWidget(self.tabs)
        bar = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        bar.addWidget(self.progress, 1)
        bar.addWidget(self.cancel_button)
        rl.addLayout(bar)
        splitter.addWidget(right)
        splitter.setSizes([700, 900])
        self.statusBar().showMessage(f"Session: {session}")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(33)
        QtCore.QTimer.singleShot(0, self.reader.start)

    def block(self, rate: float) -> int:
        return sa.block_samples(rate)

    def to_mv(self, codes: np.ndarray, meta: dict | None = None) -> np.ndarray:
        if meta is not None:
            slope, intercept = meta["calibration"]
            from usb201 import codes_to_volts

            return codes_to_volts(codes, slope, intercept) * 1e3
        return self.daq.to_volts(codes, self.channel) * 1e3

    def _live_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        plots = pg.GraphicsLayoutWidget()
        plots.setBackground("w")
        self.p_trace = plots.addPlot(row=0, col=0, title=f"Trace, last {TRACE_S:g} s")
        self.p_trace.setLabels(left="mV", bottom="time (s)")
        self.p_trace.showGrid(x=True, y=True, alpha=0.25)
        self.p_trace.setDownsampling(auto=True, mode="peak")
        self.p_trace.setClipToView(True)
        self.c_trace = self.p_trace.plot(pen=pg.mkPen(color(0), width=1))
        self.p_hist = plots.addPlot(row=1, col=0, title=f"Mean per {1e3 * sa.DEFAULT_BLOCK_S:g} ms block, last {HISTORY_S:g} s")
        self.p_hist.setLabels(left="mV", bottom="time (s)")
        self.p_hist.showGrid(x=True, y=True, alpha=0.25)
        self.c_hist = self.p_hist.plot(pen=pg.mkPen(color(1), width=1.5))
        layout.addWidget(plots, 1)
        self.stats = QtWidgets.QLabel()
        self.stats.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)
        form = QtWidgets.QFormLayout()
        self.bias = QtWidgets.QDoubleSpinBox(minimum=0, maximum=10, decimals=1, value=5, singleStep=1)
        self.bias.valueChanged.connect(self.on_bias)
        form.addRow("Bias step (the knob)", self.bias)
        form.addRow("Input", QtWidgets.QLabel(f"Ch{self.channel}, {self.rate:.0f} S/s, {self.lsb_mv:.2f} mV per code"))
        layout.addLayout(form)
        if self.simulate:
            box = QtWidgets.QGroupBox("Simulator")
            bl = QtWidgets.QFormLayout(box)
            self.sim_connected = QtWidgets.QCheckBox("detector connected")
            self.sim_connected.setChecked(True)
            self.sim_connected.toggled.connect(lambda v: setattr(self.daq, "connected", v))
            self.sim_covered = QtWidgets.QCheckBox("SiPM covered")
            self.sim_covered.setChecked(True)
            self.sim_covered.toggled.connect(lambda v: setattr(self.daq, "covered", v))
            self.sim_light = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e11, decimals=0, value=2e5, singleStep=5e4)
            self.sim_light.valueChanged.connect(lambda v: setattr(self.daq, "light", v))
            bl.addRow(self.sim_connected)
            bl.addRow(self.sim_covered)
            bl.addRow("Light (photons/s)", self.sim_light)
            layout.addWidget(box)
            self.on_bias(self.bias.value())
        row = QtWidgets.QHBoxLayout()
        new = QtWidgets.QPushButton("New session")
        new.clicked.connect(self.new_session)
        opened = QtWidgets.QPushButton("Open session…")
        opened.clicked.connect(self.open_session_dialog)
        row.addWidget(new)
        row.addWidget(opened)
        layout.addLayout(row)
        self.controls = [self.bias, new, opened]
        return panel

    def on_bias(self, value):
        if self.simulate:
            self.daq.bias_step = value

    # ---- acquisition
    def poll(self):
        new = []
        while True:
            try:
                item = self.reader.out.get_nowait()
            except queue.Empty:
                break
            if item is None:  # overrun: a gap in the data
                self.averager._pending = np.empty(0)
                if self.current is not None:
                    self.restarts += 1
                    self.buffer, self.collected = [], 0
                    self.statusBar().showMessage(f"Data gap (USB overrun): restarting this recording ({self.restarts})")
                continue
            new.append(item)
        if self.reader.error is not None:
            self.timer.stop()
            QtWidgets.QMessageBox.critical(self, "DAQ", f"DAQ error: {self.reader.error!r}")
            return
        if new:
            codes = np.concatenate(new)
            mv = self.to_mv(codes)
            self.trace.write(mv)
            self.trace_codes.write(codes)
            self.history.write(self.averager.push(mv))
            if self.current is not None:
                self._collect(codes)
        if self.trace.filled:
            y = self.trace.latest()
            self.c_trace.setData(np.arange(-y.size, 0) / self.rate, y)
            h = self.history.latest()
            if h.size:
                self.c_hist.setData(np.arange(-h.size, 0) * self.block(self.rate) / self.rate, h)
            rn = self.read_tab.result
            s = live_stats(self.trace_codes.latest().astype(np.int64), y, rn.baseline_code if rn else None)
            warn = s["clipped"] > WARN_FRACTION or (s["above"] > 0 and s["single"] > sa.MAX_SINGLE_CODE_FRACTION)
            self.stats.setStyleSheet("color: #c92a2a;" if warn else "")
            self.stats.setText(
                f"mean {s['mean']:8.3f} mV   sd {s['sd']:7.3f} mV   max {s['max']:8.1f} mV\n"
                f"above baseline {100 * s['above']:6.2f}%   of which 1 code {100 * s['single']:5.1f}%   "
                f"clipped {100 * s['clipped']:5.2f}%   overruns {self.reader.overruns}"
            )

    def _collect(self, codes):
        if self.skip > 0:
            drop = min(self.skip, codes.size)
            self.skip -= drop
            codes = codes[drop:]
        need = self.target - self.collected
        if need > 0 and codes.size:
            take = codes[:need]
            self.buffer.append(take)
            self.collected += take.size
            self.progress.setValue(self.collected)
        if self.collected >= self.target:
            self._finish_step()

    # ---- jobs
    def run_steps(self, steps: list[Step], callback):
        if self.current is not None or self.pending:
            return
        self.pending = list(steps)
        self.done = []
        self.callback = callback
        for w in self.controls + [t.record_button for t in (self.read_tab, self.dark_tab, self.dr_tab, self.gain_tab)]:
            w.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._start_next()

    def _start_next(self):
        if not self.pending:
            self._idle()
            self.statusBar().showMessage(f"Saved {len(self.done)} recording(s) to {self.session}")
            callback, self.callback = self.callback, None
            if callback and self.done:
                callback(self.done)
            return
        step = self.pending.pop(0)
        if step.prompt and not self.prompt(*step.prompt):
            self.cancel()
            return
        if step.bias_step is not None:
            if not self.prompt("bias", f"Turn the bias knob to {step.bias_step:g}.", step.bias_step):
                self.cancel()
                return
            self.bias.blockSignals(True)
            self.bias.setValue(step.bias_step)
            self.bias.blockSignals(False)
            self.on_bias(step.bias_step)
        self.current = step
        self.target = int(round(step.seconds * self.rate))
        self.buffer, self.collected, self.restarts = [], 0, 0
        self.skip = int(SETTLE_S * self.rate)
        self.progress.setMaximum(self.target)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"Recording {step.path.relative_to(self.session)} for {step.seconds:g} s")

    def _finish_step(self):
        step, self.current = self.current, None
        codes = np.concatenate(self.buffer)[: self.target]
        self.buffer = []
        slope, intercept = getattr(self.daq, "cal")[self.channel]
        meta = save_recording(
            step.path,
            codes,
            {
                **step.metadata,
                "bias_step": float(self.bias.value()),
                "channel": self.channel,
                "rate": self.rate,
                "calibration": [slope, intercept],
                "lsb_mv": self.lsb_mv,
                "daq_serial": self.serial,
                "restarts_after_overrun": self.restarts,
                "simulated": self.simulate,
            },
        )
        self.done.append(DaqRecording(step.path, meta, codes, self.to_mv(codes, meta)))
        self._start_next()

    def cancel(self):
        self.pending = []
        self.current = None
        self.buffer = []
        self.callback = None
        self._idle()
        self.statusBar().showMessage("Recording cancelled; nothing saved for the unfinished step.")

    def _idle(self):
        for w in self.controls + [t.record_button for t in (self.read_tab, self.dark_tab, self.dr_tab, self.gain_tab)]:
            w.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def prompt(self, kind: str, text: str, value=None) -> bool:
        if self.simulate:
            if kind == "unplug":
                self.sim_connected.setChecked(False)
            elif kind == "cover":
                self.sim_connected.setChecked(True)
                self.sim_covered.setChecked(True)
            elif kind == "light":
                self.sim_connected.setChecked(True)
                self.sim_covered.setChecked(False)
                if self.sim_light.value() >= 1e10:
                    self.sim_light.setValue(2e5)
            elif kind == "bright":
                self.sim_connected.setChecked(True)
                self.sim_covered.setChecked(False)
                self.sim_light.setValue(1e10)
            elif kind == "bias":
                self.bias.setValue(value)
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
        self.dark_tab.redraw()
        self.dr_tab.redraw()
        write_json(
            self.session / "results.json",
            {
                "daq": {"serial": self.serial, "channel": self.channel, "rate": self.rate},
                "read_noise": self.read_tab.results(),
                "dark": self.dark_tab.results(),
                "dynamic_range": self.dr_tab.results(),
                "gain": self.gain_tab.results(),
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
        self.session = session
        groups: dict[str, list[DaqRecording]] = {}
        for meta_path in sorted(session.rglob("metadata.json")):
            if not (meta_path.parent / "codes.npy").exists():
                continue
            rec = load_daq_recording(meta_path.parent, self.to_mv)
            groups.setdefault(rec.metadata.get("section"), []).append(rec)
        if groups.get("read"):
            self.read_tab.on_recorded(groups["read"][-1:])
        if self.read_tab.result is not None:
            if groups.get("dark"):
                self.dark_tab.on_recorded(groups["dark"])
            if groups.get("light"):
                self.gain_tab.on_recorded(groups["light"])
        if groups.get("saturation"):
            self.dr_tab.on_recorded(groups["saturation"][-1:])
        self.results_changed()
        self.statusBar().showMessage(f"Session: {session}")

    def closeEvent(self, event):
        self.timer.stop()
        self.reader.stop()
        try:
            self.daq.close()
        except Exception:
            pass
        super().closeEvent(event)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", type=int, default=7, choices=range(8))
    p.add_argument("--rate", type=float, default=MAX_RATE, help="samples/s (max 100000)")
    p.add_argument("--simulate", action="store_true", help="use a simulated SiPM and DAQ")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"folder for sessions (default: {DEFAULT_ROOT})")
    p.add_argument("--open", type=Path, help="record into this session folder, re-analysing anything already there")
    args = p.parse_args(argv)

    app = pg.mkQApp("SiPM noise practical")
    try:
        if args.simulate:
            from daq_simulator import SimulatedDAQ

            daq = SimulatedDAQ()
        else:
            from usb201 import USB201

            daq = USB201()
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "DAQ", str(e))
        return 1
    if args.open:
        args.open.mkdir(parents=True, exist_ok=True)
    session = args.open or new_session(args.root)
    win = MainWindow(daq, args.channel, args.rate, session, args.simulate)
    if args.open:
        win.open_session(args.open)
    win.resize(1600, 950)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

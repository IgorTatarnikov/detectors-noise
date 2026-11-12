#!/usr/bin/env python
"""Live view of a continuous USB-201 scan.

    .venv/bin/python scripts/daq_live.py                   # Ch7, 100 kS/s
    .venv/bin/python scripts/daq_live.py --channel 0 --rate 50000 --window 0.5

Top: the raw trace over the last --window seconds.
Bottom: the mean of each --block seconds, over the last --history seconds.
Space pauses the display (acquisition keeps running). Drag or scroll to zoom; right-click for options.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np

from usb201 import LSB_VOLTS, MAX_RATE, ScanOverrun, USB201, pacer_period


class RingBuffer:
    """Fixed-size buffer keeping the most recent samples."""

    def __init__(self, size: int):
        self.data = np.zeros(size)
        self.size = size
        self.filled = 0
        self._pos = 0

    def write(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if x.size >= self.size:
            self.data[:] = x[-self.size :]
            self._pos, self.filled = 0, self.size
            return
        end = self._pos + x.size
        if end <= self.size:
            self.data[self._pos : end] = x
        else:
            split = self.size - self._pos
            self.data[self._pos :] = x[:split]
            self.data[: end - self.size] = x[split:]
        self._pos = end % self.size
        self.filled = min(self.size, self.filled + x.size)

    def latest(self) -> np.ndarray:
        """Stored samples, oldest first."""
        if self.filled < self.size:
            return self.data[: self.filled].copy()
        return np.roll(self.data, -self._pos)


class BlockAverager:
    """Splits a stream into blocks of n samples and returns each block's mean."""

    def __init__(self, n: int):
        self.n = n
        self._pending = np.empty(0)

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._pending, np.asarray(x, dtype=float)])
        whole = x.size // self.n * self.n
        self._pending = x[whole:]
        return x[:whole].reshape(-1, self.n).mean(axis=1)


class Reader(threading.Thread):
    """Drains the device in the background so its small FIFO never overflows."""

    def __init__(self, daq: USB201, channel: int, rate: float):
        super().__init__(daemon=True)
        self.daq, self.channel, self.rate = daq, channel, rate
        self.out: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.overruns = 0
        self.error: Exception | None = None
        self.actual_rate = pacer_period(rate)[1]

    def run(self):
        try:
            # Start the scan here, not in __init__: the FIFO overflows within tens of ms
            # if nothing is reading it.
            self.daq.start_continuous(self.channel, self.rate)
            while not self.stop_event.is_set():
                try:
                    codes = self.daq.read_available(max_bytes=4096, timeout_ms=200)
                except ScanOverrun:
                    self.overruns += 1
                    self.out.put(None)  # marks a gap in the data
                    self.daq.start_continuous(self.channel, self.rate)
                    continue
                if codes.size:
                    self.out.put(codes)
        except Exception as e:  # surface USB errors in the GUI instead of dying silently
            self.error = e
        finally:
            try:
                self.daq.stop_scan()
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", type=int, default=7, choices=range(8))
    p.add_argument("--rate", type=float, default=MAX_RATE, help="samples/s (max 100000)")
    p.add_argument("--window", type=float, default=1.0, help="seconds of raw trace shown")
    p.add_argument("--block", type=float, default=0.01, help="seconds per averaged point")
    p.add_argument("--history", type=float, default=30.0, help="seconds of block means shown")
    args = p.parse_args(argv)

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    daq = USB201()
    reader = Reader(daq, args.channel, args.rate)
    rate = reader.actual_rate
    trace = RingBuffer(max(1, int(args.window * rate)))
    block_n = max(1, int(round(args.block * rate)))
    averager = BlockAverager(block_n)
    means = RingBuffer(max(1, int(args.history / (block_n / rate))))
    to_mv = lambda codes: daq.to_volts(codes, args.channel) * 1e3
    state = {"paused": False, "received": 0, "t0": None, "first": 0}

    app = pg.mkQApp("USB-201 live")
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"USB-201 Ch{args.channel} @ {rate:.0f} S/s")
    layout = QtWidgets.QVBoxLayout(win)
    status = QtWidgets.QLabel()
    status.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    layout.addWidget(status)
    plots = pg.GraphicsLayoutWidget()
    layout.addWidget(plots)

    p_trace = plots.addPlot(row=0, col=0, title=f"Raw trace, last {args.window:g} s")
    p_trace.setLabels(left="mV", bottom="time (s)")
    p_trace.showGrid(x=True, y=True, alpha=0.3)
    p_trace.setDownsampling(auto=True, mode="peak")
    p_trace.setClipToView(True)
    c_trace = p_trace.plot(pen=pg.mkPen("y", width=1))

    p_mean = plots.addPlot(row=1, col=0, title=f"Mean per {block_n / rate * 1e3:g} ms block, last {args.history:g} s")
    p_mean.setLabels(left="mV", bottom="time (s)")
    p_mean.showGrid(x=True, y=True, alpha=0.3)
    c_mean = p_mean.plot(pen=pg.mkPen("c", width=1.5))

    def update():
        new = []
        while True:
            try:
                item = reader.out.get_nowait()
            except queue.Empty:
                break
            if item is None:  # overrun: restart the block so no mean spans the gap
                averager._pending = np.empty(0)
                continue
            new.append(item)
        if new:
            mv = to_mv(np.concatenate(new))
            if state["t0"] is None:
                state["t0"], state["first"] = time.monotonic(), mv.size
            state["received"] += mv.size
            trace.write(mv)
            means.write(averager.push(mv))

        if reader.error is not None:
            status.setText(f"ERROR: {reader.error!r}")
            timer.stop()
            return
        if state["paused"] or trace.filled == 0:
            return

        y = trace.latest()
        c_trace.setData(np.arange(-y.size, 0) / rate, y)
        m = means.latest()
        if m.size:
            c_mean.setData(np.arange(-m.size, 0) * (block_n / rate), m)
        elapsed = time.monotonic() - state["t0"]
        # the first batch arrived at t0, so leave it out of the rate
        recv_rate = (state["received"] - state["first"]) / elapsed if elapsed > 0.5 else float("nan")
        status.setText(
            f"window mean {y.mean():8.3f} mV   sd {y.std():6.3f} mV   "
            f"min {y.min():7.2f}   max {y.max():7.2f} mV   |   "
            f"received {recv_rate:7.0f} S/s   "
            f"overruns {reader.overruns}   1 code = {LSB_VOLTS * 1e3:.2f} mV"
        )

    def on_key(event):
        if event.key() == QtCore.Qt.Key.Key_Space:
            state["paused"] = not state["paused"]
            win.setWindowTitle(
                f"USB-201 Ch{args.channel} @ {rate:.0f} S/s" + ("  [PAUSED]" if state["paused"] else "")
            )

    win.keyPressEvent = on_key
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(33)

    win.resize(1100, 750)
    win.show()
    QtCore.QTimer.singleShot(0, reader.start)  # once the event loop is running
    try:
        app.exec()
    finally:
        reader.stop()
        daq.close()


if __name__ == "__main__":
    sys.exit(main())

"""Widgets and helpers shared by the live acquisition apps (basler_live.py, sipm_daq_live.py)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

PALETTE = ["#2a78d6", "#eb6834", "#2f9e44", "#9c36b5", "#c92a2a", "#1098ad", "#e8590c", "#5c940d"]


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, sec = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {sec:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def parse_numbers(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if x.strip()]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while (candidate := path.with_name(f"{path.name}_{n}")).exists():
        n += 1
    return candidate


def table(headers: list[str]) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    return t


def fill_table(t: QtWidgets.QTableWidget, rows: list[list], warn_rows: set[int] | None = None):
    t.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            item = QtWidgets.QTableWidgetItem(value if isinstance(value, str) else fmt(value))
            if warn_rows and r in warn_rows:
                item.setForeground(QtGui.QColor("#c92a2a"))
            t.setItem(r, c, item)


def fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def plot_widget(title: str, x: str, y: str) -> pg.PlotWidget:
    w = pg.PlotWidget(title=title)
    w.setBackground("w")
    w.showGrid(x=True, y=True, alpha=0.25)
    w.setLabel("bottom", x)
    w.setLabel("left", y)
    w.addLegend(offset=(10, 10))
    return w


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def close_window(win: QtWidgets.QWidget) -> None:
    """Close a window and run its deletion now, not at some later garbage collection."""
    win.close()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

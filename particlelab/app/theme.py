"""Light and dark colour themes for the desktop application.

``current()`` returns the active palette (a dict of colours); widgets read it when
they draw, and ``apply(app, window, name)`` restyles Qt widgets and every pyqtgraph
plot, then asks the views to redraw.
"""
from __future__ import annotations

import re

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

THEMES = {
    "light": dict(
        name="light", bg="#ffffff", fg="#000000", muted="#555555", grid="#dddddd",
        detector="#bbbbbb", detector_xy="#cccccc", hit="#111111", missed="#999999", flash="#ff9800",
        crossing="#7b1fa2", vertex="#2ca02c",
        note_bg="#fff8e1", note_border="#e0c060", note_fg="#000000",
        reco={"μ⁻": "#1f77b4", "μ⁺": "#d62728"}, true={"μ⁻": "#9cc3e6", "μ⁺": "#f0a5a6"},
        ok="#008800", bad="#bb0000"),
    "dark": dict(
        name="dark", bg="#1e1f22", fg="#e6e6e6", muted="#a9adb5", grid="#3a3d42",
        detector="#5d6470", detector_xy="#4a505a", hit="#f5f5f5", missed="#8a8f98", flash="#ffb74d",
        crossing="#ce93d8", vertex="#66d17a",
        note_bg="#3b3420", note_border="#8a7530", note_fg="#f3e9c6",
        reco={"μ⁻": "#5aa9ff", "μ⁺": "#ff6b6b"}, true={"μ⁻": "#2f5e8f", "μ⁺": "#8f3b3b"},
        ok="#5fd068", bad="#ff7070"),
}
_current = THEMES["light"]
_light_style = None
_light_palette = None


def current() -> dict:
    return _current


def _dark_palette():
    t = THEMES["dark"]
    pal = QtGui.QPalette()
    c = QtGui.QColor
    roles = QtGui.QPalette.ColorRole
    pal.setColor(roles.Window, c("#26282c")); pal.setColor(roles.WindowText, c(t["fg"]))
    pal.setColor(roles.Base, c("#1b1c1f")); pal.setColor(roles.AlternateBase, c("#2a2c30"))
    pal.setColor(roles.Text, c(t["fg"])); pal.setColor(roles.Button, c("#2f3237"))
    pal.setColor(roles.ButtonText, c(t["fg"])); pal.setColor(roles.ToolTipBase, c("#2f3237"))
    pal.setColor(roles.ToolTipText, c(t["fg"])); pal.setColor(roles.Highlight, c("#3d6fb6"))
    pal.setColor(roles.HighlightedText, c("#ffffff")); pal.setColor(roles.PlaceholderText, c("#80858d"))
    pal.setColor(roles.Link, c("#7fb2ff"))
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        pal.setColor(QtGui.QPalette.ColorGroup.Disabled, role, c("#6b7078"))
    return pal


def style_plot(w: pg.PlotWidget, t: dict | None = None):
    t = t or _current
    w.setBackground(t["bg"])
    item = w.getPlotItem()
    for name in ("left", "bottom", "right", "top"):
        ax = item.getAxis(name)
        ax.setPen(t["fg"]); ax.setTextPen(t["fg"])
    if item.titleLabel.text:
        item.setTitle(item.titleLabel.text, color=t["fg"])
    if item.legend is not None:
        item.legend.setLabelTextColor(t["fg"])


def apply(app: QtWidgets.QApplication, window: QtWidgets.QWidget | None, name: str):
    """Switch the whole application to theme ``name`` ("light" or "dark")."""
    global _current, _light_style, _light_palette
    if _light_style is None:
        _light_style = app.style().name()
        _light_palette = QtGui.QPalette(app.palette())
    _current = THEMES[name]
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):
        # Qt ≥ 6.8: the native style switches itself; without this the "light" theme would
        # inherit a dark system setting and put white plots on dark panels
        hints.setColorScheme(QtCore.Qt.ColorScheme.Dark if name == "dark" else QtCore.Qt.ColorScheme.Light)
    elif name == "dark":
        app.setStyle("Fusion")
        app.setPalette(_dark_palette())
    else:
        app.setStyle(_light_style)
        app.setPalette(_light_palette)
    pg.setConfigOptions(background=_current["bg"], foreground=_current["fg"])
    if window is not None:
        for w in window.findChildren(QtWidgets.QWidget):
            # a widget with its own style sheet keeps its old text colour unless told
            base = w.property("theme_base_sheet")
            if base is None:
                sheet = w.styleSheet()
                if not sheet or re.search(r"(^|[;\s])color\s*:", sheet):
                    continue           # no sheet, or the widget sets its colour itself
                base = sheet
                w.setProperty("theme_base_sheet", base)
            w.setStyleSheet(f"{base.rstrip('; ')}; color: {_current['fg']};")
        for w in window.findChildren(pg.PlotWidget):
            style_plot(w)
        for w in window.findChildren(QtWidgets.QWidget):
            hook = getattr(w, "theme_changed", None)
            if callable(hook):
                hook()

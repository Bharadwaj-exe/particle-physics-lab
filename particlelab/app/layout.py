"""Sizing and plot-view helpers shared by the GUI.

- ``em``: lengths in units of the font height, so the layout scales with the font
  and with the display scaling instead of being fixed in pixels.
- ``histogram_view`` / ``curve_view`` / ``geometry_view``: plot views that cannot be
  dragged or zoomed into an unreadable state. Histograms and curves zoom and pan in x
  only, with y always fitted to what is visible; geometry views zoom both axes inside
  the detector. Double-clicking a plot (or its 'A' button) restores the default view.
- ``splitter``: panes the user can resize; MainWindow saves their sizes. Responsive
  splitters put their panes side by side when there is room and one above the other
  when there is not.
- ``min_height_for_width``: the height a widget really needs at a given width,
  including word-wrapped text (Qt's minimum size ignores it, which clips text).
- ``log_ticks``: readable 1-2-5 ticks for logarithmic axes.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

VIEW_TIP = "Wheel or right-drag: zoom · drag: pan · double-click: reset the view"


def em(widget: QtWidgets.QWidget | None = None, n: float = 1.0) -> int:
    """n times the height of the font of ``widget`` (or of the application), in pixels."""
    fm = widget.fontMetrics() if widget is not None else QtWidgets.QApplication.fontMetrics()
    return int(round(n * fm.height()))


def text_width(widget: QtWidgets.QWidget, text: str) -> int:
    return widget.fontMetrics().horizontalAdvance(text)


def min_size(widget: QtWidgets.QWidget) -> QtCore.QSize:
    return widget.minimumSizeHint().expandedTo(widget.minimumSize())


def min_height_for_width(widget: QtWidgets.QWidget, width: int) -> int:
    """Minimum height of ``widget`` at ``width``, counting word-wrapped labels at that width."""
    base = min_size(widget).height()
    lay = widget.layout()
    if lay is not None and lay.hasHeightForWidth() and width > 0:
        return max(base, lay.minimumHeightForWidth(max(width, min_size(widget).width())))
    return base


def log_ticks(axis: pg.AxisItem, lo: float, hi: float):
    """Major ticks at 1, 2, 5 × 10^n inside [lo, hi] for a log-mode axis; no crowded minor labels."""
    lo, hi = max(min(lo, hi), 1e-12), max(lo, hi)
    vals = [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 2)
            for m in (1, 2, 5) if lo / 1.0001 <= m * 10.0 ** e <= hi * 1.0001]
    if len(vals) < 2:
        vals = [10.0 ** e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)]
    axis.setTicks([[(np.log10(v), f"{v:g}") for v in vals], []])


# ---- plot views ----------------------------------------------------------------------------

def _install_reset(plot: pg.PlotWidget, reset):
    """Double-click and the 'A' (auto-range) button restore the default view."""
    plot.reset_view = reset
    if getattr(plot, "_reset_hooked", False):
        return
    plot._reset_hooked = True

    def clicked(ev):
        if ev.double() and ev.button() == QtCore.Qt.LeftButton:
            plot.reset_view()
            ev.accept()

    plot.scene().sigMouseClicked.connect(clicked)
    item = plot.getPlotItem()
    btn = getattr(item, "autoBtn", None)
    if btn is not None:
        try:
            btn.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        btn.clicked.connect(lambda *_: plot.reset_view())
    plot.setToolTip(VIEW_TIP)


def _pad(lo, hi, frac):
    d = (hi - lo) or abs(hi) or 1.0
    return lo - frac * d, hi + frac * d


def _view_x(x_range, x_log, frac):
    x0, x1 = (np.log10(x_range[0]), np.log10(x_range[1])) if x_log else (float(x_range[0]), float(x_range[1]))
    return _pad(x0, x1, frac)


def histogram_view(plot: pg.PlotWidget, x_range, x_log: bool = False, y_log: bool = False):
    """x zooms and pans inside ``x_range`` only; y is not mouse-controlled and always fits
    the data visible in x (starting at 0 for a linear axis)."""
    vb = plot.getViewBox()
    x0, x1 = _view_x(x_range, x_log, 0.01)
    vb.setMouseEnabled(x=True, y=False)
    limits = dict(xMin=x0, xMax=x1, minXRange=(x1 - x0) / 100, maxXRange=x1 - x0)
    if not y_log:
        limits["yMin"] = 0.0
    vb.setLimits(**limits)
    vb.setAutoVisible(y=True)

    def reset():
        vb.setXRange(x0, x1, padding=0)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    _install_reset(plot, reset)
    reset()


def curve_view(plot: pg.PlotWidget, x_range, y_range=None, x_log=False, y_log=False):
    """Curves against x (e.g. resolution vs pT): x zooms inside ``x_range``; y fits the
    visible data, or stays at ``y_range`` when given."""
    vb = plot.getViewBox()
    x0, x1 = _view_x(x_range, x_log, 0.04)
    vb.setMouseEnabled(x=True, y=False)
    vb.setLimits(xMin=x0, xMax=x1, minXRange=(x1 - x0) / 50, maxXRange=x1 - x0)

    def reset():
        vb.setXRange(x0, x1, padding=0)
        if y_range is None:
            vb.setAutoVisible(y=True)
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        else:
            y0, y1 = (np.log10(y_range[0]), np.log10(y_range[1])) if y_log else y_range
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            vb.setYRange(y0, y1, padding=0)

    _install_reset(plot, reset)
    reset()


def geometry_view(plot: pg.PlotWidget, x_range, y_range, limit_x, limit_y, min_span=0.02):
    """Detector drawings: both axes zoom and pan, but only inside ``limit_x`` × ``limit_y``
    and never narrower than ``min_span`` (metres). The default view is ``x_range`` × ``y_range``."""
    vb = plot.getViewBox()
    vb.setMouseEnabled(x=True, y=True)
    vb.setLimits(xMin=limit_x[0], xMax=limit_x[1], yMin=limit_y[0], yMax=limit_y[1],
                 minXRange=min_span, minYRange=min_span,
                 maxXRange=limit_x[1] - limit_x[0], maxYRange=limit_y[1] - limit_y[0])

    def reset():
        vb.setRange(xRange=x_range, yRange=y_range, padding=0)

    _install_reset(plot, reset)
    reset()


def empty_view(plot: pg.PlotWidget):
    """A plot with nothing in it yet: no mouse interaction, so it cannot be dragged away."""
    plot.getViewBox().setMouseEnabled(x=False, y=False)


# ---- splitters -----------------------------------------------------------------------------

class ResponsiveSplitter(QtWidgets.QSplitter):
    """Side by side when the panes fit next to each other, one above the other when not.

    Its minimum size is that of the arrangement in use, with word-wrapped text measured at
    the panes' real widths, so a page around it scrolls instead of clipping."""

    def __init__(self):
        super().__init__(QtCore.Qt.Horizontal)
        self._min = None
        self._fractions = {}          # remembered pane proportions for each orientation
        self.splitterMoved.connect(lambda *_: self._remember())

    def _remember(self):
        sizes = self.sizes()
        if sum(sizes) > 0:
            self._fractions[self.orientation()] = [x / sum(sizes) for x in sizes]

    def restoreState(self, state):
        ok = super().restoreState(state)
        if ok:
            self._remember()
        return ok

    def _panes(self):
        return [self.widget(i) for i in range(self.count()) if not self.widget(i).isHidden()]

    def side_by_side_width(self):
        panes = self._panes()
        return sum(min_size(w).width() for w in panes) + self.handleWidth() * max(len(panes) - 1, 0)

    def _calc_min(self):
        panes = self._panes()
        if not panes:
            return super().minimumSizeHint()
        mins = [min_size(w) for w in panes]
        handles = self.handleWidth() * (len(panes) - 1)
        if self.orientation() == QtCore.Qt.Horizontal:
            sizes = [s for s, w in zip(self.sizes(), (self.widget(i) for i in range(self.count()))) if not w.isHidden()]
            hs = [min_height_for_width(w, max(sz, m.width())) for w, m, sz in zip(panes, mins, sizes)]
            return QtCore.QSize(max(m.width() for m in mins), max(hs))
        width = max(self.width(), max(m.width() for m in mins))
        return QtCore.QSize(max(m.width() for m in mins), sum(min_height_for_width(w, width) for w in panes) + handles)

    def minimumSizeHint(self):
        return self._calc_min()

    def event(self, e):
        # a pane's minimum size changed (e.g. a table filled with wider numbers): rearrange
        if e.type() == QtCore.QEvent.LayoutRequest and not getattr(self, "_pending", False):
            self._pending = True
            QtCore.QTimer.singleShot(0, self._reconsider)
        return super().event(e)

    def _reconsider(self):
        self._pending = False
        self._update_arrangement()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_arrangement()

    def _update_arrangement(self):
        if not self.isVisible() or self.width() <= 1:
            return
        need = self.side_by_side_width()
        if self.orientation() == QtCore.Qt.Horizontal and self.width() < need:
            self._arrange(QtCore.Qt.Vertical)
        elif self.orientation() == QtCore.Qt.Vertical and self.width() >= need + em(self, 1.5):
            self._arrange(QtCore.Qt.Horizontal)
        self._pane_heights()
        m = self._calc_min()
        if m != self._min:
            self._min = m
            self.updateGeometry()

    def _pane_heights(self):
        """Stacked: QSplitter shares the height by proportion, which can give a pane with
        wrapped text less than it needs; its minimum height is set to what it needs."""
        vertical = self.orientation() == QtCore.Qt.Vertical
        for w in self._panes():
            if not hasattr(w, "_own_min_h"):
                w._own_min_h = w.minimumHeight()
            need = 0
            if vertical:
                lay = w.layout()
                need = w.minimumSizeHint().height()
                if lay is not None and lay.hasHeightForWidth():
                    need = max(need, lay.minimumHeightForWidth(max(self.width(), 1)))
            target = max(w._own_min_h, need) if vertical else w._own_min_h
            if w.minimumHeight() != target:
                w.setMinimumHeight(target)

    def _arrange(self, orientation):
        self._remember()
        self.setOrientation(orientation)
        total = self.width() if orientation == QtCore.Qt.Horizontal else self.height()
        k = self._fractions.get(orientation) or [max(x, 1) for x in self._stretch]
        if total > 0 and len(k) == self.count():
            self.setSizes([int(total * x / sum(k)) for x in k])


def splitter(name: str, widgets, stretch, orientation=QtCore.Qt.Horizontal, collapsible=(), responsive=False):
    """Panes the user can resize. ``stretch``: initial proportions; the panes whose index is
    in ``collapsible`` can be dragged closed. MainWindow saves the sizes under ``name``.
    ``responsive``: stack the panes vertically when the window is too narrow for them."""
    s = ResponsiveSplitter() if responsive else QtWidgets.QSplitter(orientation)
    s.setObjectName(name)
    s.setChildrenCollapsible(False)
    for i, (w, k) in enumerate(zip(widgets, stretch)):
        s.addWidget(w)
        s.setStretchFactor(i, k)
        s.setCollapsible(i, i in collapsible)
    s.setHandleWidth(max(6, em(s, 0.4)))
    s._stretch = list(stretch)
    return s


def wrap(layout: QtWidgets.QLayout, margins=True) -> QtWidgets.QWidget:
    """A widget holding ``layout`` (to put layouts into splitters)."""
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    if not margins:
        layout.setContentsMargins(0, 0, 0, 0)
    return w


class FlowLayout(QtWidgets.QLayout):
    """Lays widgets out in a row and wraps them onto further rows when the row is too narrow
    (a toolbar that never forces its page wider)."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items = []
        self._space = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self):
        m = self.contentsMargins()
        w = sum(i.sizeHint().width() for i in self._items) + self._space * max(len(self._items) - 1, 0)
        h = max((i.sizeHint().height() for i in self._items), default=0)
        return QtCore.QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        size = QtCore.QSize()
        for i in self._items:
            size = size.expandedTo(i.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect, apply):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if x + hint.width() > r.right() + 1 and line_h > 0:
                x, y, line_h = r.x(), y + line_h + self._space, 0
            if apply:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._space
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()

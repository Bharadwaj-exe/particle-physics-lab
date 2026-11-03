"""One animation clock shared by the 2D and 3D event displays.

The clock runs a fraction f from 0 to 1 over the event's flight time: f = 0 is the
collision, f = 1 the moment the last muon leaves the drawn volume, after which the
fitted tracks are shown. Views map f to physical time with t = f * t_max (ns), so
both muons move at their real speeds (beta is close to 1 for every muon here).
"""
from __future__ import annotations

import time

from PySide6 import QtCore, QtWidgets

SPEEDS = (("¼×", 0.25), ("½×", 0.5), ("1×", 1.0), ("2×", 2.0))
PLAY_SECONDS = 3.0          # wall-clock length of one event at 1×


class AnimationClock(QtCore.QObject):
    frame = QtCore.Signal(float)        # fraction in [0, 1]
    state = QtCore.Signal(bool)         # playing?
    autoplay_changed = QtCore.Signal(bool)

    def __init__(self, parent=None, autoplay: bool = False):
        super().__init__(parent)
        self.fraction = 1.0
        self.playing = False
        self.speed = 1.0
        self.autoplay = autoplay
        self.t_max = 0.0                # ns, for labels
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._last = None

    def set_autoplay(self, on: bool):
        self.autoplay = bool(on)
        self.autoplay_changed.emit(self.autoplay)

    def time_ns(self):
        return self.fraction * self.t_max

    def set_span(self, t_max_ns: float):
        self.t_max = float(t_max_ns)

    def seek(self, f: float):
        self.fraction = min(max(float(f), 0.0), 1.0)
        self.frame.emit(self.fraction)

    def play(self):
        if self.fraction >= 1.0:
            self.fraction = 0.0
        self.playing = True
        self._last = time.perf_counter()
        self._timer.start()
        self.state.emit(True)
        self.frame.emit(self.fraction)

    def pause(self):
        self.playing = False
        self._timer.stop()
        self.state.emit(False)

    def toggle(self):
        self.pause() if self.playing else self.play()

    def restart(self):
        self.fraction = 0.0
        self.play()

    def new_event(self, t_max_ns: float):
        """Called when a new event is shown: replay it, or show it complete."""
        self.set_span(t_max_ns)
        if self.autoplay:
            self.restart()
        else:
            self.pause()
            self.seek(1.0)

    def _tick(self):
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        self.fraction = min(self.fraction + dt * self.speed / PLAY_SECONDS, 1.0)
        self.frame.emit(self.fraction)
        if self.fraction >= 1.0:
            self.pause()


class AnimationBar(QtWidgets.QWidget):
    """Play / pause, restart, time slider, speed and auto-play, bound to a shared clock."""

    def __init__(self, clock: AnimationClock, parent=None):
        super().__init__(parent)
        self.clock = clock
        self.restart_btn = QtWidgets.QToolButton(); self.restart_btn.setText("⏮"); self.restart_btn.setToolTip("Replay from the collision")
        self.play_btn = QtWidgets.QToolButton(); self.play_btn.setToolTip("Play / pause the event (Ctrl+P)")
        fm = self.fontMetrics()
        self.play_btn.setMinimumWidth(fm.height() * 2)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, 1000)
        self.slider.setToolTip("Time after the collision")
        self.slider.setMinimumWidth(fm.height() * 4)
        self.time_label = QtWidgets.QLabel()
        self.time_label.setMinimumWidth(fm.horizontalAdvance("t = 88.88 ns (fitted)") + 4)
        self.speed = QtWidgets.QComboBox()
        for text, v in SPEEDS:
            self.speed.addItem(text, v)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip("Playback speed")
        self.auto = QtWidgets.QCheckBox("auto-play"); self.auto.setChecked(clock.autoplay)
        self.auto.setToolTip("Animate every new event automatically")
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        for w in (self.restart_btn, self.play_btn):
            lay.addWidget(w)
        lay.addWidget(self.slider, 1)
        for w in (self.time_label, self.speed, self.auto):
            lay.addWidget(w)
        self.restart_btn.clicked.connect(clock.restart)
        self.play_btn.clicked.connect(clock.toggle)
        self.slider.valueChanged.connect(self._slid)
        self.speed.currentIndexChanged.connect(lambda _: setattr(clock, "speed", self.speed.currentData()))
        self.auto.toggled.connect(clock.set_autoplay)
        clock.autoplay_changed.connect(self._sync_auto)
        clock.frame.connect(self._show)
        clock.state.connect(self._state)
        self._state(clock.playing)
        self._show(clock.fraction)

    def _slid(self, v):
        if self.clock.playing:
            self.clock.pause()
        self.clock.seek(v / 1000)

    def _sync_auto(self, on):
        if self.auto.isChecked() != on:
            self.auto.blockSignals(True); self.auto.setChecked(on); self.auto.blockSignals(False)

    def _show(self, f):
        self.slider.blockSignals(True)
        self.slider.setValue(round(1000 * f))
        self.slider.blockSignals(False)
        t = self.clock.time_ns()
        self.time_label.setText(f"t = {t:5.2f} ns" + (" (fitted)" if f >= 1.0 else ""))
        self.time_label.setToolTip("Time after the collision" + ("; both muons have left and the tracks are fitted"
                                                                 if f >= 1.0 else ""))

    def _state(self, playing):
        self.play_btn.setText("⏸" if playing else "▶")

"""Desktop interface (PySide6 + pyqtgraph). The physics never imports this module.

Long computations (simulation, fits, resolution scans) run in a worker thread so
the window stays responsive.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters

from ..domain.particles import load as load_particles, RESONANCES
from ..domain.conservation import check, EXAMPLES
from ..detector import cms_inspired, uniform_barrel, gluckstern, DetectorConfig
from ..analysis import simulate, fit_peak
from ..analysis.compare import DATA_FIT, SIM_FIT
from ..analysis.studies import pt_resolution
from ..data import cms
from ..physics import generate, boost, decay_rest
from ..provenance import run_record, save_json
from ..visualization.event_display import (make_event, detector_geometry, momentum_arrow, path_until, data_event,
                                           pick_data_row, DATA_B)
from ..education import from_display_event, gate
from ..units import C_TRACK, um
from . import theme
from .animation import AnimationBar, AnimationClock
from .layout import (em, histogram_view, curve_view, geometry_view, empty_view, splitter, wrap, log_ticks,
                     min_height_for_width, min_size, FlowLayout)

pg.setConfigOptions(antialias=True, background="w", foreground="k")


def fg():
    return theme.current()["fg"]
COLORS = {"J/psi": "#1f77b4", "Upsilon(1S)": "#2ca02c", "Z": "#d62728"}
DB = load_particles()


def label(name):
    return DB.get(name).label


# ---------------------------------------------------------------------------
class Worker(QtCore.QObject):
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self.fn(*self.args, progress=self.progress.emit, **self.kw))
        except Exception:
            self.failed.emit(traceback.format_exc())


class TaskRunner(QtCore.QObject):
    """Runs one background task at a time and reports to a status bar."""
    def __init__(self, window, report=True):
        super().__init__()
        self.window, self.thread, self.worker, self.report = window, None, None, report

    def busy(self):
        return self.thread is not None

    def start(self, fn, on_done, *args, text="Working…", **kw):
        if self.busy():
            if self.report:
                QtWidgets.QMessageBox.information(self.window, "Busy", "Another computation is still running.")
            return False
        self.thread = QtCore.QThread()
        self.worker = Worker(fn, *args, **kw)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        if self.report:
            self.worker.progress.connect(self.window.set_progress)
        self.worker.finished.connect(self._cleanup)
        self.worker.failed.connect(self._cleanup)
        self.worker.finished.connect(on_done)
        self.worker.failed.connect(self._fail)
        if self.report:
            self.window.set_status(text, busy=True)
        self.thread.start()
        return True

    def _fail(self, tb):
        QtWidgets.QMessageBox.critical(self.window, "Error", tb[-2000:])

    def _cleanup(self, *_):
        self.thread.quit()
        self.thread.wait()
        self.thread = self.worker = None
        if self.report:
            self.window.set_status("Ready", busy=False)


# ---------------------------------------------------------------------------
class SetupPanel(QtWidgets.QGroupBox):
    """Experiment configuration: physically meaningful parameters only."""
    def __init__(self):
        super().__init__("Experiment configuration")
        f = QtWidgets.QFormLayout(self)
        self.proc = QtWidgets.QComboBox()
        for k in RESONANCES:
            self.proc.addItem(f"{label(k)} → μ⁺μ⁻", k)
        self.n = QtWidgets.QSpinBox(); self.n.setRange(1000, 5_000_000); self.n.setSingleStep(50_000); self.n.setValue(100_000)
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(20261102)
        self.preset = QtWidgets.QComboBox(); self.preset.addItems(["CMS-inspired tracker", "Uniform barrel (reference)"])
        self.B = QtWidgets.QDoubleSpinBox(); self.B.setRange(0.5, 10); self.B.setValue(3.8); self.B.setSuffix(" T")
        self.res = QtWidgets.QDoubleSpinBox(); self.res.setRange(0.1, 20); self.res.setValue(1.0); self.res.setSingleStep(0.5)
        self.mat = QtWidgets.QDoubleSpinBox(); self.mat.setRange(0.0, 10); self.mat.setValue(1.0); self.mat.setSingleStep(0.5)
        self.eff = QtWidgets.QDoubleSpinBox(); self.eff.setRange(0.5, 1.0); self.eff.setDecimals(3); self.eff.setValue(0.99); self.eff.setSingleStep(0.01)
        self.ms = QtWidgets.QCheckBox("multiple scattering"); self.ms.setChecked(True)
        self.smear = QtWidgets.QCheckBox("hit smearing"); self.smear.setChecked(True)
        self.ptmin = QtWidgets.QDoubleSpinBox(); self.ptmin.setRange(0.5, 50); self.ptmin.setValue(3.0); self.ptmin.setSuffix(" GeV")
        self.etamax = QtWidgets.QDoubleSpinBox(); self.etamax.setRange(0.1, 2.5); self.etamax.setValue(2.4)
        self.source = QtWidgets.QComboBox()
        self.source.addItem("Simulated events", "sim")
        self.source.addItem("CMS 2011 data events", "data")
        self.source.setToolTip("What the event displays and the reconstruction exercise show")
        if not cms.available():
            self.source.model().item(1).setEnabled(False)
            self.source.setToolTip("CMS data not downloaded: python -c \"from particlelab.data import cms; cms.fetch()\"")
        self.gate_label = QtWidgets.QLabel(); self.gate_label.setWordWrap(True)
        self.gate_label.setStyleSheet("color:#080; font-size: 9pt;")
        self.proc.currentIndexChanged.connect(self._update_gate)
        self._update_gate()
        for lab, w in [("Process", self.proc), ("Generated events", self.n), ("Random seed", self.seed),
                       ("Detector", self.preset), ("Magnetic field", self.B), ("Hit-resolution scale", self.res),
                       ("Material scale", self.mat), ("Hit efficiency", self.eff), ("", self.ms), ("", self.smear),
                       ("Muon pT >", self.ptmin), ("Muon |η| <", self.etamax), ("Event displays", self.source)]:
            f.addRow(lab, w)
        f.insertRow(1, self.gate_label)          # full width under the process choice

    def _update_gate(self):
        g = gate(f"{self.proc.currentData()} -> mu+ mu-")
        self.gate_label.setText(("✓ " if g.simulable else "✗ ") + g.message)
        th = theme.current()
        self.gate_label.setStyleSheet(f"color:{th['ok'] if g.simulable else th['bad']}; font-size: 9pt;")

    def theme_changed(self):
        self._update_gate()

    def select_process(self, key: str) -> bool:
        i = self.proc.findData(key)
        if i >= 0:
            self.proc.setCurrentIndex(i)
        return i >= 0

    def detector(self) -> DetectorConfig:
        base = cms_inspired() if self.preset.currentIndex() == 0 else uniform_barrel()
        det = base.with_(B=self.B.value(), multiple_scattering=self.ms.isChecked() and self.mat.value() > 0,
                         smear=self.smear.isChecked(), pt_min=self.ptmin.value(), eta_max=self.etamax.value(),
                         hit_efficiency=self.eff.value() if self.preset.currentIndex() == 0 else 1.0)
        if self.res.value() != 1.0:
            det = det.scaled_resolution(self.res.value())
        if self.mat.value() not in (0.0, 1.0):
            det = det.scaled_material(self.mat.value())
        return det

    def settings(self) -> dict:
        return {"process": self.proc.currentData(), "events": self.n.value(), "seed": self.seed.value(),
                "preset": self.preset.currentText(), "B": self.B.value(), "resolution_scale": self.res.value(),
                "material_scale": self.mat.value(), "hit_efficiency": self.eff.value(),
                "multiple_scattering": self.ms.isChecked(), "smear": self.smear.isChecked(),
                "pt_min": self.ptmin.value(), "eta_max": self.etamax.value()}


def run_simulation_and_fits(settings, det, data_masses, progress=None):
    name = settings["process"]
    # skip events that cannot pass the cuts; each run measures what that loses
    sim = simulate(name, settings["events"], det, seed=settings["seed"], progress=progress, preselect=True)
    out = {"sim": sim, "settings": settings}
    if sim.reco_mass.size > 50:
        out["sim_fit"] = fit_peak(sim.reco_mass, name, background=False, **SIM_FIT[name])
    if data_masses is not None:
        out["data_fit"] = fit_peak(data_masses, name, **DATA_FIT[name])
    return out


class MassTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.result = None
        self.overlay = QtWidgets.QCheckBox("Overlay CMS 2011 data (record 545)")
        self.overlay.setEnabled(cms.available())
        self.overlay.setChecked(cms.available())
        if not cms.available():
            self.overlay.setToolTip("Run: python -c \"from particlelab.data import cms; cms.fetch()\"")
        run = QtWidgets.QPushButton("Run simulation"); run.setDefault(True)
        export = QtWidgets.QToolButton(); export.setText("Export")
        export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        menu = QtWidgets.QMenu(export)
        exp_png = menu.addAction("Plot as PNG…")
        exp_csv = menu.addAction("Masses as CSV…")
        exp_json = menu.addAction("Run record as JSON…")
        export.setMenu(menu)
        self.export_button = export
        self.plot = pg.PlotWidget(labels={"bottom": "m(μμ) [GeV]", "left": "events / bin"})
        self.plot.addLegend(offset=(-10, 10), labelTextSize="8pt", verSpacing=-4)
        self.plot.setToolTip("Simulation scaled to the fitted CMS signal yield, on top of the fitted data background. "
                             + "Wheel or right-drag: zoom · drag: pan · double-click: reset")
        empty_view(self.plot)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["quantity", "simulation", "CMS data"])
        hh = self.table.horizontalHeader()
        for c in range(3):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumWidth(em(self, 16))
        buttons = QtWidgets.QHBoxLayout()
        self.overlay.setText("Overlay CMS 2011 data")
        self.overlay.setToolTip("CERN Open Data record 545 (CMS dimuon events, 2011)")
        for w in (run, self.overlay, export):
            buttons.addWidget(w)
        buttons.addStretch()
        self.plot.setMinimumSize(em(self, 14), em(self, 10))
        self.split = splitter("mass", [self.plot, self.table], [3, 2], responsive=True)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(buttons); lay.addWidget(self.split, 1)
        run.clicked.connect(self.run)
        exp_png.triggered.connect(self.export_png)
        exp_csv.triggered.connect(self.export_csv)
        exp_json.triggered.connect(self.export_json)

    def run(self):
        s = self.main.setup.settings()
        g = gate(f"{s['process']} -> mu+ mu-")
        if not g.simulable:            # the generator only ever receives a checked process
            QtWidgets.QMessageBox.warning(self, "Process not simulated", g.message)
            return
        data = None
        if self.overlay.isChecked():
            data = self.main.data_masses()
        self.main.tasks.start(run_simulation_and_fits, self.show_result, s, self.main.setup.detector(), data,
                              text=f"Simulating {s['events']} {s['process']} events…")

    def show_result(self, r):
        self.result = r
        self.main.last_sim = r["sim"]
        self.main.new_event(seed=r["sim"].seed)
        self.main.performance.show_cutflow(r["sim"])
        self.draw_result(r)

    def theme_changed(self):
        if self.result is not None:
            self.draw_result(self.result)

    def draw_result(self, r):
        name = r["settings"]["process"]
        self.plot.clear()
        rows = []
        sim = r["sim"]
        df = r.get("data_fit")
        sf = r.get("sim_fit")
        if df is not None:
            ctr = 0.5 * (df.edges[1:] + df.edges[:-1])
            self.plot.addItem(pg.ErrorBarItem(x=ctr, y=df.counts, height=2 * np.sqrt(df.counts), pen=fg()))
            self.plot.plot(ctr, df.counts, pen=None, symbol="o", symbolSize=4, symbolBrush=fg(), symbolPen=None,
                           name="CMS data")
            fine = np.linspace(*df.window, 1200)
            self.plot.plot(fine, df.curve(fine), pen=pg.mkPen("#ff7f0e", width=2), name="data fit")
            self.plot.plot(fine, df.curve(fine, "background"), pen=pg.mkPen("#ff7f0e", width=1, style=QtCore.Qt.DashLine),
                           name="data background")
            hs, _ = np.histogram(sim.reco_mass, df.edges)
            n_in = np.sum((sim.reco_mass > df.window[0]) & (sim.reco_mass < df.window[1]))
            scale = df.n_signal / max(n_in, 1)
            self.plot.plot(df.edges, hs * scale + df.curve(ctr, "background"), stepMode="center",
                           pen=pg.mkPen(COLORS[name], width=2), name="simulation + background")
        elif sf is not None:
            hs, _ = np.histogram(sim.reco_mass, sf.edges)
            self.plot.plot(sf.edges, hs, stepMode="center", pen=pg.mkPen(COLORS[name], width=2), name="simulation")
            fine = np.linspace(*sf.window, 1200)
            self.plot.plot(fine, sf.curve(fine), pen=pg.mkPen(fg(), width=1.5, style=QtCore.Qt.DashLine),
                           name="fit")
        fit = df if df is not None else sf
        if fit is not None:
            histogram_view(self.plot, fit.window)          # x zooms inside the fit window; y follows the data
        fmt = lambda f, attr, err, k=1.0, d=5: "—" if f is None else f"{k * getattr(f, attr):.{d}f} ± {k * getattr(f, err):.{d}f}"
        rows = [("PDG mass [GeV]", f"{DB.get(name).mass:.5f}", f"{DB.get(name).mass:.5f}"),
                ("fitted mass [GeV]", fmt(sf, "mass", "mass_err"), fmt(df, "mass", "mass_err")),
                ("resolution param [MeV]", fmt(sf, "sigma", "sigma_err", 1000, 1), fmt(df, "sigma", "sigma_err", 1000, 1)),
                ("FWHM [MeV]", "—" if sf is None else f"{1000 * sf.fwhm():.1f}", "—" if df is None else f"{1000 * df.fwhm():.1f}"),
                ("signal events", "—" if sf is None else f"{sf.n_signal:.0f}", fmt(df, "n_signal", "n_signal_err", 1, 0)),
                ("background events", "—", "—" if df is None else f"{df.n_background:.0f}"),
                ("fit p-value", "—" if sf is None else f"{sf.pvalue:.3f}", "—" if df is None else f"{df.pvalue:.3f}"),
                ("signal model", "—" if sf is None else sf.signal, "—" if df is None else df.signal),
                ("selected / generated", f"{sim.reco_mass.size} / {sim.n_generated}", "—"),
                ("acceptance × efficiency", f"{sim.acceptance:.2%}", "—")]
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        # wide enough for every number (the plot takes the remaining width)
        self.table.resizeColumnsToContents()
        need = sum(self.table.columnWidth(c) for c in range(3)) + 2 * self.table.frameWidth() + em(self, 1.2)
        self.table.setMinimumWidth(need)

    def _need(self):
        if self.result is None:
            QtWidgets.QMessageBox.information(self, "No run", "Run a simulation first.")
            return False
        return True

    def export_png(self):
        if not self._need():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "mass_spectrum.png", "PNG (*.png)")
        if path:
            pg.exporters.ImageExporter(self.plot.plotItem).export(path)

    def export_csv(self):
        if not self._need():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save masses", "masses.csv", "CSV (*.csv)")
        if path:
            s = self.result["sim"]
            np.savetxt(path, np.column_stack([s.true_mass, s.reco_mass]), delimiter=",",
                       header="true_mass_GeV,reco_mass_GeV", comments="")

    def export_json(self):
        if not self._need():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save run record", "run_record.json", "JSON (*.json)")
        if path:
            r = self.result
            res = {"cutflow": r["sim"].cutflow, "seed": r["sim"].seed}
            for k in ("sim_fit", "data_fit"):
                if k in r:
                    res[k] = r[k].summary()
            save_json(run_record(None, r["sim"].detector, data=cms.RECORD if "data_fit" in r else None, results=res,
                                 extra={"gui_settings": r["settings"]}), path)


# ---------------------------------------------------------------------------
class FeynmanWidget(QtWidgets.QWidget):
    """Lowest-order diagram: q q̄ (or the bound QQ̄ state) → γ*/Z → μ⁺μ⁻."""
    def __init__(self):
        super().__init__()
        self.name = "J/psi"
        self.setMinimumSize(em(self, 13), em(self, 9))

    def set_process(self, name):
        self.name = name
        self.update()

    def sizeHint(self):
        return QtCore.QSize(em(self, 18), em(self, 12))

    def paintEvent(self, _):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        th = theme.current()
        p.fillRect(self.rect(), QtGui.QColor(th["bg"]))
        u0 = self.fontMetrics().height()                 # all offsets scale with the font
        note = {"J/psi": "cc̄ bound state annihilates through a virtual photon",
                "Upsilon(1S)": "bb̄ bound state annihilates through a virtual photon",
                "Z": "Drell–Yan: qq̄ → Z/γ* → μ⁺μ⁻"}[self.name]
        small = QtGui.QFont(self.font()); small.setPointSizeF(max(self.font().pointSizeF() * 0.9, 7))
        note_rect = QtGui.QFontMetrics(small).boundingRect(QtCore.QRect(0, 0, self.width() - 4, 10 * u0),
                                                           QtCore.Qt.TextWordWrap | QtCore.Qt.AlignHCenter, note)
        w, h = self.width(), self.height() - note_rect.height() - int(0.4 * u0)
        mx, my = max(0.6 * u0, 0.06 * w), max(1.2 * u0, 0.12 * h)   # room for the particle labels
        v1, v2 = QtCore.QPointF(w * 0.36, h * 0.5), QtCore.QPointF(w * 0.64, h * 0.5)
        p.setPen(QtGui.QPen(QtGui.QColor(th["fg"]), max(1.5, u0 / 9)))
        big = QtGui.QFont(self.font()); big.setPointSizeF(self.font().pointSizeF() * 1.2); p.setFont(big)
        quark = {"J/psi": "c", "Upsilon(1S)": "b", "Z": "q"}[self.name]
        arrow = 0.35 * u0

        def fermion(a, b, text, pos):
            p.drawLine(a, b)
            mid = (a + b) / 2; d = b - a; L = (d.x()**2 + d.y()**2) ** 0.5; u = d / L
            nrm = QtCore.QPointF(-u.y(), u.x())
            p.setBrush(QtGui.QColor(th["fg"]))
            p.drawPolygon(QtGui.QPolygonF([mid + u * arrow * 1.4, mid - u * arrow + nrm * arrow, mid - u * arrow - nrm * arrow]))
            p.drawText(pos, text)

        a1, a2 = QtCore.QPointF(mx, my), QtCore.QPointF(mx, h - my)
        b1, b2 = QtCore.QPointF(w - mx, my), QtCore.QPointF(w - mx, h - my)
        fermion(a1, v1, quark, a1 + QtCore.QPointF(-0.1 * u0, -0.35 * u0))
        fermion(v1, a2, quark + "̄", a2 + QtCore.QPointF(-0.1 * u0, 1.0 * u0))
        fermion(b1, v2, "μ⁺", b1 + QtCore.QPointF(-0.9 * u0, -0.35 * u0))
        fermion(v2, b2, "μ⁻", b2 + QtCore.QPointF(-0.9 * u0, 1.0 * u0))
        path = QtGui.QPainterPath(v1)
        amp = 0.4 * u0
        for i in range(1, 33):
            path.lineTo(v1.x() + (v2.x() - v1.x()) * i / 32, v1.y() + amp * np.sin(np.pi * i / 2))
        p.setBrush(QtCore.Qt.NoBrush); p.drawPath(path)
        label_g = "Z/γ*" if self.name == "Z" else "γ*"
        p.drawText(QtCore.QPointF((v1.x() + v2.x()) / 2 - p.fontMetrics().horizontalAdvance(label_g) / 2, v1.y() - 0.8 * u0),
                   label_g)
        p.setFont(small); p.setPen(QtGui.QColor(th["muted"]))
        p.drawText(QtCore.QRectF(2, h + 0.3 * u0, self.width() - 4, note_rect.height() + 2),
                   QtCore.Qt.TextWordWrap | QtCore.Qt.AlignHCenter, note)


def event_info_lines(ev) -> list:
    if ev.is_data:
        lines = [f"CMS 2011 data: run {ev.run}, event {ev.event_number}",
                 f"m(μμ) = {ev.reco_mass:.4f} GeV, in the {label(ev.process)} window (may be background)",
                 f"muon types {'/'.join(ev.muon_types)} (G global, T tracker); vertex not in the file, drawn at 0"]
        for t in ev.tracks:
            lines.append(f"{t.label}: q = {t.charge_reco:+d}, pT = {t.pt_reco:.3f} GeV, η = {t.eta_reco:+.2f}, "
                         f"R = {t.pt_reco / (C_TRACK * ev.B):.2f} m")
            if t.outer is not None:
                lines.append(f"   outside the tracker: {t.outer.summary()}")
        return lines
    lines = [f"{label(ev.process)} → μ⁺μ⁻   true m = {ev.true_mass:.4f} GeV",
             f"reconstructed m = {ev.reco_mass:.4f} GeV", f"vertex z = {100 * ev.vertex[2]:.1f} cm"]
    for t in ev.tracks:
        lines.append(f"{t.label}: q_reco = {t.charge_reco:+d}, pT = {t.pt_reco:.3f} GeV (hit-error σ {t.sigma_pt:.2g}) "
                     f"true {t.pt_true:.3f}, η = {t.eta_reco:+.2f}, hits = {t.n_hits}, "
                     f"R = {t.pt_reco / (C_TRACK * ev.B):.2f} m")
        if t.outer is not None:
            lines.append(f"   outside the tracker: {t.outer.summary()}")
    if ev.has_outer:
        lines.append("Outer detector: schematic (field reversed in the iron, average energy loss).")
    return lines


TRACK_COLORS = {"μ⁻": "#1f77b4", "μ⁺": "#d62728"}

DIAGRAM_VS_TRACKS = (
    "<b>Diagram ≠ picture of the event.</b> The diagram's lines stand for particles in the interaction "
    "amplitude, not for paths, and left-to-right is not a distance in the detector. The Z/γ* (or the "
    "quarkonium) decays about 10⁻¹⁶ m from the collision point and never reaches the detector: its mass is "
    "<i>inferred</i> from the two muons. The curves on the left are measured trajectories, helices of radius "
    "R = pT / (0.3 B), fitted to the recorded hits.")


class ReconstructTab(QtWidgets.QWidget):
    """The learner reconstructs the pair mass of the displayed event by hand."""
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.ex = None
        self.my_masses = []
        intro = QtWidgets.QLabel(
            "Reconstruct the μ⁺μ⁻ mass of the event shown in the event displays yourself. The tracker gives, for "
            "each muon, the chord L from the beam line to its last hit, the sagitta s of the fitted track over that "
            "chord, the azimuth φ and the pseudorapidity η. Work out R = L²/(8s) + s/2, pT = 0.2998·B·R, "
            "E = √(pT²cosh²η + m_μ²) and m² = (E₁+E₂)² − |p⃗₁+p⃗₂|² with p⃗ = pT(cos φ, sin φ, sinh η).")
        intro.setWordWrap(True)
        self.givens = QtWidgets.QTableWidget(2, 6)
        self.givens.setHorizontalHeaderLabels(["muon", "charge", "L [m]", "s [mm]", "φ [rad]", "η"])
        for c, tip in enumerate(["", "", "chord from the beam line to the last hit", "sagitta of the track over the chord",
                                 "azimuth", "pseudorapidity"]):
            if tip:
                self.givens.horizontalHeaderItem(c).setToolTip(tip)
        self.givens.verticalHeader().setVisible(False)
        self.givens.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.givens.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.givens.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._fit_table_height()
        self.consts = QtWidgets.QLabel()
        grid = QtWidgets.QGridLayout()
        self.inputs = {}
        for c, head in enumerate(["", "R [m]", "pT [GeV]", "E [GeV]"]):
            grid.addWidget(QtWidgets.QLabel(f"<b>{head}</b>"), 0, c)
        for r, lab in enumerate(["μ⁻", "μ⁺"]):
            grid.addWidget(QtWidgets.QLabel(lab), r + 1, 0)
            for c, step in enumerate(["R", "pT", "E"]):
                e = QtWidgets.QLineEdit(); e.setPlaceholderText(step)
                self.inputs[(step, r)] = e
                grid.addWidget(e, r + 1, c + 1)
        grid.addWidget(QtWidgets.QLabel("<b>m(μ⁺μ⁻) [GeV]</b>"), 3, 0)
        self.inputs[("mass", None)] = QtWidgets.QLineEdit(); self.inputs[("mass", None)].setPlaceholderText("mass")
        grid.addWidget(self.inputs[("mass", None)], 3, 1)
        check_btn = QtWidgets.QPushButton("Check my answers"); check_btn.setDefault(True)
        solve_btn = QtWidgets.QPushButton("Show worked solution")
        self.add_btn = QtWidgets.QPushButton("Add my mass to the histogram"); self.add_btn.setEnabled(False)
        nxt = QtWidgets.QPushButton("Next event")
        self.feedback = QtWidgets.QTextEdit(); self.feedback.setReadOnly(True)
        self.hist = pg.PlotWidget(title="My masses", labels={"bottom": "m(μμ) [GeV]", "left": "entries"})
        empty_view(self.hist)
        self.hist.setMinimumSize(em(self, 12), em(self, 10))
        buttons = QtWidgets.QGridLayout()
        for i, w in enumerate((check_btn, solve_btn, self.add_btn, nxt)):
            buttons.addWidget(w, i // 2, i % 2)
        buttons.setColumnStretch(2, 1)
        left = QtWidgets.QVBoxLayout()
        for w in (intro, self.givens, self.consts):
            left.addWidget(w)
        left.addLayout(grid); left.addLayout(buttons); left.addWidget(self.feedback, 1)
        self.split = splitter("reconstruct", [wrap(left, margins=False), self.hist], [3, 2], responsive=True)
        lay = QtWidgets.QHBoxLayout(self); lay.addWidget(self.split)
        check_btn.clicked.connect(self.check)
        solve_btn.clicked.connect(self.show_solution)
        self.add_btn.clicked.connect(self.add_mass)
        nxt.clicked.connect(lambda: self.main.new_event())

    def _fit_table_height(self):
        """Exactly the header and the two rows: no scroll bar, no empty space, at any font size."""
        t = self.givens
        t.resizeRowsToContents()
        rows = sum(t.rowHeight(r) for r in range(t.rowCount())) or 2 * em(t, 1.6)
        t.setFixedHeight(t.horizontalHeader().sizeHint().height() + rows + 2 * t.frameWidth() + 2)

    def show_event(self, ev):
        if ev is None:
            return
        self.ex = from_display_event(ev)
        for r, t in enumerate(self.ex.tracks):
            vals = [t.label, f"{t.charge:+d}", f"{t.chord:.4f}", f"{1000 * t.sagitta:.4f}", f"{t.phi:+.4f}", f"{t.eta:+.4f}"]
            for c, v in enumerate(vals):
                self.givens.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self._fit_table_height()
        what = (f"CMS 2011 data, run {ev.run} event {ev.event_number} ({label(ev.process)} window); chord to the last "
                "computed layer crossing" if ev.is_data else f"process: {label(ev.process)}")
        self.consts.setText(f"B = {self.ex.B:g} T,   m_μ = {self.ex.m_mu:.5f} GeV,   {what}")
        for e in self.inputs.values():
            e.clear()
        self.feedback.clear()
        self.add_btn.setEnabled(False)

    def check(self):
        if self.ex is None:
            return
        html, mass_ok = [], False
        for (step, track), edit in self.inputs.items():
            if not edit.text().strip():
                continue
            fb = self.ex.check(step, edit.text().replace(",", "."), track)
            who = "pair" if track is None else self.ex.tracks[track].label
            colour = theme.current()["ok" if fb.ok else "bad"]
            html.append(f"<p style='color:{colour}'>{'✓' if fb.ok else '✗'} {step} ({who}): {fb.message}</p>")
            mass_ok |= step == "mass" and fb.ok
        self.feedback.setHtml("".join(html) or "<p>Fill in at least one box.</p>")
        self.add_btn.setEnabled(mass_ok)

    def show_solution(self):
        if self.ex is not None:
            self.feedback.append("\n".join(["", "Worked solution:"] + self.ex.worked_solution()))

    def add_mass(self):
        try:
            self.my_masses.append(float(self.inputs[("mass", None)].text().replace(",", ".")))
        except ValueError:
            return
        self.add_btn.setEnabled(False)
        self.hist.clear()
        m = np.asarray(self.my_masses)
        lo, hi = m.min() - 0.05 * abs(m.min()), m.max() + 0.05 * abs(m.max()) + 1e-6
        h, e = np.histogram(m, bins=max(5, min(30, len(m))), range=(lo, hi))
        self.hist.plot(e, h, stepMode="center", fillLevel=0, brush=(31, 119, 180, 120))
        histogram_view(self.hist, (e[0], e[-1]))
        self.hist.setTitle(f"My masses: {len(m)}, mean {m.mean():.4f} GeV")


FLASH_NS = 0.6      # a hit is highlighted for this long after the muon crosses its surface


class EventView2D(QtWidgets.QWidget):
    """Transverse (x–y) and longitudinal (z–r) views of one event, drawn at any animation time."""

    def __init__(self):
        super().__init__()
        self.caption = QtWidgets.QLabel(); self.caption.setWordWrap(True)
        self.xy = pg.PlotWidget(title="Transverse", labels={"bottom": "x [m]", "left": "y [m]"})
        self.rz = pg.PlotWidget(title="Longitudinal", labels={"bottom": "z [m]", "left": "±r [m]"})
        self.xy.setAspectLocked(True)
        for p in (self.xy, self.rz):
            p.setMinimumSize(em(self, 9), em(self, 9))
        plots = QtWidgets.QHBoxLayout(); plots.addWidget(self.xy, 2); plots.addWidget(self.rz, 3)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.caption); lay.addLayout(plots, 1)
        self.ev = self.geom = None
        self.zoom = "whole"          # "tracker" or "whole" (used when the outer detector is shown)
        self.show_true = True
        self.progress, self.span = 1.0, None
        self._tracks = []

    # ---- building ---------------------------------------------------------------
    OUTER_COLORS = {"ECAL": "#2e9e5b", "HCAL": "#d9a227", "solenoid": "#8e9aa4", "yoke": "#c0392b", "chambers": "#6d7b86"}

    def _outer_color(self, layer_name, group):
        key = "ECAL" if layer_name.startswith("ECAL") else "HCAL" if layer_name.startswith("HCAL") else \
            "solenoid" if group == "solenoid" else "yoke"
        return self.OUTER_COLORS[key]

    @staticmethod
    def _ring(r_in, r_out, polygon):
        """Closed path of a ring in the x-y plane: round, or twelve flat sides like the CMS wheels."""
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.OddEvenFill)
        if polygon:
            phi = np.radians(np.arange(0, 361, 30)); scale = 1 / np.cos(np.pi / 12)
        else:
            phi = np.linspace(0, 2 * np.pi, 145); scale = 1.0
        for r in (r_out, r_in):
            pts = [QtCore.QPointF(r * scale * np.cos(a), r * scale * np.sin(a)) for a in phi]
            path.addPolygon(QtGui.QPolygonF(pts)); path.closeSubpath()
        return path

    def _shape(self, plot, path, color, alpha, width=1.0, dashed=False):
        item = QtWidgets.QGraphicsPathItem(path)
        c = QtGui.QColor(color); fill = QtGui.QColor(c); fill.setAlpha(alpha)
        pen = QtGui.QPen(c); pen.setCosmetic(True); pen.setWidthF(width)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        item.setPen(pen); item.setBrush(QtGui.QBrush(fill) if alpha else QtCore.Qt.NoBrush)
        item.setZValue(-20)
        plot.addItem(item)
        return item

    def draw_outer(self, od):
        """Calorimeters, coil, return yoke and muon stations (schematic) in both projections."""
        dark = theme.current()["name"] == "dark"
        for layer in od.layers:
            col = self._outer_color(layer.name, layer.group)
            alpha = 70 if layer.group != "return yoke" else 90
            if layer.kind == "barrel":
                self._shape(self.xy, self._ring(layer.r_in, layer.r_out, layer.group == "return yoke"), col, alpha)
                for s in (1, -1):
                    path = QtGui.QPainterPath()
                    path.addRect(QtCore.QRectF(-layer.z_out, s * layer.r_in if s > 0 else -layer.r_out,
                                               2 * layer.z_out, layer.r_out - layer.r_in))
                    self._shape(self.rz, path, col, alpha)
            else:
                for sz in (1, -1):
                    for sr in (1, -1):
                        z0 = layer.z_in if sz > 0 else -layer.z_out
                        r0 = layer.r_in if sr > 0 else -layer.r_out
                        path = QtGui.QPainterPath()
                        path.addRect(QtCore.QRectF(z0, r0, layer.z_out - layer.z_in, layer.r_out - layer.r_in))
                        self._shape(self.rz, path, col, alpha)
        chamber = "#cfd8dc" if dark else self.OUTER_COLORS["chambers"]
        for st in od.stations:
            if st.kind == "barrel":
                self._shape(self.xy, self._ring(st.pos - 0.04, st.pos + 0.04, True), chamber, 160, 1.0)
                for s in (1, -1):
                    path = QtGui.QPainterPath(); path.addRect(QtCore.QRectF(-st.hi, s * st.pos - 0.04, 2 * st.hi, 0.08))
                    self._shape(self.rz, path, chamber, 160)
            else:
                for sz in (1, -1):
                    for sr in (1, -1):
                        r0 = st.lo if sr > 0 else -st.hi
                        path = QtGui.QPainterPath(); path.addRect(QtCore.QRectF(sz * st.pos - 0.04, r0, 0.08, st.hi - st.lo))
                        self._shape(self.rz, path, chamber, 160)

    def draw_detector(self, geom):
        th = theme.current()
        if getattr(geom, "outer", None) is not None:
            self.draw_outer(geom.outer)
        t = np.linspace(0, 2 * np.pi, 240)
        for c in geom.cylinders:
            if c.group == "beam pipe":
                continue
            self.xy.plot(c.radius * np.cos(t), c.radius * np.sin(t), pen=pg.mkPen(th["detector_xy"]))
            for s in (1, -1):
                self.rz.plot([-c.half_length, c.half_length], [s * c.radius] * 2, pen=pg.mkPen(th["detector"], width=1.5))
        for d in geom.disks:
            for sr in (1, -1):
                self.rz.plot([d.z] * 2, [sr * d.r_min, sr * d.r_max], pen=pg.mkPen(th["detector"], width=1.5))
        self.xy.addItem(pg.TextItem("⊙ B", color=th["muted"], anchor=(0, 0)))

    def set_event(self, ev, geom, progress=None, span=None):
        self.ev, self.geom = ev, geom
        if progress is not None:
            self.progress = progress
        self.span = span
        th = theme.current()
        self.xy.clear(); self.rz.clear()
        self._tracks = []
        if ev is None:
            return
        if geom is not None:
            self.draw_detector(geom)
        if ev.is_data:
            self.caption.setText(f"<b>CMS 2011 data</b>, run {ev.run}, event {ev.event_number}: "
                                 f"m(μμ) = {ev.reco_mass:.3f} GeV · helices from the measured momenta, B = {ev.B:g} T · "
                                 f"<span style='color:{th['crossing']}'>○</span> computed layer crossings (not CMS hits)")
        else:
            self.caption.setText(f"<b>Simulation</b>: {label(ev.process)} → μ⁺μ⁻, reconstructed m = {ev.reco_mass:.3f} GeV, "
                                 f"B = {ev.B:g} T")
        for t in ev.tracks:
            col = th["reco"][t.label]
            dashed = th["true"][t.label] if th["name"] == "dark" else col
            path, times = t.flight_path
            side = np.sign(path[-1, 1]) or 1.0
            rec = {"t": t, "side": side, "path": path, "times": times, "items": [],
                   "head_path": path, "head_t": times}
            if t.outer is not None and len(t.outer.path) > 1:
                rec["head_path"] = np.vstack([path, t.outer.path[1:]])
                rec["head_t"] = np.concatenate([times, t.outer.t[1:]])
            for plot, proj in ((self.xy, "xy"), (self.rz, "rz")):
                it = {
                    "true": plot.plot(pen=pg.mkPen(dashed, width=1, style=QtCore.Qt.DashLine)),
                    "reco": plot.plot(pen=pg.mkPen(col, width=2)),
                    "flight": plot.plot(pen=pg.mkPen(col, width=3)),
                    "head": plot.plot(pen=None, symbol="o", symbolSize=11, symbolBrush=col, symbolPen=th["fg"]),
                    "hits": plot.plot(pen=None, symbol="o", symbolSize=5, symbolBrush=th["hit"], symbolPen=None),
                    "missed": plot.plot(pen=None, symbol="x", symbolSize=8, symbolPen=th["missed"]),
                    "cross": plot.plot(pen=None, symbol="o", symbolSize=7, symbolBrush=None,
                                       symbolPen=pg.mkPen(th["crossing"], width=1.5)),
                    "flash": plot.plot(pen=None, symbol="o", symbolSize=10, symbolBrush=th["flash"], symbolPen=None),
                }
                if t.outer is not None:
                    it["outer"] = plot.plot(pen=pg.mkPen(col, width=2.5))
                    it["stations"] = plot.plot(pen=None, symbol="s", symbolSize=9, symbolBrush=th["flash"],
                                               symbolPen=pg.mkPen(th["fg"], width=0.8))
                    it["stop"] = plot.plot(pen=None, symbol="s", symbolSize=12, symbolBrush=th["bad"], symbolPen=None)
                if proj == "xy":
                    o, d, L = momentum_arrow(t.p_reco, ev.vertex)
                    dt = d[:2] / max(np.linalg.norm(d[:2]), 1e-12)
                    it["arrow"] = plot.plot([o[0], o[0] + L * dt[0]], [o[1], o[1] + L * dt[1]], pen=pg.mkPen(col, width=4))
                rec["items"].append((proj, it))
            self._tracks.append(rec)
        self._set_views(geom)
        self._apply()

    def _set_views(self, geom):
        """Default views fit the tracker; zooming and panning stay inside it (double-click resets)."""
        if geom is not None and getattr(geom, "outer", None) is not None:
            R, Z = 7.6, 10.9
            # the transverse view keeps equal scales on both axes, so in a tall narrow plot the
            # whole ring needs a much taller y range: its limits are generous (the aspect lock
            # already prevents distortion)
            xy_lim = (-3 * R, 3 * R)
            if self.zoom == "whole":
                geometry_view(self.xy, (-R, R), (-R, R), xy_lim, xy_lim)
                geometry_view(self.rz, (-Z, Z), (-R, R), (-1.15 * Z, 1.15 * Z), (-1.2 * R, 1.2 * R))
            else:
                geometry_view(self.xy, (-1.27, 1.27), (-1.27, 1.27), xy_lim, xy_lim)
                geometry_view(self.rz, (-3.0, 3.0), (-1.27, 1.27), (-1.15 * Z, 1.15 * Z), (-1.2 * R, 1.2 * R))
            return
        r_out, z_out = 1.1, 2.8
        if geom is not None:
            radii = [c.radius for c in geom.cylinders if np.isfinite(c.radius)]
            zs = [c.half_length for c in geom.cylinders if np.isfinite(c.half_length)]
            zs += [abs(d.z) for d in geom.disks if np.isfinite(d.z)]
            radii += [d.r_max for d in geom.disks if np.isfinite(d.r_max)]
            r_out = max(radii, default=r_out)
            z_out = max(zs, default=z_out)
        r, z = 1.15 * r_out, 1.07 * z_out
        geometry_view(self.xy, (-r, r), (-r, r), (-1.8 * r, 1.8 * r), (-1.8 * r, 1.8 * r))
        geometry_view(self.rz, (-z, z), (-r, r), (-1.4 * z, 1.4 * z), (-1.6 * r, 1.6 * r))

    # ---- animation ----------------------------------------------------------------
    def set_progress(self, fraction, span=None):
        self.progress = min(max(float(fraction), 0.0), 1.0)
        if span is not None:
            self.span = span
        self._apply()

    @staticmethod
    def _proj(a, proj, side):
        a = np.asarray(a).reshape(-1, 3)
        if proj == "xy":
            return a[:, 0], a[:, 1]
        return a[:, 2], side * np.hypot(a[:, 0], a[:, 1])

    def _apply(self):
        if self.ev is None:
            return
        final = self.progress >= 1.0
        t_cut = self.progress * (self.span or self.ev.t_max)
        # tracks are fitted once both muons have left the tracker (they may still be flying outwards)
        fitted = bool(final or (self.ev.has_outer and t_cut >= self.ev.t_tracker))
        empty = (np.empty(0), np.empty(0))

        def shown(points, times):
            # points are stored in flight order, so the first n are those already crossed
            if final or not len(points):
                return points
            return points[:int(np.sum(np.asarray(times) <= t_cut))]

        for rec in self._tracks:
            t, side = rec["t"], rec["side"]
            flown = None if fitted else path_until(rec["path"], rec["times"], t_cut)
            head = None if final else path_until(rec["head_path"], rec["head_t"], t_cut)[-1:]
            hits, missed, cross = shown(t.hits, t.hit_t), shown(t.missed, t.missed_t), shown(t.crossings, t.crossing_t)
            fresh = [np.empty((0, 3))]
            if not final:
                for pts, times in ((t.hits, t.hit_t), (t.crossings, t.crossing_t)):
                    if len(pts):
                        tt = np.asarray(times)
                        fresh.append(pts[(tt <= t_cut) & (tt > t_cut - FLASH_NS)])
            fresh = np.vstack(fresh)
            for proj, it in rec["items"]:
                P = lambda a: self._proj(a, proj, side) if len(a) else empty
                it["true"].setData(*P(t.true_path)); it["true"].setVisible(fitted and self.show_true and t.has_truth)
                it["reco"].setData(*P(t.reco_path)); it["reco"].setVisible(fitted)
                if "arrow" in it:
                    it["arrow"].setVisible(fitted)
                it["flight"].setVisible(not fitted); it["head"].setVisible(not final)
                if not fitted:
                    it["flight"].setData(*P(flown))
                if not final:
                    it["head"].setData(*P(head))
                if "outer" in it:
                    op = t.outer
                    part = op.path if final else path_until(op.path, op.t, t_cut)
                    it["outer"].setData(*P(part)); it["outer"].setVisible(len(part) > 1)
                    n = len(op.stations) if final else int(np.sum(op.station_t <= t_cut))
                    it["stations"].setData(*P(op.station_points[:n]))
                    done = bool(op.stopped_in is not None and (final or t_cut >= op.t[-1]))
                    it["stop"].setData(*P(op.path[-1:])); it["stop"].setVisible(done)
                it["hits"].setData(*P(hits)); it["missed"].setData(*P(missed)); it["cross"].setData(*P(cross))
                it["flash"].setData(*P(fresh)); it["flash"].setVisible(not final)


class EventTab(QtWidgets.QWidget):
    """2D projections (x–y and z–r) of the shared display event, animated, optionally above a CMS data event."""
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.shown_event = None
        self.clock = getattr(main, "clock", None) or AnimationClock(self)
        nxt = QtWidgets.QPushButton("Next event")
        self.show_true = QtWidgets.QCheckBox("true trajectory"); self.show_true.setChecked(True)
        self.show_true.setToolTip("Show the true (simulated) trajectories as dashed lines")
        self.compare = QtWidgets.QCheckBox("compare with CMS data")
        self.compare.setToolTip("Show a simulated and a real CMS 2011 event of the selected resonance one above the other")
        self.compare.setEnabled(cms.available())
        save = QtWidgets.QPushButton("Save image")
        save.setToolTip("Save the event views as a PNG")
        self.view = EventView2D()
        self.other = EventView2D(); self.other.hide()
        self.xy, self.rz = self.view.xy, self.view.rz         # the primary view's plots
        self.feyn = FeynmanWidget()
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True)
        self.info.setStyleSheet("font-family: Consolas, monospace;")
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top = FlowLayout()                      # wraps onto a second row in a narrow window
        self.zoom = QtWidgets.QComboBox()
        self.zoom.addItem("Whole detector", "whole"); self.zoom.addItem("Tracker", "tracker")
        self.zoom.setToolTip("Zoom of both views (the outer detector is switched on in the View menu)")
        self.zoom.currentIndexChanged.connect(self._zoom_changed)
        for w in (nxt, self.show_true, self.compare, self.zoom, save):
            top.addWidget(w)
        self.legend = QtWidgets.QLabel(); self.legend.setWordWrap(True)
        self.bar = AnimationBar(self.clock)
        views = QtWidgets.QVBoxLayout(); views.addWidget(self.view, 1); views.addWidget(self.other, 1)
        side_w = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(side_w); side.setContentsMargins(0, 0, 4, 0)
        side.addWidget(QtWidgets.QLabel("<b>Lowest-order diagram</b>"))
        self.contrast = QtWidgets.QLabel(DIAGRAM_VS_TRACKS); self.contrast.setWordWrap(True)
        side.addWidget(self.feyn); side.addWidget(self.contrast); side.addWidget(self.info); side.addStretch()
        self.side_area = QtWidgets.QScrollArea()
        self.side_area.setWidget(side_w); self.side_area.setWidgetResizable(True)
        self.side_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.side_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.side_area.setMinimumWidth(em(self, 13))
        # the event views and the side panel share the width; the side panel can be dragged closed
        self.body = splitter("event2d", [wrap(views, margins=False), self.side_area], [4, 1], collapsible=(1,),
                             responsive=True)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.legend); lay.addWidget(self.bar)
        lay.addWidget(self.body, 1)
        nxt.clicked.connect(lambda: self.main.new_event())
        self.show_true.toggled.connect(self._true_toggled)
        self.compare.toggled.connect(self._compare_toggled)
        save.clicked.connect(self.save_image_dialog)
        self.clock.frame.connect(self._frame)
        self.theme_changed()

    def _span(self):
        return self.clock.t_max or None

    def _zoom_changed(self, _):
        for v in (self.view, self.other):
            v.zoom = self.zoom.currentData()
            if v.ev is not None:
                v._set_views(v.geom)

    def show_event(self, ev):
        if ev is None:
            return
        self.shown_event = ev
        self.zoom.setEnabled(ev.has_outer)
        if getattr(self.main, "clock", None) is None:      # standalone use: own clock
            self.clock.set_span(ev.t_max)
        self.feyn.set_process(ev.process)
        self.view.set_event(ev, getattr(self.main, "geometry", None), self.clock.fraction, self._span())
        other = getattr(self.main, "companion", None) if self.compare.isChecked() else None
        self.other.setVisible(other is not None)
        if other is not None:
            self.other.set_event(other, getattr(self.main, "companion_geometry", None), self.clock.fraction, self._span())
        lines = event_info_lines(ev)
        if other is not None:
            lines += [""] + event_info_lines(other)
        self.info.setText("\n".join(lines))

    def _frame(self, f):
        if self.isVisible():
            for v in (self.view, self.other):
                if v.ev is not None and not v.isHidden():
                    v.set_progress(f, self._span())

    def showEvent(self, e):
        super().showEvent(e)
        self._frame(self.clock.fraction)

    def _true_toggled(self, on):
        for v in (self.view, self.other):
            v.show_true = on
            v._apply()

    def _compare_toggled(self, on):
        if on and hasattr(self.main, "new_event"):
            self.main.new_event()           # draws a matching companion event
        elif not on:
            self.other.hide()

    def theme_changed(self):
        th = theme.current()
        self.contrast.setStyleSheet(f"background:{th['note_bg']}; color:{th['note_fg']}; border:1px solid "
                                    f"{th['note_border']}; padding:6px; font-size:9pt;")
        self.legend.setText(f"dots: recorded hits · crosses: crossed, not recorded · solid: fitted track · "
                            f"dashed: true trajectory · <span style='color:{th['flash']}'>●</span> hit just made")
        for v in (self.view, self.other):
            if v.ev is not None:
                v.set_event(v.ev, v.geom, v.progress, v.span)
        self.feyn.update()

    def save_image(self, path):
        self.body.grab().save(str(path), "PNG")
        return path

    def save_image_dialog(self):
        ev = self.shown_event
        name = "event2d.png" if ev is None else f"event2d_{'cms' if ev.is_data else 'sim'}_{ev.process}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save event views", name.replace("/", ""), "PNG (*.png)")
        if path:
            self.save_image(path)


def resolution_scan(det, eta, progress=None):
    pts = np.array([3, 5, 10, 20, 50, 100, 200.0])
    rng = np.random.default_rng()
    full, noms = [], []
    for i, pt in enumerate(pts):
        full.append(pt_resolution(det, pt, eta, 8000, rng))
        noms.append(pt_resolution(det.with_(multiple_scattering=False), pt, eta, 8000, rng))
        if progress:
            progress(i + 1, len(pts))
    return pts, np.array(full), np.array(noms), det


class PerformanceTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.eta = QtWidgets.QDoubleSpinBox(); self.eta.setRange(0, 2.4); self.eta.setSingleStep(0.2); self.eta.setPrefix("|η| = ")
        run = QtWidgets.QPushButton("Measure momentum resolution")
        self.plot = pg.PlotWidget(labels={"bottom": "pT [GeV]", "left": "σ(pT)/pT [%]"})
        self.plot.setLogMode(True, True)
        self.plot.addLegend()
        empty_view(self.plot)
        self.eff_plot = pg.PlotWidget(labels={"bottom": "pT [GeV]", "left": "track reconstruction efficiency"})
        self.eff_plot.setLogMode(True, False)
        empty_view(self.eff_plot)
        for p in (self.plot, self.eff_plot):
            p.setMinimumSize(em(self, 12), em(self, 8))
        self.cutflow = QtWidgets.QTableWidget(0, 3)
        self.cutflow.setHorizontalHeaderLabels(["step", "events", "fraction"])
        hh = self.cutflow.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in (1, 2):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.cutflow.verticalHeader().setVisible(False)
        row = em(self, 1.45)
        self.cutflow.verticalHeader().setDefaultSectionSize(row)
        self.cutflow.setWordWrap(False)
        # header + the six cut-flow steps
        self.cutflow.setMinimumHeight(self.cutflow.horizontalHeader().sizeHint().height() + 6 * row + 6)
        top = QtWidgets.QHBoxLayout()
        for w in (self.eta, run):
            top.addWidget(w)
        run.setToolTip("Simulates single muons at several pT with the detector settings on the left")
        top.addStretch()
        box = QtWidgets.QGroupBox("Selection cut flow of the last simulation"); QtWidgets.QVBoxLayout(box).addWidget(self.cutflow)
        right = splitter("performance_right", [self.eff_plot, box], [1, 1], orientation=QtCore.Qt.Vertical)
        self.split = splitter("performance", [self.plot, right], [3, 2], responsive=True)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.split, 1)
        run.clicked.connect(self.run)

    def run(self):
        self.main.tasks.start(resolution_scan, self.show, self.main.setup.detector(), self.eta.value(),
                              text="Measuring resolution…")

    def theme_changed(self):
        if getattr(self, "last", None) is not None:
            self.show(self.last)

    def show(self, r):
        self.last = r
        pts, full, noms, det = r
        self.plot.clear(); self.eff_plot.clear()
        self.plot.plot(pts, 100 * full[:, 1], pen=pg.mkPen("#1f77b4", width=2), symbol="o", name="simulation (central 68%)")
        self.plot.plot(pts, 100 * noms[:, 1], pen=pg.mkPen("#1f77b4", style=QtCore.Qt.DashLine), symbol="s",
                       name="without multiple scattering")
        if all(np.isinf(b.half_length) for b in det.barrels):
            xx = np.logspace(np.log10(2), np.log10(250), 60)
            self.plot.plot(xx, 100 * gluckstern(xx, det), pen=pg.mkPen("#ff7f0e", width=2), name="Gluckstern formula")
        self.eff_plot.plot(pts, full[:, 2], pen=pg.mkPen(fg()), symbol="o", symbolBrush=fg())
        x_range = (float(np.min(pts)), float(np.max(pts)))
        curve_view(self.plot, x_range, x_log=True, y_log=True)
        curve_view(self.eff_plot, x_range, y_range=(0, 1.05), x_log=True)
        ys = 100 * np.concatenate([full[:, 1], noms[:, 1]])
        ys = ys[np.isfinite(ys) & (ys > 0)]
        for p in (self.plot, self.eff_plot):
            log_ticks(p.getAxis("bottom"), *x_range)
        if ys.size:
            log_ticks(self.plot.getAxis("left"), ys.min(), ys.max())

    def show_cutflow(self, sim):
        items = list(sim.cutflow.items())
        self.cutflow.setRowCount(len(items))
        for i, (k, v) in enumerate(items):
            for j, val in enumerate((k, str(v), f"{v / max(items[0][1], 1):.2%}")):
                item = QtWidgets.QTableWidgetItem(val)
                item.setToolTip(k)
                self.cutflow.setItem(i, j, item)
        pre = getattr(sim, "preselection", None)
        if pre:
            self.cutflow.setToolTip(
                f"Pre-selection: events with a true muon below pT {pre['true_pt_min']:.2f} GeV or beyond "
                f"|eta| {pre['true_abs_eta_max']:.2f} were not simulated in the detector. "
                f"{pre['checked']} of the {pre['skipped']} skipped events were simulated anyway as a check; "
                f"{pre['checked_selected']} of them would have been selected "
                f"(estimated loss {pre['estimated_lost']:.1f} events).")


# ---------------------------------------------------------------------------
class ExplorerTab(QtWidgets.QWidget):
    def __init__(self, main=None):
        super().__init__()
        self.main = main
        # particle table
        rows = [p for p in DB.particles.values() if not p.id.startswith("anti-") and p.id not in ("mu+", "e+", "tau+", "pi-", "K-")]
        self.table = QtWidgets.QTableWidget(len(rows), 8)
        self.table.setHorizontalHeaderLabels(["particle", "mass [GeV]", "± [GeV]", "width [GeV]", "BR(μμ)",
                                              "Q, B, Le, Lμ, Lτ", "spin", "checked vs PDG Live"])
        for i, p in enumerate(rows):
            vals = [p.label, f"{p.mass:.7g}", f"{p.mass_err:.2g}" if p.mass_err else "", f"{p.width:.4g}" if p.width else "",
                    f"{p.br_mumu:.4g}" if p.br_mumu else "", f"{p.charge:+d}, {p.baryon}, {p.Le}, {p.Lmu}, {p.Ltau}",
                    f"{p.spin:g}", "yes" if p.verified else "no"]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(em(self, 9))      # header and about four rows
        src = QtWidgets.QLabel(f"Source: {DB.source}; retrieved {DB.retrieved}.")
        src.setWordWrap(True)
        # conservation checker
        self.edit = QtWidgets.QLineEdit("Z -> mu+ mu-")
        self.edit.setMinimumWidth(em(self, 9))
        self.examples = QtWidgets.QComboBox(); self.examples.addItems(EXAMPLES)
        btn = QtWidgets.QPushButton("Check")
        self.use = QtWidgets.QPushButton("Simulate this process"); self.use.setEnabled(False)
        self.out = QtWidgets.QTextEdit(); self.out.setReadOnly(True)
        self.out.setMinimumHeight(em(self, 10))
        self.examples.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.examples.setMinimumContentsLength(14)
        form = QtWidgets.QGridLayout()
        form.addWidget(QtWidgets.QLabel("Decay:"), 0, 0); form.addWidget(self.edit, 0, 1); form.addWidget(btn, 0, 2)
        form.addWidget(QtWidgets.QLabel("Examples:"), 1, 0); form.addWidget(self.examples, 1, 1, 1, 2)
        form.setColumnStretch(1, 1)
        # two-body decay demo
        self.dproc = QtWidgets.QComboBox()
        for k in RESONANCES:
            self.dproc.addItem(label(k), k)
        self.beta = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.beta.setRange(0, 99); self.beta.setValue(60)
        self.cos = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.cos.setRange(-100, 100); self.cos.setValue(50)
        self.demo = QtWidgets.QLabel(); self.demo.setStyleSheet("font-family: Consolas, monospace;")
        dbox = QtWidgets.QGroupBox("Two-body decay: rest frame → lab (boost along z)")
        dl = QtWidgets.QFormLayout(dbox)
        dl.addRow("parent", self.dproc); dl.addRow("parent velocity β", self.beta)
        dl.addRow("cos θ* of μ⁻ (rest frame)", self.cos); dl.addRow(self.demo)
        left = QtWidgets.QVBoxLayout(); left.addWidget(QtWidgets.QLabel("<b>Particle properties</b>"))
        left.addWidget(self.table, 1); left.addWidget(src); left.addWidget(dbox)
        right = QtWidgets.QVBoxLayout(); right.addWidget(QtWidgets.QLabel("<b>Conservation-law checker</b>"))
        right.addLayout(form); right.addWidget(self.use); right.addWidget(self.out, 1)
        self.split = splitter("explorer", [wrap(left, margins=False), wrap(right, margins=False)], [3, 2],
                              responsive=True)
        lay = QtWidgets.QHBoxLayout(self); lay.addWidget(self.split)
        btn.clicked.connect(self.run_check); self.edit.returnPressed.connect(self.run_check)
        self.use.clicked.connect(self.use_process)
        self.examples.textActivated.connect(lambda t: (self.edit.setText(t), self.run_check()))
        for w in (self.beta, self.cos):
            w.valueChanged.connect(self.update_demo)
        self.dproc.currentIndexChanged.connect(self.update_demo)
        self.run_check(); self.update_demo()

    def run_check(self):
        self._gate = gate(self.edit.text())
        self.use.setEnabled(self._gate.simulable and self.main is not None)
        try:
            v = check(self.edit.text())
        except (KeyError, ValueError) as e:
            self.out.setHtml(f"<p style='color:{theme.current()['bad']}'>{e}</p>")
            return
        th = theme.current()
        colour = th["ok"] if v.allowed else th["bad"]
        html = [f"<h3 style='color:{colour}'>{v.category}</h3><table cellpadding=3>",
                "<tr><th align=left>law</th><th>kind</th><th>before</th><th>after</th><th>result</th></tr>"]
        for c in v.checks:
            fmt = (lambda x: f"{x:.4g}") if c.kind == "kinematic" else (lambda x: f"{x:+d}")
            res = "ok" if c.passed else f"<b style='color:{th['bad']}'>fails</b>"
            html.append(f"<tr><td>{c.law}</td><td>{c.kind}</td><td align=center>{fmt(c.before)}</td>"
                        f"<td align=center>{fmt(c.after)}</td><td>{res}</td></tr>")
        html.append(f"</table><p style='color:{th['muted']}'>exact: never violated · accidental: conserved in the "
                    "Standard Model, never seen violated · approximate: lepton flavour (violated by neutrino "
                    "oscillations) · kinematic: energy–momentum. Passing all checks is necessary, not "
                    "sufficient: spin, parity, C and dynamics are not checked.</p>")
        g = self._gate
        html.append(f"<p><b>Simulation:</b> <span style='color:{th['ok'] if g.simulable else th['bad']}'>{g.message}</span></p>")
        self.out.setHtml("".join(html))

    def theme_changed(self):
        self.run_check()

    def use_process(self):
        g = self._gate
        if self.main is None or not g.simulable:
            return
        self.main.setup.select_process(g.process)
        self.main.tabs.setCurrentWidget(self.main.mass)
        self.main.set_status(f"{label(g.process)} → μ⁺μ⁻ selected: press Run simulation", busy=False)

    def update_demo(self):
        name = self.dproc.currentData()
        m = DB.get(name).mass
        b = self.beta.value() / 100
        c = self.cos.value() / 100
        s = np.sqrt(1 - c * c)
        from ..physics.kinematics import M_MU
        p = np.sqrt(m**2 / 4 - M_MU**2)
        mum = np.array([[m / 2, p * s, 0, p * c]]); mup = np.array([[m / 2, -p * s, 0, -p * c]])
        beta = np.array([[0, 0, b]])
        lm, lp = boost(mum, beta)[0], boost(mup, beta)[0]
        from ..physics import inv_mass
        mm = inv_mass(lm[None], lp[None])[0]
        f = lambda v: f"({v[0]:7.3f},{v[1]:7.3f},{v[2]:7.3f},{v[3]:7.3f})"
        angle = np.degrees(np.arccos(np.dot(lm[1:], lp[1:]) / np.linalg.norm(lm[1:]) / np.linalg.norm(lp[1:])))
        self.demo.setText(f"(E, px, py, pz) in GeV\n"
                          f"rest μ⁻ {f(mum[0])}\n     μ⁺ {f(mup[0])}\n"
                          f"lab  μ⁻ {f(lm)}\n     μ⁺ {f(lp)}\n"
                          f"Σ lab  {f(lm + lp)}\n"
                          f"invariant mass {mm:.6f} GeV (input {m:.6f})\n"
                          f"opening angle in lab {angle:.1f}° at β = {b:.2f}")


# ---------------------------------------------------------------------------
class _ScrollPage(QtWidgets.QWidget):
    """Holder for a scrolled tab page, without a layout of its own.

    QScrollArea sizes a page with a height-for-width layout (word-wrapped labels) from
    its preferred height and then scrolls although the minimum fits. This holder reports
    the content's minimum size as both minimum and preferred size and fills its area with
    the content, so a page scrolls only when the window is smaller than its minimum."""

    def __init__(self, content):
        super().__init__()
        self._content = content
        content.setParent(self)
        content.installEventFilter(self)
        QtWidgets.QWidget.setVisible(content, True)   # some tabs define their own show()

    def minimumSizeHint(self):
        m = min_size(self._content)
        return QtCore.QSize(m.width(), min_height_for_width(self._content, self.width()))

    def sizeHint(self):
        return self.minimumSizeHint()

    def resizeEvent(self, e):
        self._content.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(e)
        if e.oldSize().width() != e.size().width():
            self.updateGeometry()          # wrapped text needs a different height at the new width

    def eventFilter(self, obj, e):
        if obj is self._content and e.type() == QtCore.QEvent.LayoutRequest:
            self.updateGeometry()
        return False


class ScrollTabs(QtWidgets.QTabWidget):
    """Tab widget whose pages scroll when the window is smaller than their content.

    ``widget``, ``currentWidget``, ``setCurrentWidget`` and ``indexOf`` work with the
    tab contents, not with the scroll areas around them. Pages added with
    ``scroll=False`` (the 3D view, whose renderer does not scroll) are added directly."""

    def addTab(self, widget, label, scroll=True):
        if not scroll:
            return super().addTab(widget, label)
        area = QtWidgets.QScrollArea()
        area.setWidget(_ScrollPage(widget))
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.NoFrame)
        area._content = widget
        return super().addTab(area, label)

    def widget(self, i):
        page = super().widget(i)
        return getattr(page, "_content", page)

    def currentWidget(self):
        page = super().currentWidget()
        return getattr(page, "_content", page)

    def indexOf(self, widget):
        for i in range(self.count()):
            if self.widget(i) is widget:
                return i
        return -1

    def setCurrentWidget(self, widget):
        i = self.indexOf(widget)
        if i >= 0:
            self.setCurrentIndex(i)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: QtCore.QSettings | None = None):
        super().__init__()
        self.setWindowTitle("Particle Physics Lab: dimuon resonances")
        self.settings = settings or QtCore.QSettings("particle-physics-lab", "desktop")
        self.tasks = TaskRunner(self)
        self.event_tasks = TaskRunner(self, report=False)
        self.last_sim = None
        self.current_event = None
        self.geometry = None
        self.companion = None              # second event for the 2D comparison (data if the main one is simulated)
        self.companion_geometry = None
        self._event_listeners = []
        self._data = None
        self._display_events = None        # selected CMS events, loaded once for the event displays
        self._event_again = None
        self.clock = AnimationClock(self, autoplay=self.settings.value("animate_new_events", True, type=bool))
        self.clock.autoplay_changed.connect(lambda on: self.settings.setValue("animate_new_events", on))
        self.setup = SetupPanel()
        self.setup.source.currentIndexChanged.connect(lambda _: self.new_event())
        self.mass = MassTab(self)
        self.event2d = EventTab(self)
        self.performance = PerformanceTab(self)
        tabs = ScrollTabs()
        tabs.addTab(self.mass, "Mass spectrum")
        tabs.addTab(self.event2d, "Event display (2D)")
        from .view3d import Event3DTab
        self.event3d = Event3DTab(self)
        tabs.addTab(self.event3d, "Event display (3D)", scroll=False)
        self.add_event_listener(self.event2d.show_event)
        self.add_event_listener(self.event3d.show_event)
        self.reconstruct = ReconstructTab(self)
        tabs.addTab(self.reconstruct, "Reconstruct it yourself")
        self.add_event_listener(self.reconstruct.show_event)
        tabs.addTab(self.performance, "Detector performance")
        tabs.addTab(ExplorerTab(self), "Physics explorer")
        self.tabs = tabs
        self.setup_area = QtWidgets.QScrollArea(); self.setup_area.setWidget(self.setup)
        self.setup_area.setWidgetResizable(True)
        self.setup_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setup_area.setMinimumWidth(self.setup.minimumSizeHint().width()
                                        + self.setup_area.verticalScrollBar().sizeHint().width() + 4)
        # the settings panel and the tabs share the width; the panel can be dragged closed or hidden (Ctrl+B)
        self.main_split = splitter("main", [self.setup_area, tabs], [0, 1], collapsible=(0,))
        self.setCentralWidget(self.main_split)
        self.progress = QtWidgets.QProgressBar(); self.progress.setMaximumWidth(em(self, 12)); self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.set_status("Ready", busy=False)
        self._build_menus()
        self.setMinimumSize(em(self, 30), em(self, 20))
        if not self.restore_layout():
            self.fit_to_screen()
        self.set_theme("dark" if self.settings.value("theme", "light") == "dark" else "light")
        self.new_event()

    PREFERRED_SIZE = (1450, 860)

    def fit_to_screen(self):
        """Open at the preferred size, or maximised when the screen is smaller than that."""
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(*self.PREFERRED_SIZE)
            return False
        avail = screen.availableGeometry()
        w, h = self.PREFERRED_SIZE
        small = avail.width() < w + 40 or avail.height() < h + 60
        self.resize(min(w, avail.width() - 20), min(h, avail.height() - 60))
        self.move(avail.x() + (avail.width() - self.width()) // 2, avail.y() + max(0, (avail.height() - self.height()) // 2 - 20))
        self.start_maximized = small
        return small

    def show(self):
        """Shows the window maximised on small screens (see fit_to_screen)."""
        if getattr(self, "start_maximized", False):
            self.showMaximized()
        else:
            super().show()

    def splitters(self):
        return [s for s in self.findChildren(QtWidgets.QSplitter) if s.objectName()]

    def restore_layout(self) -> bool:
        """Window geometry and pane sizes from the last session, if the window still fits a screen."""
        geo = self.settings.value("window/geometry")
        ok = False
        if isinstance(geo, (QtCore.QByteArray, bytes)) and self.restoreGeometry(QtCore.QByteArray(geo)):
            frame = self.frameGeometry()
            # Qt moves a restored window back onto a screen; refuse only if it is on none of them
            ok = any(sc.availableGeometry().intersects(frame) for sc in QtWidgets.QApplication.screens())
            if not ok:
                self.setWindowState(QtCore.Qt.WindowNoState)
        for s in self.splitters():
            state = self.settings.value(f"splitter/{s.objectName()}")
            if isinstance(state, (QtCore.QByteArray, bytes)):
                s.restoreState(QtCore.QByteArray(state))
                s._restored = True
        hidden = self.settings.value("window/settings_panel_hidden", False, type=bool)
        self.panel_action.setChecked(not hidden)
        return ok

    def save_layout(self):
        self.settings.setValue("window/geometry", self.saveGeometry())
        for s in self.splitters():
            self.settings.setValue(f"splitter/{s.objectName()}", s.saveState())
        self.settings.setValue("window/settings_panel_hidden", not self.panel_action.isChecked())

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_panes_sized", False):
            self._panes_sized = True
            QtCore.QTimer.singleShot(0, self._initial_pane_sizes)

    def _initial_pane_sizes(self):
        """First start: panes with stretch 0 get their preferred size, the others share the rest."""
        for s in self.splitters():
            if getattr(s, "_restored", False) or not hasattr(s, "_stretch"):
                continue
            horizontal = s.orientation() == QtCore.Qt.Horizontal
            total = (s.width() if horizontal else s.height()) - s.handleWidth() * (s.count() - 1)
            if total <= 0:
                continue
            k = s._stretch
            hint = lambda w: max(w.sizeHint().width() if horizontal else w.sizeHint().height(),
                                 w.minimumSizeHint().width() if horizontal else w.minimumSizeHint().height())
            fixed = sum(min(hint(s.widget(i)), total // 2) for i in range(s.count()) if k[i] == 0)
            rest = max(total - fixed, 0)
            sizes = [min(hint(s.widget(i)), total // 2) if k[i] == 0 else int(rest * k[i] / max(sum(k), 1))
                     for i in range(s.count())]
            s.setSizes(sizes)

    # ---- menus, theme, images ------------------------------------------------------
    def _build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        save = file_menu.addAction("Save image of the current tab…")
        save.setShortcut(QtGui.QKeySequence.Save)
        save.triggered.connect(self.save_image_dialog)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)
        view = self.menuBar().addMenu("&View")
        self.dark_action = view.addAction("Dark theme")
        self.dark_action.setCheckable(True)
        self.dark_action.setShortcut("Ctrl+D")
        self.dark_action.toggled.connect(lambda on: self.set_theme("dark" if on else "light"))
        self.outer_action = view.addAction("Outer detector and muon system")
        self.outer_action.setCheckable(True)
        self.outer_action.setChecked(self.settings.value("outer_detector", True, type=bool))
        self.outer_action.setShortcut("Ctrl+M")
        self.outer_action.setToolTip("Follow the muons through the calorimeters, the coil and the iron to the "
                                     "muon chambers (schematic, display only)")
        self.outer_action.toggled.connect(self._outer_toggled)
        self.panel_action = view.addAction("Settings panel")
        self.panel_action.setCheckable(True)
        self.panel_action.setChecked(True)
        self.panel_action.setShortcut("Ctrl+B")
        self.panel_action.toggled.connect(lambda on: self.setup_area.setVisible(on))
        reset = view.addAction("Reset window layout")
        reset.triggered.connect(self.reset_layout)
        view.addSeparator()
        play = view.addAction("Play / pause event animation")
        play.setShortcut("Ctrl+P")
        play.triggered.connect(self.clock.toggle)

    def set_theme(self, name: str):
        app = QtWidgets.QApplication.instance()
        theme.apply(app, self, name)
        self.settings.setValue("theme", name)
        if self.dark_action.isChecked() != (name == "dark"):
            self.dark_action.blockSignals(True); self.dark_action.setChecked(name == "dark"); self.dark_action.blockSignals(False)

    def save_image(self, path, widget=None):
        """Save the current tab (or ``widget``) as a PNG; the 3D tab uses the renderer's own screenshot."""
        w = widget or self.tabs.currentWidget()
        if hasattr(w, "save_image"):
            return w.save_image(path)
        w.grab().save(str(path), "PNG")
        return path

    def save_image_dialog(self):
        name = self.tabs.tabText(self.tabs.currentIndex()).lower().replace(" ", "_").replace("(", "").replace(")", "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", f"{name}.png", "PNG (*.png)")
        if path:
            self.save_image(path)
            self.set_status(f"Saved {path}", busy=False)

    def _outer_toggled(self, on):
        self.settings.setValue("outer_detector", on)
        self.new_event()

    def outer_on(self) -> bool:
        return self.outer_action.isChecked() if hasattr(self, "outer_action") else False

    def reset_layout(self):
        """Default pane sizes and window size (also forgets the saved layout)."""
        for key in ("window/geometry", "window/settings_panel_hidden"):
            self.settings.remove(key)
        for s in self.splitters():
            self.settings.remove(f"splitter/{s.objectName()}")
            s._restored = False
        self.panel_action.setChecked(True)
        self.showNormal()
        if self.fit_to_screen():
            self.showMaximized()
        QtCore.QTimer.singleShot(0, self._initial_pane_sizes)
        for p in self.findChildren(pg.PlotWidget):
            if hasattr(p, "reset_view"):
                p.reset_view()

    def closeEvent(self, e):
        self.save_layout()
        self.clock.pause()
        self.event3d.close_renderer()        # a tab's own closeEvent is not called when the window closes
        super().closeEvent(e)

    # ---- shared display event ------------------------------------------------
    def add_event_listener(self, fn):
        self._event_listeners.append(fn)

    def _events_for_display(self):
        """Selected CMS events (loaded once, in the worker thread)."""
        if self._display_events is None:
            self._display_events = cms.load(check_file=False).events
        return self._display_events

    def new_event(self, seed=None):
        """Generate the display event(s) with the current settings in a worker thread.

        The main event follows the "Event displays" source; with the 2D comparison on,
        a companion event of the other kind is made for the same resonance.
        """
        det = self.setup.detector()
        name = self.setup.proc.currentData()
        rng = np.random.default_rng(seed)
        have_data = cms.available()
        source = self.setup.source.currentData() if have_data else "sim"
        compare = self.event2d.compare.isChecked() and have_data
        data_det = cms_inspired().with_(B=DATA_B)
        outer = self.outer_on()

        def job(progress=None):
            sim = make_event(name, det, rng, outer=outer) if (source == "sim" or compare) else None
            dat = None
            if source == "data" or compare:
                dat = data_event(pick_data_row(self._events_for_display(), name, rng), name, data_det, outer=outer)
            main_ev = dat if source == "data" else sim
            other = (sim if source == "data" else dat) if compare else None
            return main_ev, other, det, data_det

        started = self.event_tasks.start(job, self._event_ready)
        if not started:                     # one is running: make another when it finishes
            self._event_again = (seed,)
        return started

    def _event_ready(self, result):
        ev, other, det, data_det = result
        geom = lambda e: None if e is None else detector_geometry(data_det if e.is_data else det, outer=e.has_outer)
        self.geometry, self.companion, self.companion_geometry = geom(ev), other, geom(other)
        self.current_event = ev
        self.clock.set_span(max(e.t_max for e in (ev, other) if e is not None))
        if self.clock.autoplay:
            self.clock.pause(); self.clock.fraction = 0.0
        for fn in self._event_listeners:
            fn(ev)
        self.clock.new_event(self.clock.t_max)
        if self._event_again is not None:
            (seed,), self._event_again = self._event_again, None
            QtCore.QTimer.singleShot(0, lambda: self.new_event(seed))

    def data_masses(self):
        if self._data is None and cms.available():
            s = self.setup
            self._data = cms.load(selection=cms.Selection(pt_min=s.ptmin.value(), eta_max=s.etamax.value()),
                                  check_file=False).masses
        return self._data

    def set_status(self, text, busy):
        self.statusBar().showMessage(text)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)

    def set_progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

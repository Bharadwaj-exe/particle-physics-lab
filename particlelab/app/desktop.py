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
from ..visualization.event_display import make_event, detector_geometry, momentum_arrow
from ..education import from_display_event, gate
from ..units import C_TRACK, um

pg.setConfigOptions(antialias=True, background="w", foreground="k")
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
        self.gate_label = QtWidgets.QLabel(); self.gate_label.setWordWrap(True)
        self.gate_label.setStyleSheet("color:#080; font-size: 9pt;")
        self.proc.currentIndexChanged.connect(self._update_gate)
        self._update_gate()
        for lab, w in [("Process", self.proc), ("Generated events", self.n), ("Random seed", self.seed),
                       ("Detector", self.preset), ("Magnetic field", self.B), ("Hit-resolution scale", self.res),
                       ("Material scale", self.mat), ("Hit efficiency", self.eff), ("", self.ms), ("", self.smear),
                       ("Muon pT >", self.ptmin), ("Muon |η| <", self.etamax)]:
            f.addRow(lab, w)
        f.insertRow(1, self.gate_label)          # full width under the process choice

    def _update_gate(self):
        g = gate(f"{self.proc.currentData()} -> mu+ mu-")
        self.gate_label.setText(("✓ " if g.simulable else "✗ ") + g.message)
        self.gate_label.setStyleSheet(f"color:{'#080' if g.simulable else '#b00'}; font-size: 9pt;")

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
    sim = simulate(name, settings["events"], det, seed=settings["seed"], progress=progress)
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
        exp_png = QtWidgets.QPushButton("Export plot (PNG)")
        exp_csv = QtWidgets.QPushButton("Export masses (CSV)")
        exp_json = QtWidgets.QPushButton("Export run record (JSON)")
        self.plot = pg.PlotWidget(labels={"bottom": "m(μμ) [GeV]", "left": "events / bin"})
        self.plot.addLegend(offset=(-10, 10))
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["quantity", "simulation", "CMS data"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        buttons = QtWidgets.QHBoxLayout()
        for w in (run, self.overlay, exp_png, exp_csv, exp_json):
            buttons.addWidget(w)
        buttons.addStretch()
        right = QtWidgets.QVBoxLayout(); right.addWidget(self.table)
        body = QtWidgets.QHBoxLayout(); body.addWidget(self.plot, 3); body.addLayout(right, 2)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(buttons); lay.addLayout(body, 1)
        run.clicked.connect(self.run)
        exp_png.clicked.connect(self.export_png)
        exp_csv.clicked.connect(self.export_csv)
        exp_json.clicked.connect(self.export_json)

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
        name = r["settings"]["process"]
        self.plot.clear()
        rows = []
        sim = r["sim"]
        df = r.get("data_fit")
        sf = r.get("sim_fit")
        if df is not None:
            ctr = 0.5 * (df.edges[1:] + df.edges[:-1])
            self.plot.addItem(pg.ErrorBarItem(x=ctr, y=df.counts, height=2 * np.sqrt(df.counts), pen="k"))
            self.plot.plot(ctr, df.counts, pen=None, symbol="o", symbolSize=4, symbolBrush="k", name="CMS data")
            fine = np.linspace(*df.window, 1200)
            self.plot.plot(fine, df.curve(fine), pen=pg.mkPen("#ff7f0e", width=2), name=f"data fit ({df.signal})")
            self.plot.plot(fine, df.curve(fine, "background"), pen=pg.mkPen("#ff7f0e", width=1, style=QtCore.Qt.DashLine),
                           name="fitted background")
            hs, _ = np.histogram(sim.reco_mass, df.edges)
            n_in = np.sum((sim.reco_mass > df.window[0]) & (sim.reco_mass < df.window[1]))
            scale = df.n_signal / max(n_in, 1)
            self.plot.plot(df.edges, hs * scale + df.curve(ctr, "background"), stepMode="center",
                           pen=pg.mkPen(COLORS[name], width=2), name="simulation (scaled) + data bkg")
        elif sf is not None:
            hs, _ = np.histogram(sim.reco_mass, sf.edges)
            self.plot.plot(sf.edges, hs, stepMode="center", pen=pg.mkPen(COLORS[name], width=2), name="simulation")
            fine = np.linspace(*sf.window, 1200)
            self.plot.plot(fine, sf.curve(fine), pen=pg.mkPen("k", width=1.5, style=QtCore.Qt.DashLine),
                           name=f"fit ({sf.signal})")
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
        self.setMinimumSize(300, 200)

    def set_process(self, name):
        self.name = name
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("white"))
        w, h = self.width(), self.height() - 24
        v1, v2 = QtCore.QPointF(w * 0.35, h * 0.5), QtCore.QPointF(w * 0.65, h * 0.5)
        p.setPen(QtGui.QPen(QtGui.QColor("black"), 2))
        f = p.font(); f.setPointSize(11); p.setFont(f)
        quark = {"J/psi": "c", "Upsilon(1S)": "b", "Z": "q"}[self.name]

        def fermion(a, b, text, pos):
            p.drawLine(a, b)
            mid = (a + b) / 2; d = b - a; L = (d.x()**2 + d.y()**2) ** 0.5; u = d / L
            nrm = QtCore.QPointF(-u.y(), u.x())
            p.setBrush(QtGui.QColor("black"))
            p.drawPolygon(QtGui.QPolygonF([mid + u * 7, mid - u * 5 + nrm * 5, mid - u * 5 - nrm * 5]))
            p.drawText(pos, text)

        a1, a2 = QtCore.QPointF(w * 0.08, h * 0.12), QtCore.QPointF(w * 0.08, h * 0.88)
        b1, b2 = QtCore.QPointF(w * 0.92, h * 0.12), QtCore.QPointF(w * 0.92, h * 0.88)
        fermion(a1, v1, quark, a1 + QtCore.QPointF(-2, -6))
        fermion(v1, a2, quark + "̄", a2 + QtCore.QPointF(-2, 18))
        fermion(b1, v2, "μ⁺", b1 + QtCore.QPointF(-10, -6))
        fermion(v2, b2, "μ⁻", b2 + QtCore.QPointF(-10, 18))
        path = QtGui.QPainterPath(v1)
        for i in range(1, 33):
            path.lineTo(v1.x() + (v2.x() - v1.x()) * i / 32, v1.y() + 7 * np.sin(np.pi * i / 2))
        p.setBrush(QtCore.Qt.NoBrush); p.drawPath(path)
        p.drawText(QtCore.QPointF((v1.x() + v2.x()) / 2 - 10, v1.y() - 14), "Z/γ*" if self.name == "Z" else "γ*")
        f.setPointSize(9); p.setFont(f); p.setPen(QtGui.QColor("#555"))
        note = {"J/psi": "cc̄ bound state annihilates through a virtual photon",
                "Upsilon(1S)": "bb̄ bound state annihilates through a virtual photon",
                "Z": "Drell–Yan: qq̄ → Z/γ* → μ⁺μ⁻"}[self.name]
        p.drawText(QtCore.QRectF(0, h + 4, w, 20), QtCore.Qt.AlignCenter, note)


def event_info_lines(ev) -> list:
    lines = [f"{label(ev.process)} → μ⁺μ⁻   true m = {ev.true_mass:.4f} GeV",
             f"reconstructed m = {ev.reco_mass:.4f} GeV", f"vertex z = {100 * ev.vertex[2]:.1f} cm"]
    for t in ev.tracks:
        lines.append(f"{t.label}: q_reco = {t.charge_reco:+d}, pT = {t.pt_reco:.3f} GeV (hit-error σ {t.sigma_pt:.2g}) "
                     f"true {t.pt_true:.3f}, η = {t.eta_reco:+.2f}, hits = {t.n_hits}, "
                     f"R = {t.pt_reco / (C_TRACK * ev.B):.2f} m")
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
        self.givens.setHorizontalHeaderLabels(["muon", "charge", "chord L [m]", "sagitta s [mm]", "φ [rad]", "η"])
        self.givens.verticalHeader().setVisible(False)
        self.givens.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.givens.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.givens.setMaximumHeight(90)
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
        self.add_btn = QtWidgets.QPushButton("Add my mass to my measurements"); self.add_btn.setEnabled(False)
        nxt = QtWidgets.QPushButton("Next event")
        self.feedback = QtWidgets.QTextEdit(); self.feedback.setReadOnly(True)
        self.hist = pg.PlotWidget(title="My reconstructed masses", labels={"bottom": "m(μμ) [GeV]", "left": "entries"})
        buttons = QtWidgets.QHBoxLayout()
        for w in (check_btn, solve_btn, self.add_btn, nxt):
            buttons.addWidget(w)
        buttons.addStretch()
        left = QtWidgets.QVBoxLayout()
        for w in (intro, self.givens, self.consts):
            left.addWidget(w)
        left.addLayout(grid); left.addLayout(buttons); left.addWidget(self.feedback, 1)
        lay = QtWidgets.QHBoxLayout(self); lay.addLayout(left, 3); lay.addWidget(self.hist, 2)
        check_btn.clicked.connect(self.check)
        solve_btn.clicked.connect(self.show_solution)
        self.add_btn.clicked.connect(self.add_mass)
        nxt.clicked.connect(lambda: self.main.new_event())

    def show_event(self, ev):
        if ev is None:
            return
        self.ex = from_display_event(ev)
        for r, t in enumerate(self.ex.tracks):
            vals = [t.label, f"{t.charge:+d}", f"{t.chord:.4f}", f"{1000 * t.sagitta:.4f}", f"{t.phi:+.4f}", f"{t.eta:+.4f}"]
            for c, v in enumerate(vals):
                self.givens.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.consts.setText(f"B = {self.ex.B:g} T,   m_μ = {self.ex.m_mu:.5f} GeV,   process: {label(ev.process)}")
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
            colour = "#080" if fb.ok else "#b00"
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
        self.hist.setTitle(f"My reconstructed masses: {len(m)} events, mean {m.mean():.4f} GeV")


class EventTab(QtWidgets.QWidget):
    """2D projections (x–y and z–r) of the shared display event."""
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.shown_event = None
        nxt = QtWidgets.QPushButton("Next event")
        self.show_true = QtWidgets.QCheckBox("show true trajectory"); self.show_true.setChecked(True)
        self.xy = pg.PlotWidget(title="Transverse view (B along z, out of page)", labels={"bottom": "x [m]", "left": "y [m]"})
        self.rz = pg.PlotWidget(title="Longitudinal view", labels={"bottom": "z [m]", "left": "±r [m]"})
        self.xy.setAspectLocked(True)
        self.feyn = FeynmanWidget()
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True)
        self.info.setStyleSheet("font-family: Consolas, monospace;")
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top = QtWidgets.QHBoxLayout()
        for w in (nxt, self.show_true):
            top.addWidget(w)
        top.addWidget(QtWidgets.QLabel("  black dots: recorded hits · grey crosses: crossed but not recorded · "
                                       "solid: reconstructed track · dashed: true trajectory"))
        top.addStretch()
        plots = QtWidgets.QHBoxLayout(); plots.addWidget(self.xy, 2); plots.addWidget(self.rz, 3)
        side = QtWidgets.QVBoxLayout(); side.addWidget(QtWidgets.QLabel("<b>Lowest-order diagram</b>"))
        self.contrast = QtWidgets.QLabel(DIAGRAM_VS_TRACKS); self.contrast.setWordWrap(True)
        self.contrast.setStyleSheet("background:#fff8e1; border:1px solid #e0c060; padding:6px; font-size:9pt;")
        side.addWidget(self.feyn); side.addWidget(self.contrast); side.addWidget(self.info); side.addStretch()
        body = QtWidgets.QHBoxLayout(); body.addLayout(plots, 4); body.addLayout(side, 1)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(top); lay.addLayout(body, 1)
        nxt.clicked.connect(lambda: self.main.new_event())
        self.show_true.toggled.connect(lambda _: self.show_event(self.shown_event))

    def draw_detector(self, geom):
        t = np.linspace(0, 2 * np.pi, 240)
        for c in geom.cylinders:
            if c.group == "beam pipe":
                continue
            self.xy.plot(c.radius * np.cos(t), c.radius * np.sin(t), pen=pg.mkPen("#cccccc"))
            for s in (1, -1):
                self.rz.plot([-c.half_length, c.half_length], [s * c.radius] * 2, pen=pg.mkPen("#bbbbbb", width=1.5))
        for d in geom.disks:
            for sr in (1, -1):
                self.rz.plot([d.z] * 2, [sr * d.r_min, sr * d.r_max], pen=pg.mkPen("#bbbbbb", width=1.5))
        self.xy.addItem(pg.TextItem("⊙ B", color="#888", anchor=(0, 0)))

    def show_event(self, ev):
        if ev is None:
            return
        self.shown_event = ev
        self.feyn.set_process(ev.process)
        self.xy.clear(); self.rz.clear()
        self.draw_detector(self.main.geometry)
        for t in ev.tracks:
            col = TRACK_COLORS[t.label]
            side = np.sign(t.true_path[-1, 1]) or 1.0
            r = lambda a: side * np.hypot(a[:, 0], a[:, 1])
            if self.show_true.isChecked():
                self.xy.plot(t.true_path[:, 0], t.true_path[:, 1], pen=pg.mkPen(col, width=1, style=QtCore.Qt.DashLine))
                self.rz.plot(t.true_path[:, 2], r(t.true_path), pen=pg.mkPen(col, width=1, style=QtCore.Qt.DashLine))
            self.xy.plot(t.reco_path[:, 0], t.reco_path[:, 1], pen=pg.mkPen(col, width=2))
            self.rz.plot(t.reco_path[:, 2], r(t.reco_path), pen=pg.mkPen(col, width=2))
            self.xy.plot(t.hits[:, 0], t.hits[:, 1], pen=None, symbol="o", symbolSize=5, symbolBrush="k")
            self.rz.plot(t.hits[:, 2], r(t.hits), pen=None, symbol="o", symbolSize=5, symbolBrush="k")
            if len(t.missed):
                self.xy.plot(t.missed[:, 0], t.missed[:, 1], pen=None, symbol="x", symbolSize=8, symbolPen="#999")
                self.rz.plot(t.missed[:, 2], r(t.missed), pen=None, symbol="x", symbolSize=8, symbolPen="#999")
            o, d, L = momentum_arrow(t.p_reco, ev.vertex)
            dt = d[:2] / max(np.linalg.norm(d[:2]), 1e-12)
            self.xy.plot([o[0], o[0] + L * dt[0]], [o[1], o[1] + L * dt[1]], pen=pg.mkPen(col, width=4))
        self.xy.setRange(xRange=(-1.25, 1.25), yRange=(-1.25, 1.25))
        self.rz.setRange(xRange=(-3, 3), yRange=(-1.25, 1.25))
        self.info.setText("\n".join(event_info_lines(ev)))


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
        self.eff_plot = pg.PlotWidget(labels={"bottom": "pT [GeV]", "left": "track reconstruction efficiency"})
        self.eff_plot.setLogMode(True, False)
        self.cutflow = QtWidgets.QTableWidget(0, 3)
        self.cutflow.setHorizontalHeaderLabels(["step", "events", "fraction"])
        self.cutflow.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.cutflow.verticalHeader().setVisible(False)
        top = QtWidgets.QHBoxLayout()
        for w in (self.eta, run):
            top.addWidget(w)
        top.addWidget(QtWidgets.QLabel("  uses the detector settings on the left"))
        top.addStretch()
        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.plot, 0, 0, 2, 1)
        grid.addWidget(self.eff_plot, 0, 1)
        box = QtWidgets.QGroupBox("Selection cut flow of the last simulation"); QtWidgets.QVBoxLayout(box).addWidget(self.cutflow)
        grid.addWidget(box, 1, 1)
        grid.setColumnStretch(0, 3); grid.setColumnStretch(1, 2)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(top); lay.addLayout(grid, 1)
        run.clicked.connect(self.run)

    def run(self):
        self.main.tasks.start(resolution_scan, self.show, self.main.setup.detector(), self.eta.value(),
                              text="Measuring resolution…")

    def show(self, r):
        pts, full, noms, det = r
        self.plot.clear(); self.eff_plot.clear()
        self.plot.plot(pts, 100 * full[:, 1], pen=pg.mkPen("#1f77b4", width=2), symbol="o", name="simulation (central 68%)")
        self.plot.plot(pts, 100 * noms[:, 1], pen=pg.mkPen("#1f77b4", style=QtCore.Qt.DashLine), symbol="s",
                       name="without multiple scattering")
        if all(np.isinf(b.half_length) for b in det.barrels):
            xx = np.logspace(np.log10(2), np.log10(250), 60)
            self.plot.plot(xx, 100 * gluckstern(xx, det), pen=pg.mkPen("#ff7f0e", width=2), name="Gluckstern formula")
        self.eff_plot.plot(pts, full[:, 2], pen=pg.mkPen("k"), symbol="o")
        self.eff_plot.setYRange(0, 1.05)

    def show_cutflow(self, sim):
        items = list(sim.cutflow.items())
        self.cutflow.setRowCount(len(items))
        for i, (k, v) in enumerate(items):
            for j, val in enumerate((k, str(v), f"{v / max(items[0][1], 1):.2%}")):
                self.cutflow.setItem(i, j, QtWidgets.QTableWidgetItem(val))


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
        src = QtWidgets.QLabel(f"Source: {DB.source}; retrieved {DB.retrieved}.")
        src.setWordWrap(True)
        # conservation checker
        self.edit = QtWidgets.QLineEdit("Z -> mu+ mu-")
        self.edit.setMinimumWidth(220)
        self.examples = QtWidgets.QComboBox(); self.examples.addItems(EXAMPLES)
        btn = QtWidgets.QPushButton("Check")
        self.use = QtWidgets.QPushButton("Simulate this process"); self.use.setEnabled(False)
        self.out = QtWidgets.QTextEdit(); self.out.setReadOnly(True)
        form = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Decay:"), self.edit, btn, QtWidgets.QLabel("Examples:"), self.examples):
            form.addWidget(w)
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
        dl.addRow("cos θ* of μ⁻ (rest frame, from z)", self.cos); dl.addRow(self.demo)
        left = QtWidgets.QVBoxLayout(); left.addWidget(QtWidgets.QLabel("<b>Particle properties</b>"))
        left.addWidget(self.table, 1); left.addWidget(src); left.addWidget(dbox)
        right = QtWidgets.QVBoxLayout(); right.addWidget(QtWidgets.QLabel("<b>Conservation-law checker</b>"))
        right.addLayout(form); right.addWidget(self.use); right.addWidget(self.out, 1)
        lay = QtWidgets.QHBoxLayout(self); lay.addLayout(left, 3); lay.addLayout(right, 2)
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
            self.out.setHtml(f"<p style='color:#b00'>{e}</p>")
            return
        colour = "#080" if v.allowed else "#b00"
        html = [f"<h3 style='color:{colour}'>{v.category}</h3><table cellpadding=3>",
                "<tr><th align=left>law</th><th>kind</th><th>before</th><th>after</th><th>result</th></tr>"]
        for c in v.checks:
            fmt = (lambda x: f"{x:.4g}") if c.kind == "kinematic" else (lambda x: f"{x:+d}")
            res = "ok" if c.passed else "<b style='color:#b00'>fails</b>"
            html.append(f"<tr><td>{c.law}</td><td>{c.kind}</td><td align=center>{fmt(c.before)}</td>"
                        f"<td align=center>{fmt(c.after)}</td><td>{res}</td></tr>")
        html.append("</table><p style='color:#555'>exact: never violated · accidental: conserved in the "
                    "Standard Model, never seen violated · approximate: lepton flavour (violated by neutrino "
                    "oscillations) · kinematic: energy–momentum. Passing all checks is necessary, not "
                    "sufficient: spin, parity, C and dynamics are not checked.</p>")
        g = self._gate
        html.append(f"<p><b>Simulation:</b> <span style='color:{'#080' if g.simulable else '#b00'}'>{g.message}</span></p>")
        self.out.setHtml("".join(html))

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
        f = lambda v: f"({v[0]:8.3f}, {v[1]:8.3f}, {v[2]:8.3f}, {v[3]:8.3f})"
        self.demo.setText(f"rest frame  μ⁻ {f(mum[0])}\n            μ⁺ {f(mup[0])}\n"
                          f"lab (β={b:.2f}) μ⁻ {f(lm)}\n            μ⁺ {f(lp)}\n"
                          f"Σ lab = {f(lm + lp)}   invariant mass = {mm:.6f} GeV (input {m:.6f})\n"
                          f"opening angle in lab = {np.degrees(np.arccos(np.dot(lm[1:], lp[1:]) / np.linalg.norm(lm[1:]) / np.linalg.norm(lp[1:]))):.1f}°   (E, px, py, pz) in GeV")


# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle Physics Lab: dimuon resonances")
        self.tasks = TaskRunner(self)
        self.event_tasks = TaskRunner(self, report=False)
        self.last_sim = None
        self.current_event = None
        self.geometry = None
        self._event_listeners = []
        self._data = None
        self.setup = SetupPanel()
        self.mass = MassTab(self)
        self.event2d = EventTab(self)
        self.performance = PerformanceTab(self)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.mass, "Mass spectrum")
        tabs.addTab(self.event2d, "Event display (2D)")
        from .view3d import Event3DTab
        self.event3d = Event3DTab(self)
        tabs.addTab(self.event3d, "Event display (3D)")
        self.add_event_listener(self.event2d.show_event)
        self.add_event_listener(self.event3d.show_event)
        self.reconstruct = ReconstructTab(self)
        tabs.addTab(self.reconstruct, "Reconstruct it yourself")
        self.add_event_listener(self.reconstruct.show_event)
        tabs.addTab(self.performance, "Detector performance")
        tabs.addTab(ExplorerTab(self), "Physics explorer")
        self.tabs = tabs
        split = QtWidgets.QSplitter()
        scroll = QtWidgets.QScrollArea(); scroll.setWidget(self.setup); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        split.addWidget(scroll); split.addWidget(tabs); split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.progress = QtWidgets.QProgressBar(); self.progress.setMaximumWidth(220); self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.set_status("Ready", busy=False)
        self.resize(1450, 860)
        self.new_event()

    # ---- shared display event ------------------------------------------------
    def add_event_listener(self, fn):
        self._event_listeners.append(fn)

    def new_event(self, seed=None):
        """Generate one display event with the current settings in a worker thread."""
        det = self.setup.detector()
        name = self.setup.proc.currentData()
        rng = np.random.default_rng(seed)

        def job(progress=None):
            return make_event(name, det, rng), det
        return self.event_tasks.start(job, self._event_ready)

    def _event_ready(self, result):
        ev, det = result
        self.geometry = detector_geometry(det)
        self.current_event = ev
        for fn in self._event_listeners:
            fn(ev)

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

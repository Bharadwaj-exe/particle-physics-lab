"""3D event display (PyVista / VTK) for the desktop application.

Rendering only: the scene is built from ``visualization.event_display`` data, which
comes from the unchanged physics and detector code.

PyVista takes ~10 s to import on the reference laptop, so it is imported in a
background thread the first time the tab is shown; the window stays responsive.
VTK objects themselves are created on the GUI thread, as Qt requires.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

os.environ.setdefault("QT_API", "pyside6")     # make pyvistaqt/QtPy use the same binding as the app

from PySide6 import QtCore, QtWidgets  # noqa: E402

from ..visualization.event_display import DetectorGeometry, DisplayEvent, momentum_arrow  # noqa: E402

DETECTOR_GROUPS = ("beam pipe", "pixel barrel", "strip barrel", "pixel disks", "inner disks", "endcap disks", "beam axis")
EVENT_GROUPS = ("true tracks", "reconstructed tracks", "hits", "missed hits", "vertex", "momentum vectors")
STYLE = {
    "beam pipe": dict(color="#90a4ae", opacity=0.35),
    "pixel barrel": dict(color="#4f9bd9", opacity=0.18),
    "strip barrel": dict(color="#9e9e9e", opacity=0.06),
    "pixel disks": dict(color="#4f9bd9", opacity=0.15),
    "inner disks": dict(color="#c8a165", opacity=0.07),
    "endcap disks": dict(color="#c8a165", opacity=0.05),
}
RECO_COLOR = {"μ⁻": "#1f77b4", "μ⁺": "#d62728"}
TRUE_COLOR = {"μ⁻": "#9cc3e6", "μ⁺": "#f0a5a6"}
DEFAULT_OFF = {"missed hits": False}


class SceneBuilder:
    """Builds and updates PyVista actors on any pyvista Plotter (Qt or off-screen)."""

    def __init__(self, plotter, pv):
        self.pl, self.pv = plotter, pv
        self.actors = {g: [] for g in DETECTOR_GROUPS + EVENT_GROUPS}
        self.visible = {g: DEFAULT_OFF.get(g, True) for g in DETECTOR_GROUPS + EVENT_GROUPS}
        self.geometry_key = None
        self.residual_scale = 1.0
        self.event = None
        self.last_update_s = 0.0
        self._text = None
        self.pl.set_background("white")
        self.pl.add_axes(line_width=2, labels_off=False)

    # ---- detector ------------------------------------------------------------
    def set_geometry(self, geom: DetectorGeometry):
        if geom.key == self.geometry_key:
            return
        for g in DETECTOR_GROUPS:
            self._clear(g)
        pv = self.pv
        by_group = {}
        for c in geom.cylinders:
            mesh = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=c.radius, height=2 * c.half_length,
                               resolution=72, capping=False)
            by_group.setdefault(c.group, []).append(mesh)
        for d in geom.disks:
            by_group.setdefault(d.group, []).append(pv.Disc(center=(0, 0, d.z), inner=d.r_min, outer=d.r_max,
                                                            normal=(0, 0, 1), r_res=1, c_res=72))
        for group, meshes in by_group.items():
            merged = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
            self._add(group, merged, smooth_shading=False, show_edges=False, **STYLE[group])
        (a, b) = geom.beam_axis
        self._add("beam axis", pv.Line(a, b), color="black", line_width=1.5)
        self.geometry_key = geom.key

    # ---- event ---------------------------------------------------------------
    def set_event(self, ev: DisplayEvent, residual_scale: float | None = None):
        t0 = time.perf_counter()
        if residual_scale is not None:
            self.residual_scale = residual_scale
        for g in EVENT_GROUPS:
            self._clear(g)
        pv = self.pv
        for t in ev.tracks:
            self._add("true tracks", pv.lines_from_points(t.true_path), color=TRUE_COLOR[t.label], line_width=3)
            self._add("reconstructed tracks", pv.lines_from_points(t.reco_path).tube(radius=0.008, n_sides=12),
                      color=RECO_COLOR[t.label])
            hits = t.hits_exaggerated(self.residual_scale)
            if len(hits):
                self._add("hits", pv.PolyData(hits), color="#111111", point_size=9, render_points_as_spheres=True)
                if self.residual_scale > 1:
                    seg = np.empty((2 * len(hits), 3)); seg[0::2] = t.hits_ideal; seg[1::2] = hits
                    self._add("hits", pv.line_segments_from_points(seg), color="#555555", line_width=1)
            if len(t.missed):
                self._add("missed hits", pv.PolyData(t.missed), color="#aaaaaa", point_size=8,
                          render_points_as_spheres=False)
            o, d, L = momentum_arrow(t.p_reco, ev.vertex)
            self._add("momentum vectors", pv.Arrow(start=o, direction=d, scale=L, tip_length=0.25,
                                                   tip_radius=0.08, shaft_radius=0.025), color=RECO_COLOR[t.label])
        self._add("vertex", pv.Sphere(radius=0.012, center=ev.vertex), color="#2ca02c")
        if self._text is not None:
            self.pl.remove_actor(self._text)
        self._text = self.pl.add_text(self._info(ev), position="upper_left", font_size=9, color="black")
        self.event = ev
        self.pl.render()
        self.last_update_s = time.perf_counter() - t0

    def _info(self, ev):
        lines = [f"{ev.process} -> mu+ mu-   true m = {ev.true_mass:.4f} GeV, reco m = {ev.reco_mass:.4f} GeV",
                 f"vertex z = {100 * ev.vertex[2]:.1f} cm,  B = {ev.B} T along +z"]
        for t in ev.tracks:
            lab = "mu-" if t.label == "μ⁻" else "mu+"
            lines.append(f"{lab}: pT reco {t.pt_reco:.2f} / true {t.pt_true:.2f} GeV, eta {t.eta_reco:+.2f}, {t.n_hits} hits")
        if self.residual_scale > 1:
            lines.append(f"hit residuals magnified x{self.residual_scale:g} (display only)")
        return "\n".join(lines)

    # ---- visibility and camera -----------------------------------------------
    def set_visible(self, group: str, on: bool):
        self.visible[group] = on
        for a in self.actors[group]:
            a.SetVisibility(bool(on))
        self.pl.render()

    def camera(self, view: str):
        if view == "beam":
            self.pl.view_xy()
        elif view == "side":
            self.pl.view_zy()
        else:
            # oblique view with the beam horizontal, as in collider event displays
            self.pl.camera_position = [(4.2, 2.6, 3.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.pl.reset_camera()
        self.pl.render()

    def n_actors(self, group=None):
        return sum(len(v) for k, v in self.actors.items() if group in (None, k))

    # ---- helpers ---------------------------------------------------------------
    def _add(self, group, mesh, **kw):
        actor = self.pl.add_mesh(mesh, reset_camera=False, pickable=False, **kw)
        actor.SetVisibility(bool(self.visible[group]))
        self.actors[group].append(actor)

    def _clear(self, group):
        for a in self.actors[group]:
            self.pl.remove_actor(a, render=False)
        self.actors[group] = []


class _Loader(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    thread_ident = None     # recorded so tests can confirm the import runs off the GUI thread

    @QtCore.Slot()
    def run(self):
        self.thread_ident = threading.get_ident()
        try:
            t0 = time.perf_counter()
            import pyvista
            import pyvistaqt
            self.done.emit((pyvista, pyvistaqt, time.perf_counter() - t0))
        except Exception as e:     # missing package or broken OpenGL
            self.failed.emit(f"{type(e).__name__}: {e}")


class Event3DTab(QtWidgets.QWidget):
    ready = QtCore.Signal()

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.builder = None
        self.plotter = None
        self.pending = None
        self.load_seconds = None
        self._thread = None
        self.status = QtWidgets.QLabel("The 3D renderer (PyVista) loads when this tab is first opened "
                                       "(about 10 s on this laptop; the window stays usable).")
        self.status.setWordWrap(True)
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.view_box = QtWidgets.QVBoxLayout()
        self.view_box.addWidget(self.status, 1)
        self.controls = self._build_controls()
        self.controls.setEnabled(False)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addLayout(self.view_box, 4)
        lay.addWidget(self.controls, 1)

    # ---- UI --------------------------------------------------------------------
    def _build_controls(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        nxt = QtWidgets.QPushButton("Next event")
        nxt.clicked.connect(lambda: self.main.new_event())
        v.addWidget(nxt)
        cam = QtWidgets.QGroupBox("Camera")
        cl = QtWidgets.QGridLayout(cam)
        for i, (text, view) in enumerate([("3D (reset)", "iso"), ("Along beam (x–y)", "beam"), ("Side (z–y)", "side")]):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, vw=view: self.builder and self.builder.camera(vw))
            cl.addWidget(b, i, 0)
        cl.addWidget(QtWidgets.QLabel("mouse: left-drag rotate · wheel zoom · shift-drag pan"), 3, 0)
        v.addWidget(cam)
        self.checks = {}
        for title, groups in (("Detector", DETECTOR_GROUPS), ("Event", EVENT_GROUPS)):
            g = QtWidgets.QGroupBox(title)
            gl = QtWidgets.QVBoxLayout(g)
            for name in groups:
                c = QtWidgets.QCheckBox(name)
                c.setChecked(DEFAULT_OFF.get(name, True))
                c.toggled.connect(lambda on, n=name: self.builder and self.builder.set_visible(n, on))
                gl.addWidget(c)
                self.checks[name] = c
            v.addWidget(g)
        res = QtWidgets.QGroupBox("Hit residuals (display only)")
        rl = QtWidgets.QVBoxLayout(res)
        self.residual = QtWidgets.QComboBox()
        for f in (1, 10, 100, 1000):
            self.residual.addItem("true scale" if f == 1 else f"magnify ×{f}", f)
        self.residual.currentIndexChanged.connect(self._residual_changed)
        rl.addWidget(self.residual)
        rl.addWidget(QtWidgets.QLabel("Smearing and multiple scattering move hits by µm–mm, invisible at metre scale. "
                                      "Magnifying draws each hit away from the true crossing by the chosen factor."))
        for w in rl.parent().findChildren(QtWidgets.QLabel):
            w.setWordWrap(True)
        v.addWidget(res)
        legend = QtWidgets.QLabel(
            "<b>Legend</b><br>thick tube: reconstructed track (blue μ⁻, red μ⁺)<br>thin light line: true trajectory<br>"
            "black spheres: recorded hits · grey squares: lost hits<br>green sphere: decay vertex · arrows: reconstructed "
            "momentum (length ∝ log p)<br>black line: beam axis (z)")
        legend.setWordWrap(True)
        v.addWidget(legend)
        v.addStretch()
        return box

    # ---- lazy loading ------------------------------------------------------------
    def showEvent(self, e):
        super().showEvent(e)
        self.start_loading()

    def start_loading(self):
        if self.builder is not None or self._thread is not None:
            return
        self.status.setText("Loading the 3D renderer (PyVista/VTK)…")
        self._thread = QtCore.QThread()
        self._loader = _Loader()
        self._loader.moveToThread(self._thread)
        self._thread.started.connect(self._loader.run)
        self._loader.done.connect(self._loaded)
        self._loader.failed.connect(self._load_failed)
        self._thread.start()

    def _finish_thread(self):
        self._thread.quit(); self._thread.wait()

    def _load_failed(self, msg):
        self._finish_thread()
        self.status.setText(f"3D view unavailable: {msg}\nInstall with: pip install pyvista pyvistaqt")

    def _loaded(self, mods):
        self._finish_thread()
        pv, pvqt, secs = mods
        self.load_seconds = secs
        self.plotter = pvqt.QtInteractor(self)
        self.view_box.removeWidget(self.status)
        self.status.hide()
        self.view_box.addWidget(self.plotter.interactor, 1)
        self.builder = SceneBuilder(self.plotter, pv)
        self.controls.setEnabled(True)
        ev = self.pending or self.main.current_event
        if ev is not None:
            self._draw(ev)
            self.builder.camera("iso")
        self.ready.emit()

    # ---- updates -----------------------------------------------------------------
    def show_event(self, ev):
        if self.builder is None:
            self.pending = ev
            return
        self._draw(ev)

    def _draw(self, ev):
        if self.main.geometry is not None:
            self.builder.set_geometry(self.main.geometry)
        self.builder.set_event(ev, self.residual.currentData())

    def _residual_changed(self, _):
        if self.builder is not None and self.builder.event is not None:
            self.builder.set_event(self.builder.event, self.residual.currentData())

    def closeEvent(self, e):
        if self.plotter is not None:
            self.plotter.close()
        super().closeEvent(e)

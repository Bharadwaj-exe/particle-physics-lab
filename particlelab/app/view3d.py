"""3D event display (PyVista / VTK) for the desktop application.

Rendering only: the scene is built from ``visualization.event_display`` data, which
comes from the unchanged physics and detector code.

PyVista takes ~10 s to import on the reference laptop, so it is imported in a
background thread the first time the tab is shown; the window stays responsive.
VTK objects themselves are created on the GUI thread, as Qt requires.

Animation: the shared clock gives a fraction of the event's flight time. While the
muons fly, the path flown so far and each muon's position are drawn and hits appear
when their surface is crossed; the fitted tracks and momentum arrows appear at the
end. Frames update existing VTK data in place instead of rebuilding actors.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

os.environ.setdefault("QT_API", "pyside6")     # make pyvistaqt/QtPy use the same binding as the app

from PySide6 import QtCore, QtWidgets  # noqa: E402

from ..visualization.event_display import DetectorGeometry, DisplayEvent, momentum_arrow, path_until  # noqa: E402
from . import theme  # noqa: E402
from .animation import AnimationBar, AnimationClock  # noqa: E402

DETECTOR_GROUPS = ("beam pipe", "pixel barrel", "strip barrel", "pixel disks", "inner disks", "endcap disks", "beam axis")
EVENT_GROUPS = ("true tracks", "reconstructed tracks", "hits", "missed hits", "layer crossings", "vertex",
                "momentum vectors")
FLIGHT = "muons in flight"          # animation only: the path flown so far and each muon's position
STYLE = {
    "beam pipe": dict(color="#90a4ae", opacity=0.35),
    "pixel barrel": dict(color="#4f9bd9", opacity=0.18),
    "strip barrel": dict(color="#9e9e9e", opacity=0.06),
    "pixel disks": dict(color="#4f9bd9", opacity=0.15),
    "inner disks": dict(color="#c8a165", opacity=0.07),
    "endcap disks": dict(color="#c8a165", opacity=0.05),
}
RECO_COLOR = theme.THEMES["light"]["reco"]
TRUE_COLOR = theme.THEMES["light"]["true"]
DEFAULT_OFF = {"missed hits": False}
FINAL_ONLY = ("true tracks", "reconstructed tracks", "momentum vectors")   # hidden while the muons fly


class SceneBuilder:
    """Builds and updates PyVista actors on any pyvista Plotter (Qt or off-screen)."""

    def __init__(self, plotter, pv, palette=None):
        self.pl, self.pv = plotter, pv
        self.t = palette or theme.current()
        groups = DETECTOR_GROUPS + EVENT_GROUPS + (FLIGHT,)
        self.actors = {g: [] for g in groups}
        self.visible = {g: DEFAULT_OFF.get(g, True) for g in groups}
        self.geometry_key = None
        self.geometry = None
        self.residual_scale = 1.0
        self.event = None
        self.progress = 1.0
        self.span = None              # ns mapped to progress 1 (None: this event's own flight time)
        self.last_update_s = 0.0
        self._text = None
        self._phase_off = set()       # ids of actors hidden by the animation phase
        self._anim = []               # (actor, kind, full points, flight times)
        self._heads = []              # (actor, path, flight times)
        self._apply_background()

    # ---- theme ----------------------------------------------------------------
    def _apply_background(self):
        self.pl.set_background(self.t["bg"])
        try:
            self.pl.hide_axes()
        except Exception:      # no axes yet
            pass
        self.pl.add_axes(line_width=2, labels_off=False, color=self.t["fg"])

    def set_theme(self, palette):
        self.t = palette
        self._apply_background()
        geom, ev = self.geometry, self.event
        self.geometry_key = None
        if geom is not None:
            self.set_geometry(geom)
        if ev is not None:
            self.set_event(ev)
        else:
            self.pl.render()

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
        dark = self.t["name"] == "dark"
        for group, meshes in by_group.items():
            merged = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
            st = dict(STYLE[group])
            if dark:                  # thin shells need a little more opacity on a dark background
                st["opacity"] = min(1.0, st["opacity"] * 1.6)
            self._add(group, merged, smooth_shading=False, show_edges=False, **st)
        (a, b) = geom.beam_axis
        self._add("beam axis", pv.Line(a, b), color=self.t["fg"], line_width=1.5)
        self.geometry_key = geom.key
        self.geometry = geom

    # ---- event ---------------------------------------------------------------
    def set_event(self, ev: DisplayEvent, residual_scale: float | None = None):
        t0 = time.perf_counter()
        if residual_scale is not None:
            self.residual_scale = residual_scale
        for g in EVENT_GROUPS + (FLIGHT,):
            self._clear(g)
        self._anim, self._heads, self._phase_off = [], [], set()
        pv, th = self.pv, self.t
        for t in ev.tracks:
            reco_c, true_c = th["reco"][t.label], th["true"][t.label]
            if t.has_truth:
                self._add("true tracks", pv.lines_from_points(t.true_path), color=true_c, line_width=3)
            self._add("reconstructed tracks", pv.lines_from_points(t.reco_path).tube(radius=0.008, n_sides=12),
                      color=reco_c)
            if len(t.hits):
                hits = t.hits_exaggerated(self.residual_scale)
                a = self._add("hits", pv.PolyData(hits), color=th["hit"], point_size=9, render_points_as_spheres=True)
                self._anim.append((a, "points", hits, t.hit_t))
                if self.residual_scale > 1:
                    seg = np.empty((2 * len(hits), 3)); seg[0::2] = t.hits_ideal; seg[1::2] = hits
                    a = self._add("hits", pv.line_segments_from_points(seg), color=th["muted"], line_width=1)
                    self._anim.append((a, "segments", seg, t.hit_t))
            if len(t.missed):
                a = self._add("missed hits", pv.PolyData(t.missed), color=th["missed"], point_size=8,
                              render_points_as_spheres=False)
                self._anim.append((a, "points", t.missed, t.missed_t))
            if len(t.crossings):
                a = self._add("layer crossings", pv.PolyData(t.crossings), color=th["crossing"], point_size=8,
                              render_points_as_spheres=True)
                self._anim.append((a, "points", t.crossings, t.crossing_t))
            o, d, L = momentum_arrow(t.p_reco, ev.vertex)
            self._add("momentum vectors", pv.Arrow(start=o, direction=d, scale=L, tip_length=0.25,
                                                   tip_radius=0.08, shaft_radius=0.025), color=reco_c)
            path, times = t.flight_path
            a = self._add(FLIGHT, pv.lines_from_points(path), color=reco_c, line_width=4)
            self._anim.append((a, "line", path, times))
            head = self._add(FLIGHT, pv.Sphere(radius=0.04, center=(0, 0, 0)), color=reco_c)
            self._heads.append((head, path, times))
        self._add("vertex", pv.Sphere(radius=0.012, center=ev.vertex), color=th["vertex"])
        if self._text is not None:
            self.pl.remove_actor(self._text)
        self._text = self.pl.add_text(self._info(ev), position="upper_left", font_size=9, color=th["fg"])
        self.event = ev
        self._apply_progress()
        self.pl.render()
        self.last_update_s = time.perf_counter() - t0

    def _info(self, ev):
        if ev.is_data:
            lines = [f"CMS 2011 data, run {ev.run}, event {ev.event_number}:  m(mu mu) = {ev.reco_mass:.4f} GeV",
                     f"candidate in the {ev.process} mass window (may be background); muon types "
                     f"{'/'.join(ev.muon_types)} (G global, T tracker)",
                     f"helices from the measured momenta in B = {ev.B} T; vertex drawn at the origin",
                     "layer crossings are computed from the momenta, not CMS hits"]
        else:
            lines = [f"{ev.process} -> mu+ mu-   true m = {ev.true_mass:.4f} GeV, reco m = {ev.reco_mass:.4f} GeV",
                     f"vertex z = {100 * ev.vertex[2]:.1f} cm,  B = {ev.B} T along +z"]
        for t in ev.tracks:
            lab = "mu-" if t.label == "μ⁻" else "mu+"
            if ev.is_data:
                lines.append(f"{lab}: pT {t.pt_reco:.2f} GeV, eta {t.eta_reco:+.2f}")
            else:
                lines.append(f"{lab}: pT reco {t.pt_reco:.2f} / true {t.pt_true:.2f} GeV, eta {t.eta_reco:+.2f}, "
                             f"{t.n_hits} hits")
        if self.residual_scale > 1 and not ev.is_data:
            lines.append(f"hit residuals magnified x{self.residual_scale:g} (display only)")
        return "\n".join(lines)

    # ---- animation -------------------------------------------------------------
    def set_progress(self, fraction: float, render: bool = True):
        """Show the event at ``fraction`` of its flight time (1 = complete, tracks fitted)."""
        self.progress = min(max(float(fraction), 0.0), 1.0)
        if self.event is None:
            return
        self._apply_progress()
        if render:
            self.pl.render()

    def _apply_progress(self):
        final = self.progress >= 1.0
        t_cut = self.progress * (self.span or self.event.t_max)
        off = set()
        for a, kind, pts, times in self._anim:
            ds = a.GetMapper().GetInput()
            if kind == "line":
                part = None if final else path_until(pts, times, t_cut)
                if part is None or len(part) < 2:
                    off.add(id(a)); continue
                ds.ShallowCopy(self.pv.lines_from_points(part))
            else:
                # hits are stored in flight order, so the first n are those already crossed
                n = len(times) if final else int(np.sum(times <= t_cut))
                if n == 0:
                    off.add(id(a)); continue
                if kind == "segments":
                    ds.ShallowCopy(self.pv.line_segments_from_points(pts[:2 * n]))
                else:
                    ds.ShallowCopy(self.pv.PolyData(pts[:n]))
            ds.Modified()
        for head, path, times in self._heads:
            if final:
                off.add(id(head)); continue
            head.SetPosition(*path_until(path, times, t_cut)[-1])
        if not final:
            for g in FINAL_ONLY:
                off.update(id(a) for a in self.actors[g])
        self._phase_off = off
        for g, acts in self.actors.items():
            for a in acts:
                a.SetVisibility(bool(self.visible[g]) and id(a) not in off)

    # ---- visibility and camera -----------------------------------------------
    def set_visible(self, group: str, on: bool):
        self.visible[group] = on
        for a in self.actors[group]:
            a.SetVisibility(bool(on) and id(a) not in self._phase_off)
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

    def screenshot(self, path):
        """Save the current view as a PNG."""
        self.pl.render()
        self.pl.screenshot(str(path))
        return path

    def n_actors(self, group=None):
        return sum(len(v) for k, v in self.actors.items() if group in (None, k))

    # ---- helpers ---------------------------------------------------------------
    def _add(self, group, mesh, **kw):
        actor = self.pl.add_mesh(mesh, reset_camera=False, pickable=False, **kw)
        actor.SetVisibility(bool(self.visible[group]))
        self.actors[group].append(actor)
        return actor

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
        self.clock = getattr(main, "clock", None) or AnimationClock(self)
        self.builder = None
        self.plotter = None
        self.pending = None
        self.load_seconds = None
        self._thread = None
        self.status = QtWidgets.QLabel("The 3D renderer (PyVista) loads when this tab is first opened "
                                       "(about 10 s on this laptop; the window stays usable).")
        self.status.setWordWrap(True)
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.bar = AnimationBar(self.clock)
        self.view_box = QtWidgets.QVBoxLayout()
        self.view_box.addWidget(self.status, 1)
        self.view_box.addWidget(self.bar)
        self.controls = self._build_controls()
        self.controls.setEnabled(False)
        lay = QtWidgets.QHBoxLayout(self)
        lay.addLayout(self.view_box, 4)
        lay.addWidget(self.controls, 1)
        self.clock.frame.connect(self._frame)

    # ---- UI --------------------------------------------------------------------
    def _build_controls(self):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        nxt = QtWidgets.QPushButton("Next event")
        nxt.clicked.connect(lambda: self.main.new_event())
        save = QtWidgets.QPushButton("Save image")
        save.setToolTip("Save the current 3D view as a PNG")
        save.clicked.connect(self.save_image_dialog)
        row.addWidget(nxt); row.addWidget(save)
        v.addLayout(row)
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
            "dark spheres: recorded hits · grey squares: lost hits<br>purple spheres (CMS data): computed layer "
            "crossings, not hits<br>green sphere: vertex · arrows: momentum (length ∝ log p)<br>"
            "line along z: beam axis<br><b>Animation:</b> bright line and ball: the muon in flight at its real "
            "speed; the fitted track appears when both muons have left the tracker.")
        legend.setWordWrap(True)
        v.addWidget(legend)
        v.addStretch()
        return box

    # ---- lazy loading ------------------------------------------------------------
    def showEvent(self, e):
        super().showEvent(e)
        self.start_loading()
        if self.builder is not None and self.builder.progress != self.clock.fraction:
            self.builder.set_progress(self.clock.fraction)

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
        self.view_box.insertWidget(0, self.plotter.interactor, 1)
        self.builder = SceneBuilder(self.plotter, pv)
        self.builder.progress = self.clock.fraction
        self.controls.setEnabled(True)
        ev = self.pending or getattr(self.main, "current_event", None)
        if ev is not None:
            self._draw(ev)
            self.builder.camera("iso")
        self.ready.emit()

    # ---- updates -----------------------------------------------------------------
    def show_event(self, ev):
        if getattr(self.main, "clock", None) is None:     # standalone use: own clock
            self.clock.set_span(ev.t_max)
        if self.builder is None:
            self.pending = ev
            return
        self._draw(ev)

    def _draw(self, ev):
        geom = getattr(self.main, "geometry", None)
        if geom is not None:
            self.builder.set_geometry(geom)
        self.builder.progress = self.clock.fraction
        self.builder.span = self.clock.t_max or None
        self.builder.set_event(ev, self.residual.currentData())

    def _frame(self, f):
        if self.builder is not None and self.isVisible():
            self.builder.span = self.clock.t_max or None
            self.builder.set_progress(f)

    def _residual_changed(self, _):
        if self.builder is not None and self.builder.event is not None:
            self.builder.set_event(self.builder.event, self.residual.currentData())

    def theme_changed(self):
        if self.builder is not None:
            self.builder.set_theme(theme.current())

    def save_image(self, path):
        if self.builder is None:
            return None
        return self.builder.screenshot(path)

    def save_image_dialog(self):
        if self.builder is None:
            return
        ev = self.builder.event
        name = "event3d.png" if ev is None else f"event3d_{'cms' if ev.is_data else 'sim'}_{ev.process}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save 3D view", name.replace("/", ""), "PNG (*.png)")
        if path:
            self.save_image(path)

    def close_renderer(self):
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    def closeEvent(self, e):
        self.close_renderer()
        super().closeEvent(e)

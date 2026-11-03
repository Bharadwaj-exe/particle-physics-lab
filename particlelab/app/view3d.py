"""3D event display (PyVista / VTK) for the desktop application.

Rendering only: the scene is built from ``visualization.event_display`` data, which
comes from the unchanged physics and detector code.

PyVista takes ~10 s to import on the reference laptop, so it is imported in a
background thread the first time the tab is shown; the window stays responsive.
VTK objects themselves are created on the GUI thread, as Qt requires.

Animation: the shared clock gives a fraction of the event's flight time. While the
muons fly, the path flown so far and each muon's position are drawn and hits appear
when their surface is crossed; the fitted tracks and momentum arrows appear when both
muons have left the tracker. With the outer detector shown, the muons then cross the
calorimeters, the coil and the return yoke (drawn cut away), muon-chamber hits light up
as each station is crossed, and a muon that runs out of energy is marked where it stops.
Frames update existing VTK data in place instead of rebuilding actors.
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
                "momentum vectors", "muons outside the tracker", "muon chamber hits")
OUTER_GROUPS = ("calorimeters", "solenoid", "return yoke", "muon chambers")   # schematic, display only
FLIGHT = "muons in flight"          # animation only: the path flown so far and each muon's position
STYLE = {
    "beam pipe": dict(color="#90a4ae", opacity=0.35),
    "pixel barrel": dict(color="#4f9bd9", opacity=0.18),
    "strip barrel": dict(color="#9e9e9e", opacity=0.06),
    "pixel disks": dict(color="#4f9bd9", opacity=0.15),
    "inner disks": dict(color="#c8a165", opacity=0.07),
    "endcap disks": dict(color="#c8a165", opacity=0.05),
}
# outer detector: colours as in the usual CMS drawings (ECAL green, HCAL amber, coil grey, iron red)
# faint faces plus clear outlines, as in collider event displays: stacked translucent
# faces alone turn into a haze in which the muons disappear
OUTER_STYLE = {
    "ECAL": dict(color="#2e9e5b", opacity=0.10, edge=0.9),
    "HCAL": dict(color="#d9a227", opacity=0.07, edge=0.8),
    "solenoid": dict(color="#8e9aa4", opacity=0.10, edge=0.8),
    "yoke": dict(color="#c0392b", opacity=0.025, edge=0.75),       # many layers: outlines carry them
    "chambers": dict(color="#6d7b86", opacity=0.03, edge=0.8),
}
CUT_AWAY = (0.0, 90.0)              # degrees of azimuth removed from the outer detector, facing the camera
WHOLE_VIEW = [(15.5, 9.5, 12.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
TRACKER_VIEW = [(4.2, 2.6, 3.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]


def sector_mesh(pv, r_in, r_out, z0, z1, polygon=False):
    """A ring (r_in..r_out, z0..z1) with the CUT_AWAY wedge removed, as a closed surface.
    ``polygon``: twelve flat sides like the CMS barrel wheels, otherwise round."""
    start = CUT_AWAY[1]
    span = 360.0 - (CUT_AWAY[1] - CUT_AWAY[0])
    if polygon:
        phi = np.radians(np.arange(start, start + span + 1e-6, 30.0))
        scale = 1.0 / np.cos(np.pi / 12)              # flats (not corners) at the given radii
    else:
        phi = np.radians(np.linspace(start, start + span, 55))
        scale = 1.0
    R = np.array([r_in, r_out]) * scale
    Z = np.array([z0, z1])
    rr, pp, zz = np.meshgrid(R, phi, Z, indexing="ij")
    grid = pv.StructuredGrid(rr * np.cos(pp), rr * np.sin(pp), zz)
    return grid.extract_surface(algorithm="dataset_surface")
RECO_COLOR = theme.THEMES["light"]["reco"]
TRUE_COLOR = theme.THEMES["light"]["true"]
DEFAULT_OFF = {"missed hits": False}
FINAL_ONLY = ("true tracks", "reconstructed tracks", "momentum vectors")   # hidden while the muons fly


class SceneBuilder:
    """Builds and updates PyVista actors on any pyvista Plotter (Qt or off-screen)."""

    def __init__(self, plotter, pv, palette=None):
        self.pl, self.pv = plotter, pv
        self.t = palette or theme.current()
        groups = DETECTOR_GROUPS + OUTER_GROUPS + EVENT_GROUPS + (FLIGHT,)
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
        # (no depth peeling: it crashes the on-screen Qt view with the reference laptop's Intel
        # driver; faint faces with outlines read well without it)
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
        for g in DETECTOR_GROUPS + OUTER_GROUPS:
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
        if geom.outer is not None:
            self._add_outer(geom.outer, dark)
        self.geometry_key = geom.key
        self.geometry = geom

    def _add_outer(self, od, dark):
        """Calorimeters, coil, return yoke and muon stations, cut away towards the camera."""
        pv = self.pv
        meshes = {}
        for layer in od.layers:
            style = "ECAL" if layer.name.startswith("ECAL") else "HCAL" if layer.name.startswith("HCAL") else \
                "solenoid" if layer.group == "solenoid" else "yoke"
            polygon = layer.group == "return yoke"
            if layer.kind == "barrel":
                parts = [sector_mesh(pv, layer.r_in, layer.r_out, -layer.z_out, layer.z_out, polygon)]
            else:
                parts = [sector_mesh(pv, layer.r_in, layer.r_out, s * layer.z_in, s * layer.z_out, polygon) for s in (1, -1)]
            meshes.setdefault((layer.group, style), []).extend(parts)
        half = 0.04
        for st in od.stations:
            if st.kind == "barrel":
                parts = [sector_mesh(pv, st.pos - half, st.pos + half, -st.hi, st.hi, True)]
            else:
                parts = [sector_mesh(pv, st.lo, st.hi, s * (st.pos - half), s * (st.pos + half), True) for s in (1, -1)]
            meshes.setdefault(("muon chambers", "chambers"), []).extend(parts)
        for (group, style), parts in meshes.items():
            merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
            st = dict(OUTER_STYLE[style])
            edge = st.pop("edge")
            if dark:
                st["opacity"] = min(1.0, st["opacity"] * 1.4)
                if style == "chambers":
                    st["color"] = "#cfd8dc"
            self._add(group, merged, smooth_shading=False, show_edges=False, lighting=False, **st)
            outline = merged.extract_feature_edges(feature_angle=20, boundary_edges=True, manifold_edges=False,
                                                   non_manifold_edges=False)
            self._add(group, outline, color=st["color"], opacity=edge, line_width=1.2)

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
            head_path, head_t = path, times
            if t.outer is not None and len(t.outer.path) > 1:
                op = t.outer
                a = self._add("muons outside the tracker", pv.lines_from_points(op.path), color=reco_c, line_width=6,
                              render_lines_as_tubes=True)
                self._anim.append((a, "grow", op.path, op.t))
                if op.stations:
                    a = self._add("muon chamber hits", pv.PolyData(op.station_points), color=th["flash"],
                                  point_size=16, render_points_as_spheres=True)
                    self._anim.append((a, "points", op.station_points, op.station_t))
                if op.stopped_in:
                    a = self._add("muons outside the tracker", pv.PolyData(op.path[-1:]), color=th["bad"],
                                  point_size=18, render_points_as_spheres=False)
                    self._anim.append((a, "points", op.path[-1:], op.t[-1:]))
                head_path, head_t = np.vstack([path, op.path[1:]]), np.concatenate([times, op.t[1:]])
            # the ball grows with the scene, so it stays visible in the whole-detector view
            radius = 0.04 if t.outer is None else 0.09
            head = self._add(FLIGHT, pv.Sphere(radius=radius, center=(0, 0, 0)), color=reco_c)
            self._heads.append((head, head_path, head_t))
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
            if t.outer is not None:
                lines.append(f"     outside the tracker: {t.outer.summary()}")
        if ev.has_outer:
            lines.append("outer detector: schematic (field reversed in the iron, average energy loss)")
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
        # tracks are fitted once both muons have left the tracker (with the outer detector the
        # muons are still flying outwards then); without it, at the end as before
        fitted = bool(final or (self.event.has_outer and t_cut >= self.event.t_tracker))
        off = set()
        for a, kind, pts, times in self._anim:
            ds = a.GetMapper().GetInput()
            if kind in ("line", "grow"):
                if kind == "line":           # the path flown inside the tracker, until the tracks are fitted
                    part = None if fitted else path_until(pts, times, t_cut)
                else:                        # the path outside the tracker: grows, then stays
                    part = pts if final else path_until(pts, times, t_cut)
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
        if not fitted:
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
        """'iso' (the tracker), 'whole' (the whole detector), 'beam' (along z) or 'side'."""
        if view == "beam":
            self.pl.view_xy()
            self.pl.reset_camera()
        elif view == "side":
            self.pl.view_zy()
            self.pl.reset_camera()
        elif view == "whole":
            # oblique view of the cut-away detector, beam horizontal, as in the CMS drawings
            self.pl.camera_position = WHOLE_VIEW
            self.pl.reset_camera()
        else:
            # oblique view of the tracker, close enough to see the hits
            self.pl.camera_position = TRACKER_VIEW
            self.pl.reset_camera(bounds=(-1.3, 1.3, -1.3, 1.3, -3.0, 3.0))
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
        self.controls_btn = QtWidgets.QToolButton(); self.controls_btn.setText("Controls")
        self.controls_btn.setCheckable(True); self.controls_btn.setChecked(True)
        self.controls_btn.setToolTip("Show or hide the camera, visibility and residual controls")
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.bar, 1); row.addWidget(self.controls_btn)
        self.view_box.addLayout(row)
        self.controls = self._build_controls()
        self.controls.setEnabled(False)
        # the control column scrolls vertically, so the tab fits short screens
        self.controls_area = QtWidgets.QScrollArea()
        self.controls_area.setWidget(self.controls)
        self.controls_area.setWidgetResizable(True)
        self.controls_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.controls_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._fit_controls_width()
        from .layout import splitter, wrap
        # the view takes the spare width; the control column keeps its size and can be dragged closed
        self.split = splitter("event3d", [wrap(self.view_box, margins=False), self.controls_area], [1, 0],
                              collapsible=(1,))
        lay = QtWidgets.QHBoxLayout(self)
        lay.addWidget(self.split)
        self._auto_folded = False
        self.controls_btn.toggled.connect(self._show_controls)
        self.split.splitterMoved.connect(lambda *_: self._sync_controls_btn())
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
        self.camera_buttons = {}
        for i, (text, view) in enumerate([("Tracker", "iso"), ("Whole detector", "whole"),
                                          ("Along beam (x–y)", "beam"), ("Side (z–y)", "side")]):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, vw=view: self.builder and self.builder.camera(vw))
            cl.addWidget(b, i // 2, i % 2)
            self.camera_buttons[view] = b
        hint = QtWidgets.QLabel("mouse: left-drag rotate · wheel zoom · shift-drag pan")
        hint.setWordWrap(True)
        cl.addWidget(hint, 2, 0, 1, 2)
        v.addWidget(cam)
        self.checks = {}
        for title, groups in (("Detector", DETECTOR_GROUPS + OUTER_GROUPS), ("Event", EVENT_GROUPS)):
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
            "speed; the fitted track appears when both muons have left the tracker.<br>"
            "<b>Outer detector</b> (View menu; schematic): green ECAL, amber HCAL, grey coil, red iron return "
            "yoke, grey muon stations (MB1–4 barrel, ME1–4 endcap). Outside the coil the field points the other way, "
            "so muons bend back; orange balls: muon-chamber hits; red square: where a muon ran out of energy. "
            "Only muons get this far, and soft ones (below about 3 GeV) stop in the calorimeters or the iron.")
        legend.setWordWrap(True)
        v.addWidget(legend)
        v.addStretch()
        return box

    def minimumSizeHint(self):
        """The 3D view alone: in a narrow window the control column folds away (see resizeEvent)."""
        view = self.split.widget(0).minimumSizeHint()
        m = self.layout().contentsMargins()
        return QtCore.QSize(view.width() + m.left() + m.right(), view.height() + m.top() + m.bottom())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        need = self.split.widget(0).minimumSizeHint().width() + self.controls_area.minimumWidth() + self.split.handleWidth()
        have = self.split.width()
        sizes = self.split.sizes()
        if have < need and sizes[1] > 0:
            self._auto_folded = True
            self.split.setSizes([have, 0])
        elif have >= need and self._auto_folded:
            self._auto_folded = False
            self.split.setSizes([have - self.controls_area.minimumWidth() - self.split.handleWidth(),
                                 self.controls_area.minimumWidth()])
        self._sync_controls_btn()

    def _show_controls(self, on):
        have = self.split.width()
        if on and self.split.sizes()[1] == 0:
            cw = self.controls_area.minimumWidth()
            self.split.setSizes([max(have - cw - self.split.handleWidth(), 0), cw])
        elif not on and self.split.sizes()[1] > 0:
            self.split.setSizes([have, 0])
        self._auto_folded = False

    def _sync_controls_btn(self):
        shown = self.split.sizes()[1] > 0
        if self.controls_btn.isChecked() != shown:
            self.controls_btn.blockSignals(True); self.controls_btn.setChecked(shown); self.controls_btn.blockSignals(False)

    def _fit_controls_width(self):
        """Width of the control column: its content plus a vertical scroll bar (no horizontal scrolling)."""
        self.controls.ensurePolished()
        self.controls.adjustSize()
        w = max(self.controls.sizeHint().width(), self.controls.minimumSizeHint().width())
        # never narrower than the controls (nothing is cut off); the user may widen it
        self.controls_area.setMinimumWidth(w + self.controls_area.verticalScrollBar().sizeHint().width() + 6)

    # ---- lazy loading ------------------------------------------------------------
    def showEvent(self, e):
        super().showEvent(e)
        self._fit_controls_width()
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
            geom = self.builder.geometry
            self.builder.camera("whole" if geom is not None and geom.outer is not None else "iso")
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
        had_outer = self.builder.geometry is not None and self.builder.geometry.outer is not None
        if geom is not None:
            self.builder.set_geometry(geom)
        self.builder.progress = self.clock.fraction
        self.builder.span = self.clock.t_max or None
        self.builder.set_event(ev, self.residual.currentData())
        has_outer = geom is not None and geom.outer is not None
        for g in OUTER_GROUPS + ("muons outside the tracker", "muon chamber hits"):
            if g in self.checks:
                self.checks[g].setEnabled(has_outer)
        self.camera_buttons["whole"].setEnabled(has_outer)
        if has_outer != had_outer:
            self.builder.camera("whole" if has_outer else "iso")

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

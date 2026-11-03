"""PyVista scene construction (off-screen) and GUI responsiveness while the renderer loads."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
os.environ.setdefault("QT_API", "pyside6")

from particlelab.app import theme  # noqa: E402
from particlelab.app.view3d import SceneBuilder, DETECTOR_GROUPS, EVENT_GROUPS, FLIGHT  # noqa: E402
from particlelab.detector import cms_inspired, uniform_barrel  # noqa: E402
from particlelab.visualization.event_display import detector_geometry, make_event, data_event  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def builder():
    pl = pv.Plotter(off_screen=True, window_size=(400, 300))
    b = SceneBuilder(pl, pv)
    yield b
    pl.close()


def test_detector_actors_one_per_group_and_cached(builder):
    g = detector_geometry(cms_inspired())
    builder.set_geometry(g)
    for grp in DETECTOR_GROUPS:
        assert builder.n_actors(grp) == 1, grp
    first = builder.actors["pixel barrel"][0]
    builder.set_geometry(g)                                   # same geometry: nothing rebuilt
    assert builder.actors["pixel barrel"][0] is first
    builder.set_geometry(detector_geometry(uniform_barrel()))  # new geometry: rebuilt
    assert builder.n_actors("pixel barrel") == 0 and builder.n_actors("strip barrel") == 1


def test_detector_mesh_coordinates(builder):
    det = cms_inspired()
    builder.set_geometry(detector_geometry(det))
    mesh = builder.actors["pixel barrel"][0].GetMapper().GetInput()
    pts = np.asarray(pv.wrap(mesh).points)
    r = np.hypot(pts[:, 0], pts[:, 1])
    for b in det.barrels[:3]:
        assert np.any(np.isclose(r, b.radius, rtol=1e-3))
    assert np.isclose(np.abs(pts[:, 2]).max(), det.barrels[0].half_length, rtol=1e-6)


def test_event_actors_replaced_not_accumulated(builder):
    det = cms_inspired()
    builder.set_geometry(detector_geometry(det))
    rng = np.random.default_rng(1)
    for _ in range(4):
        ev = make_event("Z", det, rng)
        builder.set_event(ev)
        n_missed = sum(1 for t in ev.tracks if len(t.missed))
        expected_event = 2 + 2 + 2 + n_missed + 2 + 1 + 4  # true, reco, hits, missed, arrows, vertex, in flight
        assert builder.n_actors() == len(DETECTOR_GROUPS) + expected_event
        # nothing left behind in the renderer from earlier events (+1 info text, +1 axes widget at most)
        assert len(builder.pl.renderer.actors) <= builder.n_actors() + 2
    for grp in ("true tracks", "reconstructed tracks", "momentum vectors"):
        assert builder.n_actors(grp) == 2
    assert builder.n_actors("vertex") == 1
    assert builder.last_update_s < 1.0


def test_track_mesh_uses_simulation_coordinates(builder):
    det = cms_inspired()
    ev = make_event("Upsilon(1S)", det, np.random.default_rng(4))
    builder.set_geometry(detector_geometry(det))
    builder.set_event(ev)
    true_line = pv.wrap(builder.actors["true tracks"][0].GetMapper().GetInput())
    assert np.allclose(true_line.points, ev.tracks[0].true_path, atol=1e-6)    # float32 storage in VTK
    hits = pv.wrap(builder.actors["hits"][0].GetMapper().GetInput())
    assert np.allclose(hits.points, ev.tracks[0].hits, atol=1e-6)


def test_visibility_controls(builder):
    det = cms_inspired()
    builder.set_geometry(detector_geometry(det))
    builder.set_event(make_event("J/psi", det, np.random.default_rng(2)))
    for grp in DETECTOR_GROUPS + EVENT_GROUPS:
        builder.set_visible(grp, False)
        assert all(not a.GetVisibility() for a in builder.actors[grp])
        builder.set_visible(grp, True)
        assert all(a.GetVisibility() for a in builder.actors[grp])
    # hidden groups stay hidden for the next event
    builder.set_visible("hits", False)
    builder.set_event(make_event("J/psi", det, np.random.default_rng(3)))
    assert all(not a.GetVisibility() for a in builder.actors["hits"])


def test_residual_magnification_moves_hit_points(builder):
    det = cms_inspired()
    ev = make_event("Z", det, np.random.default_rng(6))
    builder.set_geometry(detector_geometry(det))
    builder.set_event(ev, residual_scale=100.0)
    pts = pv.wrap(builder.actors["hits"][0].GetMapper().GetInput()).points
    assert np.allclose(pts, ev.tracks[0].hits_exaggerated(100.0), atol=1e-5)
    assert builder.n_actors("hits") == 4        # points + residual segments for each muon


def _points(actor):
    return np.asarray(pv.wrap(actor.GetMapper().GetInput()).points)


def test_animation_reveals_hits_in_flight_order(builder):
    det = cms_inspired()
    ev = make_event("Z", det, np.random.default_rng(6))
    builder.set_geometry(detector_geometry(det))
    builder.set_event(ev)
    t_max = ev.t_max
    for f in (0.0, 0.3, 0.7):
        builder.set_progress(f)
        t_cut = f * t_max
        for tr, hit_actor in zip(ev.tracks, builder.actors["hits"]):
            n = int(np.sum(tr.hit_t <= t_cut))
            if n:
                assert hit_actor.GetVisibility()
                assert np.allclose(_points(hit_actor), tr.hits[:n], atol=1e-6)
            else:
                assert not hit_actor.GetVisibility()
        # fitted tracks and arrows only once both muons have left; the flying muons are shown meanwhile
        for g in ("reconstructed tracks", "true tracks", "momentum vectors"):
            assert not any(a.GetVisibility() for a in builder.actors[g])
        heads = builder.actors[FLIGHT][1::2]
        for tr, head in zip(ev.tracks, heads):
            path, times = tr.flight_path
            expected = path[min(np.searchsorted(times, t_cut), len(times) - 1)]
            assert np.linalg.norm(np.array(head.GetPosition()) - expected) < 0.05
    builder.set_progress(1.0)
    assert all(a.GetVisibility() for a in builder.actors["reconstructed tracks"])
    assert not any(a.GetVisibility() for a in builder.actors[FLIGHT])
    for tr, hit_actor in zip(ev.tracks, builder.actors["hits"]):
        assert len(_points(hit_actor)) == len(tr.hits)
    # a hidden group stays hidden through the animation
    builder.set_visible("hits", False)
    builder.set_progress(0.5)
    assert not any(a.GetVisibility() for a in builder.actors["hits"])


def test_data_event_scene_and_theme(builder, tmp_path):
    det = cms_inspired().with_(B=3.8)
    row = dict(px1=-9.5104, py1=0.3662, pz1=1.8633, Q1=-1, px2=7.3277, py2=-1.1524, pz2=6.3473, Q2=1,
               Run=165617, Event=74601703, type1="G", type2="G")
    ev = data_event(row, "Upsilon(1S)", det)
    builder.set_geometry(detector_geometry(det))
    builder.set_event(ev)
    assert builder.n_actors("true tracks") == 0 and builder.n_actors("hits") == 0
    assert builder.n_actors("layer crossings") == 2 and builder.n_actors("reconstructed tracks") == 2
    assert "run 165617" in builder._text.GetText(2)          # upper-left corner text
    builder.set_theme(theme.THEMES["dark"])
    assert np.allclose(builder.pl.background_color.float_rgb, (0x1e / 255, 0x1f / 255, 0x22 / 255), atol=0.01)
    assert builder.n_actors("layer crossings") == 2            # event redrawn with the new colours
    out = builder.screenshot(tmp_path / "view.png")
    from PIL import Image
    img = np.asarray(Image.open(out))
    assert img.shape[0] > 100 and img[-5, -5, :3].mean() < 60  # dark background in the saved image


def test_gui_stays_responsive_while_pyvista_loads():
    """Run in a fresh interpreter so PyVista is not already imported."""
    script = textwrap.dedent(f"""
        import sys, time, threading
        sys.path.insert(0, {str(ROOT)!r})
        import numpy as np
        from PySide6 import QtWidgets, QtCore
        assert "pyvista" not in sys.modules
        from particlelab.app.view3d import Event3DTab
        from particlelab.detector import cms_inspired
        from particlelab.visualization.event_display import make_event, detector_geometry
        app = QtWidgets.QApplication(["t"])
        class Main: pass
        main = Main(); det = cms_inspired()
        main.current_event = make_event("Z", det, np.random.default_rng(1))
        main.geometry = detector_geometry(det)
        main.new_event = lambda seed=None: None
        tab = Event3DTab(main); tab.resize(600, 400); tab.show()
        ticks = []
        timer = QtCore.QTimer(); timer.timeout.connect(lambda: ticks.append(time.perf_counter())); timer.start(20)
        t0 = time.perf_counter()
        while tab.builder is None and time.perf_counter() - t0 < 120:
            app.processEvents(); time.sleep(0.005)
        gaps = np.diff(ticks) if len(ticks) > 1 else np.array([999.0])
        ident = tab._loader.thread_ident
        off_main = ident is not None and ident != threading.main_thread().ident
        print("RESULT", tab.builder is not None, len(ticks), float(gaps.max()), tab.load_seconds,
              tab.builder.n_actors("reconstructed tracks") if tab.builder else -1, off_main)
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "QT_API": "pyside6"})
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
    assert line, out.stdout + out.stderr
    _, ready, n_ticks, max_gap, load_s, n_tracks, off_main = line[0].split()
    assert ready == "True" and int(n_tracks) == 2, line[0]   # renderer came up and drew the pending event
    assert off_main == "True", line[0]                       # PyVista was imported off the GUI thread
    # the event loop kept running during the import (loose limits: shared CPU in test runs)
    assert float(max_gap) < 2.0, line[0]
    assert int(n_ticks) > 0.25 * float(load_s) / 0.02, line[0]

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

from particlelab.app.view3d import SceneBuilder, DETECTOR_GROUPS, EVENT_GROUPS  # noqa: E402
from particlelab.detector import cms_inspired, uniform_barrel  # noqa: E402
from particlelab.visualization.event_display import detector_geometry, make_event  # noqa: E402

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
        expected_event = 2 + 2 + 2 + n_missed + 2 + 1      # true, reco, hits, missed, arrows, vertex
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

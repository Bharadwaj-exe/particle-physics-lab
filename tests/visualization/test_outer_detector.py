"""Schematic outer detector (display only): field map, energy loss, stopping, the S-bend,
continuity with the tracker, and how the 3D and 2D displays animate it."""
import os

import numpy as np
import pytest

from particlelab.detector import cms_inspired
from particlelab.physics.kinematics import M_MU
from particlelab.units import C_TRACK
from particlelab.visualization.event_display import data_event, detector_geometry, make_event
from particlelab.visualization.outer_detector import (COIL_R_IN, DEDX, cms_outer, propagate)

OD = cms_outer()
B = 3.8


def test_field_map():
    assert OD.bz(1.0, 0.0, B) == B and OD.bz(2.9, 5.9, B) == B           # inside the coil
    assert OD.bz(4.35, 0.0, B) == pytest.approx(-0.5 * B)                # barrel iron: reversed
    assert OD.bz(4.7, 0.0, B) == 0.0 and OD.bz(3.6, 0.0, B) == 0.0       # gap, cryostat
    assert OD.bz(1.0, 6.5, B) == 0.0                                      # beyond the coil ends
    assert 0 < OD.bz(3.1, 0.0, B) < B                                     # across the winding


def _material_to(radius):
    """Energy lost by a straight radial muon at eta = 0 from r = 1.2 m to ``radius`` (GeV)."""
    total = 0.0
    for layer in OD.layers:
        if layer.kind == "barrel":
            total += layer.dedx * max(0.0, min(layer.r_out, radius) - layer.r_in)
    return total


def test_energy_loss_matches_the_material_table():
    # B = 0: a straight radial path, so the path length in each layer is its thickness
    tr = propagate([1.2, 0, 0], [1, 0, 0], 20.0, -1, 0.0, OD, step=0.005)
    p_at = dict((name, p) for name, _, _, p in tr.stations)
    e0 = np.hypot(20.0, M_MU)
    for name, r in (("MB1", 4.02), ("MB4", 7.00)):
        expected_e = e0 - _material_to(r)
        assert np.hypot(p_at[name], M_MU) == pytest.approx(expected_e, abs=0.02), name
    # the numbers quoted in the module docstring
    assert _material_to(4.02) == pytest.approx(1.6, abs=0.1) and _material_to(7.0) == pytest.approx(3.3, abs=0.1)
    assert DEDX["Fe"] == pytest.approx(1.143, rel=1e-3)


def test_soft_muons_stop_hard_muons_cross():
    soft = propagate([1.2, 0, 0], [1, 0, 0], 2.0, -1, B, OD)
    mid = propagate([1.2, 0, 0], [1, 0, 0], 3.2, -1, B, OD)
    hard = propagate([1.2, 0, 0], [1, 0, 0], 8.0, -1, B, OD)
    assert soft.stations == [] and soft.stopped_in is not None
    assert "MB1" in mid.station_names and "MB4" not in mid.station_names and mid.stopped_in
    assert hard.station_names == ["MB1", "MB2", "MB3", "MB4"] and hard.stopped_in is None
    for tr in (soft, mid, hard):
        assert np.all(np.diff(tr.t) > 0) and np.all(np.diff(tr.p) <= 1e-12)
        assert np.all(np.diff(tr.station_t) > 0)
    for name, point, _, _ in hard.stations:                    # station hits lie on the station radius
        r = dict(MB1=4.02, MB2=4.87, MB3=5.90, MB4=7.00)[name]
        assert np.hypot(*point[:2]) == pytest.approx(r, abs=0.02)


def test_s_bend_field_reverses_outside_the_coil():
    tr = propagate([1.2, 0, 0], [1, 0, 0], 10.0, -1, B, OD)
    d = np.diff(tr.path[:, :2], axis=0)
    phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    r = np.hypot(tr.path[1:, 0], tr.path[1:, 1])
    turn = np.diff(phi)
    inside = turn[(r[1:] > 1.3) & (r[1:] < COIL_R_IN)]
    iron = turn[(r[1:] > 5.1) & (r[1:] < 5.6)]
    gap = turn[(r[1:] > 4.6) & (r[1:] < 4.9)]
    assert np.all(np.sign(inside) == np.sign(inside[0])) and np.all(np.sign(iron) == -np.sign(inside[0]))
    assert np.allclose(gap, 0, atol=1e-12)                    # no field between the iron layers
    # positive charges turn the other way
    trp = propagate([1.2, 0, 0], [1, 0, 0], 10.0, +1, B, OD)
    dp = np.diff(trp.path[:20, :2], axis=0)
    assert np.sign(np.diff(np.arctan2(dp[:, 1], dp[:, 0]))[0]) == -np.sign(inside[0])


def test_matches_the_tracker_helix_in_uniform_field():
    """Between the tracker and the ECAL (no material, B uniform) the steps follow the helix exactly."""
    p = np.array([3.0, 1.0, 0.5])
    q = -1
    pt = np.hypot(*p[:2])
    R = pt / (C_TRACK * B)
    tr = propagate([0.0, 0.0, 0.0], p / np.linalg.norm(p), np.linalg.norm(p), q, B, OD, step=0.01, max_steps=100)
    # circle through the origin with centre k R (sin phi0, -cos phi0), k = sign(q)
    phi0 = np.arctan2(p[1], p[0])
    centre = q * R * np.array([np.sin(phi0), -np.cos(phi0)])
    inner = np.hypot(tr.path[:, 0], tr.path[:, 1]) < 1.25
    d = np.hypot(tr.path[inner, 0] - centre[0], tr.path[inner, 1] - centre[1])
    assert np.allclose(d, R, atol=1e-5)
    assert np.allclose(np.diff(tr.path[inner, 2]) / np.hypot(*np.diff(tr.path[inner, :2], axis=0).T), p[2] / pt, atol=1e-6)


def test_endcap_muons_reach_the_endcap_stations_and_go_straight_beyond_the_coil():
    th = 0.35
    tr = propagate([0.4, 0, 1.1], [np.sin(th), 0, np.cos(th)], 15.0, 1, B, OD)
    assert [n for n in tr.station_names if n.startswith("ME")] == ["ME1", "ME2", "ME3", "ME4"]
    beyond = np.abs(tr.path[:, 2]) > 6.1
    d = np.diff(tr.path[beyond], axis=0)
    u = d / np.linalg.norm(d, axis=1)[:, None]
    assert np.allclose(u, u[0], atol=1e-9)                      # no bending outside the coil in the endcap


def test_display_events_continue_from_the_tracker():
    det = cms_inspired()
    ev = make_event("Upsilon(1S)", det, np.random.default_rng(4), outer=True)
    assert ev.has_outer and ev.t_max > ev.t_tracker
    for t in ev.tracks:
        path, times = t.flight_path
        assert np.allclose(t.outer.path[0], path[-1]) and t.outer.t[0] == pytest.approx(times[-1])
        assert t.outer.p[0] == pytest.approx(np.linalg.norm(t.p_true))
    plain = make_event("Upsilon(1S)", det, np.random.default_rng(4))
    assert not plain.has_outer and plain.t_max == plain.t_tracker
    assert np.allclose(plain.tracks[0].hits, ev.tracks[0].hits)    # the tracker part is unchanged
    row = dict(px1=-9.5104, py1=0.3662, pz1=1.8633, Q1=-1, px2=7.3277, py2=-1.1524, pz2=6.3473, Q2=1,
               Run=165617, Event=74601703, type1="G", type2="G")
    dev = data_event(row, "Upsilon(1S)", det.with_(B=3.8), outer=True)
    assert dev.has_outer and all(len(t.outer.path) > 10 for t in dev.tracks)
    geom = detector_geometry(det, outer=True)
    assert geom.outer is not None and geom.key != detector_geometry(det).key


pv = pytest.importorskip("pyvista")
os.environ.setdefault("QT_API", "pyside6")


def test_3d_scene_animates_the_outer_detector():
    from particlelab.app.view3d import FINAL_ONLY, FLIGHT, OUTER_GROUPS, SceneBuilder
    det = cms_inspired()
    pl = pv.Plotter(off_screen=True, window_size=(500, 350))
    try:
        b = SceneBuilder(pl, pv)
        b.set_geometry(detector_geometry(det, outer=True))
        for g in OUTER_GROUPS:
            assert b.n_actors(g) >= 2, g                       # faces and outlines
        ev = make_event("Z", det, np.random.default_rng(7), outer=True)
        b.set_event(ev)
        assert b.n_actors("muons outside the tracker") >= 2 and b.n_actors("muon chamber hits") >= 1
        vis = lambda g: any(a.GetVisibility() for a in b.actors[g])
        b.set_progress(0.5 * ev.t_tracker / ev.t_max)          # still in the tracker
        assert not vis("reconstructed tracks") and not vis("muons outside the tracker") and vis(FLIGHT)
        b.set_progress(0.5 * (ev.t_tracker + ev.t_max) / ev.t_max)   # left the tracker, still flying outwards
        assert all(vis(g) for g in FINAL_ONLY) and vis("muons outside the tracker") and vis(FLIGHT)
        b.set_progress(1.0)
        assert not vis(FLIGHT) and vis("muon chamber hits")
        n_hits = sum(len(t.outer.stations) for t in ev.tracks)
        shown = sum(a.GetMapper().GetInput().GetNumberOfPoints() for a in b.actors["muon chamber hits"])
        assert shown == n_hits
        b.camera("whole")
        img = b.screenshot(os.path.join(os.environ.get("TMP", "."), "outer3d_test.png"))
        assert os.path.getsize(img) > 1000
    finally:
        pl.close()

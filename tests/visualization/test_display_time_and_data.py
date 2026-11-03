"""Flight times for the animation, and display events built from CMS open data (no renderer needed)."""
import numpy as np
import pandas as pd
import pytest

from particlelab.data import cms
from particlelab.detector import cms_inspired
from particlelab.education import from_display_event
from particlelab.physics.kinematics import M_MU
from particlelab.units import C_TRACK
from particlelab.visualization.event_display import (C_LIGHT, DATA_DISPLAY_WINDOWS, data_event, exit_arc_length,
                                                     flight_time, layer_crossings, make_event, path_until,
                                                     pick_data_row)

ROW = dict(px1=-9.5104, py1=0.3662, pz1=1.8633, Q1=-1, px2=7.3277, py2=-1.1524, pz2=6.3473, Q2=1,
           Run=165617, Event=74601703, type1="G", type2="T")     # first event of the CSV, M = 17.49 GeV


def test_flight_time_is_path_length_over_speed():
    p = np.array([3.0, 4.0, 12.0])                      # pT = 5, |p| = 13 GeV
    s = 0.8                                             # transverse arc length, m
    beta = 13 / np.hypot(13, M_MU)
    assert flight_time(s, p) == pytest.approx(s * 13 / 5 / (beta * C_LIGHT), rel=1e-12)
    # a 1 m straight transverse flight of a relativistic muon takes about 3.34 ns
    assert flight_time(1.0, [100.0, 0, 0]) == pytest.approx(1 / C_LIGHT, rel=1e-5)


def test_event_times_are_consistent_with_hits():
    ev = make_event("Z", cms_inspired(), np.random.default_rng(3))
    for t in ev.tracks:
        assert np.all(np.diff(t.true_t) > 0) and t.true_t[0] == 0
        assert np.all(np.diff(t.hit_t) >= 0)            # hits are stored in flight order
        # each hit is crossed when the true path passes its surface (path sampled every ~1 cm)
        for h, th in zip(t.hits_ideal, t.hit_t):
            assert np.linalg.norm(path_until(t.true_path, t.true_t, th)[-1] - h) < 0.02
        assert t.hit_t[-1] <= t.true_t[-1] + 1e-9
    assert ev.t_max == max(t.true_t[-1] for t in ev.tracks)
    assert 1.0 < ev.t_max < 15.0                        # ns: metres at the speed of light


def test_path_until_interpolates_and_clamps():
    path = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.0]])
    t = np.array([0.0, 1.0, 2.0])
    assert np.array_equal(path_until(path, t, -1), path[:1])
    assert np.array_equal(path_until(path, t, 5), path)
    part = path_until(path, t, 1.25)
    assert len(part) == 3 and np.allclose(part[-1], [1, 0.25, 0])


def test_layer_crossings_equal_unsmeared_simulation_hits():
    det = cms_inspired()
    ideal = det.with_(smear=False, multiple_scattering=False, hit_efficiency=1.0)
    ev = make_event("Upsilon(1S)", ideal, np.random.default_rng(5))
    for t in ev.tracks:
        pts, s = layer_crossings(t.p_true, t.charge_true, det, ev.vertex[2])
        assert np.allclose(pts, t.hits, atol=1e-9)
        assert np.allclose(flight_time(s, t.p_true), t.hit_t, rtol=1e-12)


def test_exit_arc_length():
    B = 3.8
    p = np.array([10.0, 0.0, 0.0])                      # R = 8.78 m, central
    R = 10 / (C_TRACK * B)
    assert exit_arc_length(p, B, 1.1, 2.8) == pytest.approx(2 * R * np.arcsin(1.1 / (2 * R)))
    p_fwd = np.array([10.0, 0.0, 100.0])                # leaves through the endcap plane
    assert exit_arc_length(p_fwd, B, 1.1, 2.8) == pytest.approx(0.28)
    p_loop = np.array([0.5, 0.0, 0.0])                  # 2R < r_out: turns back at half a turn
    R_loop = 0.5 / (C_TRACK * B)
    assert exit_arc_length(p_loop, B, 1.1, 2.8) == pytest.approx(np.pi * R_loop)


def test_data_event_from_csv_row():
    det = cms_inspired().with_(B=3.8)
    ev = data_event(ROW, "Upsilon(1S)", det)
    assert ev.is_data and ev.run == 165617 and ev.event_number == 74601703 and ev.muon_types == ("G", "T")
    assert ev.reco_mass == pytest.approx(17.49, abs=0.01)         # CSV column M = 17.4922
    assert np.isnan(ev.true_mass) and np.allclose(ev.vertex, 0)
    mu_m, mu_p = ev.tracks
    assert (mu_m.label, mu_m.charge_reco, mu_p.label, mu_p.charge_reco) == ("μ⁻", -1, "μ⁺", +1)
    assert np.allclose(mu_m.p_reco, [ROW["px1"], ROW["py1"], ROW["pz1"]])
    for t in ev.tracks:
        assert not t.has_truth and len(t.hits) == 0 and len(t.crossings) > 8
        # drawn helix has the radius set by the measured pT in 3.8 T and ends at the tracker boundary
        R = t.pt_reco / (C_TRACK * 3.8)
        xy = t.reco_path[:, :2]
        p_dir = t.p_reco[:2] / t.pt_reco
        centre = t.charge_reco * R * np.array([p_dir[1], -p_dir[0]])
        assert np.allclose(np.linalg.norm(xy - centre, axis=1), R, rtol=1e-9)
        end = t.reco_path[-1]
        assert np.isclose(np.hypot(*end[:2]), 1.08, atol=0.03) or np.isclose(abs(end[2]), 2.8, atol=0.01)
        assert np.all(np.diff(t.crossing_t) >= 0) and t.crossing_t[-1] <= t.reco_t[-1] + 1e-9
    # the learner's reconstruction of a real event gives the CMS mass
    ex = from_display_event(ev)
    assert ex.expected()["mass"] == pytest.approx(ev.reco_mass, rel=1e-9)


def test_pick_data_row_respects_the_window():
    df = pd.DataFrame({"M_recomputed": [3.05, 3.10, 9.40, 50.0, 91.0], "i": range(5)})
    rng = np.random.default_rng(0)
    for proc, allowed in (("J/psi", {0, 1}), ("Upsilon(1S)", {2}), ("Z", {4})):
        assert {int(pick_data_row(df, proc, rng)["i"]) for _ in range(20)} == allowed
    with pytest.raises(LookupError):
        pick_data_row(df.iloc[3:4], "Z", rng)


@pytest.mark.skipif(not cms.available(), reason="CMS open-data file not downloaded")
def test_real_events_in_every_window():
    events = cms.load(check_file=False).events
    det = cms_inspired().with_(B=3.8)
    rng = np.random.default_rng(1)
    for proc, (lo, hi) in DATA_DISPLAY_WINDOWS.items():
        row = pick_data_row(events, proc, rng)
        ev = data_event(row, proc, det)
        assert lo < ev.reco_mass < hi
        assert ev.reco_mass == pytest.approx(row["M_recomputed"], rel=1e-9)

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from particlelab.detector import (uniform_barrel, cms_inspired, simulate_hits, fit_tracks, measure,
                                  gluckstern, helix_at)
from particlelab.units import C_TRACK


def tracks(rng, n, pt_range=(3, 200), eta_range=(-2.4, 2.4)):
    pt = rng.uniform(*pt_range, n); phi = rng.uniform(-np.pi, np.pi, n); eta = rng.uniform(*eta_range, n)
    return np.column_stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)]), rng.choice([-1.0, 1.0], n)


def test_hits_lie_on_layers_and_charge_sign_convention(rng):          # original test 5
    det = uniform_barrel().ideal()
    p = np.array([[30.0, 10.0, 5.0]])
    h = simulate_hits(p, np.array([1.0]), det, rng)
    assert np.allclose(np.hypot(h.x, h.y)[0], det.radii)
    assert np.all(np.diff(np.unwrap(np.arctan2(h.y, h.x)[0])) < 0)      # q > 0 turns clockwise
    h2 = simulate_hits(p, np.array([-1.0]), det, rng)
    assert np.all(np.diff(np.unwrap(np.arctan2(h2.y, h2.x)[0])) > 0)


def test_curvature_radius():
    # R = pT / (0.299792458 B): 3.8 T, 10 GeV → 8.778 m
    assert np.isclose(10 / (C_TRACK * 3.8), 8.7779, rtol=1e-4)


@settings(max_examples=100, deadline=None)
@given(pt=st.floats(1.0, 500.0), phi=st.floats(-3.14, 3.14), cot=st.floats(-5, 5), q=st.sampled_from([-1.0, 1.0]))
def test_property_helix_radius_and_start(pt, phi, cot, q):
    R = pt / (C_TRACK * 3.8)
    x0, y0, z0 = helix_at(0.0, R, q, phi, cot)
    assert abs(x0) < 1e-9 and abs(y0) < 1e-9 and z0 == 0
    s = np.linspace(0, min(1.0, np.pi * R), 20)
    x, y, _ = helix_at(s, R, q, phi, cot)
    xc, yc = q * R * np.sin(phi), -q * R * np.cos(phi)
    assert np.allclose(np.hypot(x - xc, y - yc), R)


@pytest.mark.parametrize("det_factory", [uniform_barrel, cms_inspired])
def test_unsmeared_fit_recovers_momentum_and_charge_exactly(det_factory, rng):   # original test 6
    det = det_factory().ideal()
    p, q = tracks(rng, 3000)
    zv = rng.normal(0, 0.06, len(q)) if det.vertex_sigma_z else None
    f = measure(p, q, det, rng, zv)
    ok = f.ok
    assert ok.mean() > 0.9
    assert np.allclose(f.p[ok], p[ok], rtol=1e-7, atol=1e-9)
    assert np.all(f.q[ok] == q[ok])
    if zv is not None:
        assert np.allclose(f.z0[ok], zv[ok], atol=1e-9)


def test_barrel_has_finite_length_and_endcaps_extend_coverage(rng):
    det = cms_inspired().ideal()
    p = np.array([[10.0, 0.0, 10.0 * np.sinh(2.3)]])
    h = simulate_hits(p, np.array([1.0]), det, rng)
    hit_z = np.abs(h.z[0][h.detected[0]])
    assert h.is_disk[0][h.detected[0]].sum() >= 5            # forward track measured by disks
    assert hit_z.max() < 2.8                                  # never beyond the last disk
    assert h.n_detected[0] >= det.min_hits


def test_track_outside_tracker_is_not_reconstructed(rng):
    det = cms_inspired()
    p = np.array([[5.0, 0.0, 5.0 * np.sinh(3.5)]])            # |η| = 3.5
    f = measure(p, np.array([1.0]), det, rng)
    assert not f.ok[0]


def test_low_pt_looper_does_not_reach_outer_layers(rng):
    det = cms_inspired().ideal()
    pt = 0.3                                                  # 2R = 0.53 m
    h = simulate_hits(np.array([[pt, 0.0, 0.0]]), np.array([1.0]), det, rng)
    r = np.hypot(h.x, h.y)[0][h.crossed[0]]
    assert r.max() < 2 * pt / (C_TRACK * det.B)


def test_hit_efficiency(rng):
    det = cms_inspired().with_(hit_efficiency=0.9)
    p, q = tracks(rng, 4000, eta_range=(-0.5, 0.5))
    h = simulate_hits(p, q, det, rng)
    frac = h.detected.sum() / h.crossed.sum()
    assert abs(frac - 0.9) < 0.01


def test_multiple_scattering_scales_as_one_over_p(rng):
    """At low momentum the MS term dominates: σ(pT)/pT is ~constant in pT, so the
    absolute displacement ∝ 1/p. Check the hit displacement spread scales as 1/p."""
    det = cms_inspired().with_(smear=False)
    ideal = det.ideal()
    out = []
    for pt in (2.0, 8.0):
        p = np.tile([[pt, 0.0, 0.0]], (5000, 1))
        q = np.ones(5000)
        h = simulate_hits(p, q, det, rng)
        h0 = simulate_hits(p, q, ideal, rng)
        out.append(np.std(np.arctan2(h.y, h.x)[:, -1] - np.arctan2(h0.y, h0.x)[:, -1]))
    assert 3.5 < out[0] / out[1] < 4.5


def test_fit_reports_uncertainty_and_quality(rng):
    det = uniform_barrel()
    p, q = tracks(rng, 2000, pt_range=(40, 50), eta_range=(0, 0.01))
    f = measure(p, q, det, rng)
    pull = (np.hypot(f.p[:, 0], f.p[:, 1]) - np.hypot(p[:, 0], p[:, 1])) / f.sigma_pt
    assert 0.9 < np.std(pull) < 1.1                           # hit-only covariance is right without MS
    assert 0.8 < np.median(f.chi2_ndf) < 1.2

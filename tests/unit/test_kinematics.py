import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from particlelab.physics import boost, inv_mass, minkowski_mass, pt_eta_phi, decay_rest, M_MU, generate
from particlelab.physics.decay import sample_cos_theta

vel = st.floats(-0.57, 0.57)


def test_muons_on_shell_and_pair_equals_parent():          # original test 1
    ev = generate("Z", 5000, seed=1)
    assert np.allclose(minkowski_mass(ev.mu_minus), M_MU, atol=1e-6)
    pair = ev.mu_minus + ev.mu_plus
    assert np.allclose(minkowski_mass(pair), ev.mass, rtol=1e-9)
    assert np.allclose(pair, ev.parent, rtol=1e-9, atol=1e-9)


def test_boost_then_inverse_boost_returns_original(rng):    # original test 2
    mum, _ = decay_rest(np.full(1000, 3.0969), rng)
    beta = rng.uniform(-0.55, 0.55, (1000, 3))
    assert np.allclose(boost(boost(mum, beta), -beta), mum, atol=1e-10)


def test_zero_velocity_boost_is_identity():                 # original test 3
    p = np.array([[5.0, 1.0, 2.0, 3.0]])
    assert np.allclose(boost(p, np.zeros((1, 3))), p)


def test_boost_rejects_superluminal():
    with pytest.raises(ValueError):
        boost(np.array([[1.0, 0, 0, 0]]), np.array([[1.0, 0, 0]]))


def test_boost_along_z_matches_textbook():
    # E' = γ(E + β pz), pz' = γ(pz + β E)
    b = 0.6; g = 1 / np.sqrt(1 - b * b)
    p = np.array([[10.0, 1.0, 2.0, 3.0]])
    out = boost(p, np.array([[0, 0, b]]))[0]
    assert np.allclose(out, [g * (10 + b * 3), 1.0, 2.0, g * (3 + b * 10)])


@settings(max_examples=200, deadline=None)
@given(bx=vel, by=vel, bz=vel, px=st.floats(-50, 50), py=st.floats(-50, 50), pz=st.floats(-50, 50))
def test_property_invariant_mass_is_lorentz_invariant(bx, by, bz, px, py, pz):
    p3 = np.array([px, py, pz])
    p4 = np.array([[np.sqrt(p3 @ p3 + 4.0), *p3]])       # mass 2
    out = boost(p4, np.array([[bx, by, bz]]))
    assert np.isclose(minkowski_mass(out)[0], 2.0, rtol=1e-6)


@settings(max_examples=200, deadline=None)
@given(bx=vel, by=vel, bz=vel)
def test_property_boost_composition_with_inverse(bx, by, bz):
    p = np.array([[7.0, 1.0, -2.0, 3.0]])
    b = np.array([[bx, by, bz]])
    assert np.allclose(boost(boost(p, b), -b), p, atol=1e-9)


def test_rest_frame_back_to_back_and_threshold(rng):
    m = np.full(500, 9.4604)
    a, b = decay_rest(m, rng)
    assert np.allclose(a[:, 1:] + b[:, 1:], 0)                         # momentum conservation
    assert np.allclose(a[:, 0] + b[:, 0], m)                           # energy conservation
    assert np.allclose(np.linalg.norm(a[:, 1:], axis=1), np.sqrt(m**2 / 4 - M_MU**2))
    with pytest.raises(ValueError):
        decay_rest(np.array([0.2]), rng)                                # below 2 m_mu


def test_inv_mass_stable_for_collinear_high_energy_pair():
    # two muons at 1 TeV separated by 1e-4 rad: exact m² = 2m² + 2(E1E2 − p1p2 cosθ)
    p = 1000.0; th = 1e-4
    p1 = np.array([[0, 0, p]]); p2 = np.array([[p * np.sin(th), 0, p * np.cos(th)]])
    E = np.sqrt(p * p + M_MU**2)
    from mpmath import mp, mpf, cos, sqrt
    mp.dps = 50
    exact = sqrt(2 * mpf(M_MU)**2 + 2 * (mpf(E)**2 - mpf(p)**2 * cos(mpf(th))))
    assert np.isclose(inv_mass(p1, p2)[0], float(exact), rtol=1e-12)


def test_pt_eta_phi_edge_cases():
    pt, eta, phi = pt_eta_phi(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]))
    assert np.allclose(pt, [5, 0]) and eta[0] == 0 and np.isinf(eta[1])


def test_one_plus_cos2_distribution():                       # original test 4
    c = sample_cos_theta(400_000, np.random.default_rng(3), 1.0)
    assert abs(np.mean(c**2) - 0.4) < 0.003


@pytest.mark.parametrize("lam", [-1.0, -0.5, 0.0, 0.5])
def test_general_lambda_theta(lam):
    # <cos²θ> for 1 + λcos²θ is (1/3 + λ/5)/(1 + λ/3)
    c = sample_cos_theta(300_000, np.random.default_rng(4), lam)
    assert abs(np.mean(c**2) - (1 / 3 + lam / 5) / (1 + lam / 3)) < 0.003

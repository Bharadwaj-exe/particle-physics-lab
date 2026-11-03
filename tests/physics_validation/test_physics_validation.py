"""End-to-end physics checks with known answers."""
import numpy as np
import pytest

from particlelab.domain.particles import load, RESONANCES
from particlelab.physics import generate, cos_theta_cs
from particlelab.physics.production import sample_pt, sample_y
from particlelab.detector import uniform_barrel, cms_inspired, measure, gluckstern, ms_resolution
from particlelab.analysis import simulate, fit_peak, toy_study
from particlelab.analysis.studies import pt_resolution


@pytest.mark.parametrize("name", RESONANCES)
def test_zero_smearing_reconstructs_input_mass_exactly(name):     # original test 7
    r = simulate(name, 30000, seed=5, smear=False)
    assert r.reco_mass.size > 100
    assert np.allclose(r.reco_mass, r.true_mass, rtol=1e-7)


@pytest.mark.parametrize("pt", [5.0, 20.0, 50.0, 100.0])
def test_resolution_matches_gluckstern(pt):                       # original tests 8-11
    det = uniform_barrel()
    rms, _, eff = pt_resolution(det, pt, 0.0, 20000, np.random.default_rng(6))
    assert eff == 1.0
    assert 0.9 < rms / gluckstern(pt, det) < 1.1


def test_multiple_scattering_floor_matches_analytic_estimate():
    """At low pT the CMS-inspired resolution is dominated by multiple scattering and
    should agree with the uniform-scatterer estimate to within a factor of two."""
    det = cms_inspired().with_(smear=False)
    _, robust, _ = pt_resolution(det, 3.0, 0.0, 20000, np.random.default_rng(8))
    est = ms_resolution(det, 3.0)
    assert 0.5 < robust / est < 2.0


def test_collins_soper_distribution_recovered_by_independent_formula():
    ev = generate("Z", 300_000, seed=9)
    c = cos_theta_cs(ev.mu_minus, ev.mu_plus)
    assert abs(np.mean(c**2) - 0.4) < 0.003                       # 1 + cos²θ
    assert abs(np.mean(c)) < 0.005                                # symmetric (no A_FB modelled)


def test_quarkonia_are_unpolarised():
    ev = generate("J/psi", 300_000, seed=10)
    c = cos_theta_cs(ev.mu_minus, ev.mu_plus)
    assert abs(np.mean(c**2) - 1 / 3) < 0.003


def test_production_spectra_match_their_definitions(rng):
    pt = sample_pt(400_000, 4.5, 3.5, rng)
    # CDF: 1 − (1 + pT²/p0²)^(1−n) at pT = p0 → 1 − 2^(−2.5)
    assert abs(np.mean(pt < 4.5) - (1 - 2 ** -2.5)) < 0.003
    y = sample_y(400_000, 2.0, 3.6, rng)
    assert np.abs(y).max() <= 3.6 and abs(np.mean(np.abs(y) < 2.0) - 2.0 / 2.8) < 0.003


@pytest.mark.parametrize("name", RESONANCES)
def test_fit_recovers_input_mass_ideal_detector(name):           # original tests 13-15
    r = simulate(name, 60000 if name != "J/psi" else 200000, seed=7, det=cms_inspired().with_(multiple_scattering=False))
    sig = {"J/psi": "double_gauss", "Upsilon(1S)": "double_gauss", "Z": "rbw_gauss"}[name]
    f = fit_peak(r.reco_mass, name, background=False, signal=sig,
                 window={"J/psi": (3.04, 3.16), "Upsilon(1S)": (9.36, 9.56), "Z": (70, 110)}[name])
    M = load().get(name).mass
    assert abs(f.mass - M) < 4 * f.mass_err + 2e-4 * M, (f.mass, f.mass_err)


def test_curvature_is_unbiased_and_pt_bias_follows_resolution():
    """1/pT is measured with Gaussian errors, so <1/pT> is unbiased while <pT> is
    biased upward by ≈ σ_rel². This explains the small positive mass shifts."""
    det = cms_inspired()
    rng = np.random.default_rng(11)
    n, pt = 60000, 20.0
    phi = rng.uniform(-np.pi, np.pi, n)
    p = np.column_stack([pt * np.cos(phi), pt * np.sin(phi), np.full(n, pt * np.sinh(1.0))])
    f = measure(p, rng.choice([-1.0, 1.0], n), det, rng)
    ptr = np.hypot(f.p[f.ok, 0], f.p[f.ok, 1])
    k = pt / ptr - 1
    assert abs(k.mean()) < 4 * k.std() / np.sqrt(k.size)
    assert abs((ptr / pt - 1).mean() - k.var()) < 4 * k.std() / np.sqrt(k.size)


@pytest.mark.slow
def test_toy_pulls_have_unit_width():
    rng = np.random.default_rng(12)
    m = np.concatenate([rng.normal(3.097, 0.025, 3000), 2.9 + rng.exponential(0.5, 6000)])
    f = fit_peak(m[m < 3.3], "J/psi", signal="gauss")
    t = toy_study(f, 150, "mu", rng)
    assert t.n_failed == 0
    assert abs(t.pull_mean) < 0.25 and 0.85 < t.pull_std < 1.15

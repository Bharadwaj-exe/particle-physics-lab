import numpy as np
import pandas as pd
import pytest

from particlelab.analysis import fit_peak, fit_counts
from particlelab.analysis.shapes import SIGNALS, crystal_ball, _cb_integral, rbw_gauss, rbw_cb
from particlelab.physics.production import rbw_pdf_unnorm
from particlelab.data import cms
from particlelab.physics.kinematics import M_MU


def test_crystal_ball_normalisation_and_continuity():
    m = np.linspace(-2000, 10, 2_000_001)       # power-law tail: integrate far out
    for a, n in [(1.0, 3.0), (2.0, 5.0)]:
        f = crystal_ball(m, 0.0, 1.0, a, n)
        assert np.isclose(np.trapezoid(f, m), _cb_integral(1.0, a, n), rtol=2e-3)
        left, right = crystal_ball(np.array([-a - 1e-9, -a + 1e-9]), 0.0, 1.0, a, n)
        assert np.isclose(left, right, rtol=1e-6)                   # continuous at t = −α


def test_rbw_convolution_has_no_shift():
    # convolving with a very narrow kernel must reproduce the input shape
    m = np.linspace(80, 100, 2001)
    base = rbw_pdf_unnorm(m, 91.1879, 2.4955)
    for g in (rbw_gauss(m, 91.1879, 0.01, 2.4955), rbw_cb(m, 91.1879, 0.01, 5.0, 3.0, 2.4955)):
        assert m[np.argmax(g)] == pytest.approx(m[np.argmax(base)], abs=0.011)


def test_fit_with_background_recovers_yield(rng):                 # original test 16 (now likelihood)
    sig = rng.normal(3.0969, 0.03, 20000)
    bkg = 2.9 + rng.exponential(0.6, 40000)
    m = np.concatenate([sig, bkg[bkg < 3.3]])
    f = fit_peak(m, "J/psi", signal="gauss")
    n_sig = np.sum((sig > 2.9) & (sig < 3.3))
    assert abs(f.mass - 3.0969) < 4 * f.mass_err
    assert abs(f.n_signal - n_sig) < 4 * f.n_signal_err
    assert f.converged and f.pvalue > 1e-3


def test_likelihood_handles_empty_bins(rng):
    m = rng.normal(9.46, 0.05, 300)                                # sparse histogram
    f = fit_peak(m, "Upsilon(1S)", signal="gauss", background=False, bins=200)
    assert abs(f.mass - 9.46) < 4 * f.mass_err


def test_fixed_width_voigt_is_biased_for_the_z(rng):
    """The PDG Z mass uses an s-dependent width. Fitting a fixed-width line shape to
    a running-width resonance shifts the mass by about Γ²/2M ≈ 34 MeV."""
    from particlelab.physics.production import sample_mass
    m = sample_mass("Z", 300_000, rng)
    good = fit_peak(m, "Z", background=False, signal="rbw_gauss")
    bad = fit_peak(m, "Z", background=False, signal="voigt")
    assert abs(good.mass - 91.1879) < 4 * good.mass_err
    assert 0.020 < bad.mass - good.mass < 0.050


def test_multi_peak_fit(rng):
    m = np.concatenate([rng.normal(9.46, 0.08, 6000), rng.normal(10.023, 0.085, 2000),
                        rng.normal(10.355, 0.088, 1200), 8.5 + rng.exponential(2.0, 8000)])
    m = m[(m > 8.5) & (m < 11.5)]
    f = fit_peak(m, "Upsilon(1S)", window=(8.5, 11.5), bins=120, signal="gauss", extra_peaks=(10.02, 10.35))
    assert abs(f.mass - 9.46) < 4 * f.mass_err
    assert abs(f.params["mu2"] - 10.023) < 0.01 and abs(f.params["f2"] - 1 / 3) < 0.06


# ---- CMS data pipeline -----------------------------------------------------

def _fake_rows(n=50):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p1 = rng.normal(0, 10, 3); p2 = rng.normal(0, 10, 3)
        e1 = np.sqrt(p1 @ p1 + M_MU**2); e2 = np.sqrt(p2 @ p2 + M_MU**2)
        pt1, pt2 = np.hypot(*p1[:2]), np.hypot(*p2[:2])
        M = np.sqrt((e1 + e2)**2 - np.sum((p1 + p2)**2))
        rows.append([1, i, "G", e1, *p1, pt1, np.arcsinh(p1[2] / pt1), np.arctan2(p1[1], p1[0]), 1,
                     "T", e2, *p2, pt2, np.arcsinh(p2[2] / pt2), np.arctan2(p2[1], p2[0]), -1, M])
    return pd.DataFrame(rows, columns=cms.EXPECTED_COLUMNS)


def test_validation_counts_every_rejection():
    df = _fake_rows()
    df.loc[0, "Q1"] = 0
    df.loc[1, "E1"] = 0.0
    df.loc[2, "px2"] = np.nan
    df.loc[3, "type2"] = "X"
    df.loc[4, "Q2"] = 1                     # same sign
    valid, rejected, rep = cms.validate(df)
    assert rep.n_rows == 50 and rep.n_rejected == 5 and len(valid) == 45 and len(rejected) == 5
    assert rep.rejected["charge not ±1"] >= 1 and rep.rejected["muon 1: E < |p|"] == 1
    assert rep.rejected["non-finite or non-numeric value"] == 1 and rep.rejected["unknown muon type"] == 1
    assert all(rejected["reasons"].str.len() > 0)


def test_schema_error(tmp_path):
    df = _fake_rows().drop(columns=["M"])
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(cms.SchemaError):
        cms.load_raw(p)


def test_selection_cutflow_and_recomputed_mass():
    valid, _, rep = cms.validate(_fake_rows(200))
    assert rep.mass_check["max_abs_diff_GeV"] < 1e-9
    sel, flow = cms.select(valid, cms.Selection(pt_min=5.0, eta_max=1.5))
    counts = list(flow.values())
    assert counts == sorted(counts, reverse=True) and counts[0] == 200 and len(sel) == counts[-1]
    assert (sel.pt1_re > 5).all() and (sel.eta2_re.abs() < 1.5).all()


@pytest.mark.skipif(not cms.available(), reason="CMS file not downloaded")
def test_real_file_integrity_and_schema():
    v = cms.verify()
    assert v["size_ok"] and v["checksum_ok"]
    d = cms.load()
    assert d.report.n_rows == 100000 and d.report.n_rejected == 0
    assert d.report.mass_check["max_abs_diff_GeV"] < 1e-3   # published M = recomputed M


# --- audit A4: smooth line shapes, bounded fallback, separate status flags ------------

def test_rbw_line_shapes_are_smooth_in_sigma():
    """The convolution grid must not move with σ: second differences of the shape in σ
    must be at the level of rounding, not of the grid spacing."""
    m = np.linspace(75.0, 105.0, 301)
    sig = 0.9 + 1e-5 * np.arange(41)
    for fn in (lambda s: rbw_gauss(m, 91.19, s, 2.4955), lambda s: rbw_cb(m, 91.19, s, 1.2, 2.0, 2.4955)):
        v = np.array([fn(s) for s in sig])
        d2 = np.abs(np.diff(v, 2, axis=0)).max()
        assert d2 < 1e-9 * np.abs(v).max()


def test_fallback_minimisers_respect_parameter_bounds(rng, monkeypatch):
    """Force the first L-BFGS-B call to stop early, so every fallback runs, and check
    that no likelihood evaluation leaves the physical region (σ > 0, α > 0, n > 1)."""
    from particlelab.analysis import fitting
    from particlelab.physics.production import sample_mass
    m = np.concatenate([sample_mass("Z", 1500, rng) + rng.normal(0, 1.2, 1500), 70 + rng.exponential(25, 150)])
    m = m[(m > 70) & (m < 110)]
    seen, calls = [], {"n": 0}
    real_min, real_exp = fitting.optimize.minimize, fitting._Model.expected

    def first_call_fails(fun, x0, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            kw = dict(kw, options=dict(kw.get("options", {}), maxiter=2))
        return real_min(fun, x0, **kw)

    def spy(self, x):
        seen.append(np.array(x, float))
        return real_exp(self, x)

    monkeypatch.setattr(fitting.optimize, "minimize", first_call_fails)
    monkeypatch.setattr(fitting._Model, "expected", spy)
    f = fit_peak(m, "Z", signal="rbw_cb")
    assert calls["n"] >= 2                                  # the fallback path ran
    xs = np.array(seen)
    names = f.names
    for k in ("sigma", "alpha"):
        assert (xs[:, names.index(k)] > 0).all()
    assert (xs[:, names.index("n")] > 1).all()
    assert np.isfinite(f.mass) and f.optimizer_ok
    assert f.converged == (f.optimizer_ok and f.hessian_ok)


@pytest.mark.skipif(not cms.available(), reason="CMS file not downloaded")
def test_small_cms_z_subsample_fits_without_raising():
    """The 1 305-event barrel–barrel Z subsample crashed the old unbounded fallback."""
    from particlelab.analysis.compare import DATA_FIT
    ev = cms.load(selection=cms.Selection(pt_min=3.0, eta_max=2.4)).events
    bb = (ev["eta1_re"].abs() < 0.9) & (ev["eta2_re"].abs() < 0.9)
    f = fit_peak(ev["M_recomputed"].to_numpy()[bb.to_numpy()], "Z", **DATA_FIT["Z"])
    assert 90.0 < f.mass < 92.0 and f.optimizer_ok

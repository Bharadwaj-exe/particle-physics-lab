"""The optional compiled kernels (rust/) against the NumPy reference.

Python draws every random number, so for the same seed both versions must give the
same hits and fits up to floating-point rounding. Skipped if particlelab_fast is not built.
"""
import numpy as np
import pytest

from particlelab.detector import _backend, cms_inspired, uniform_barrel
from particlelab.detector.simulation import _simulate_hits_fast, _simulate_hits_numpy
from particlelab.detector.tracking import _fit_tracks_numpy, fit_tracks
from particlelab.physics import generate

pytestmark = pytest.mark.skipif(not _backend.available(), reason="particlelab_fast not built (see rust/README)")

_b = cms_inspired()
CONFIGS = {
    "cms": _b,
    "no scattering": _b.with_(multiple_scattering=False),
    "ideal": _b.ideal(),
    "hit efficiency 0.9": _b.with_(hit_efficiency=0.9),
    "beam-spot constraint": _b.with_(beamspot_constraint=20e-6),
    "uniform barrel": uniform_barrel(),
}


def _hits(det, proc="Z", n=4000, seed=3):
    ev = generate(proc, n, seed=seed)
    zv = np.random.default_rng(9).normal(0, det.vertex_sigma_z, n)
    a = _simulate_hits_numpy(ev.mu_minus, -np.ones(n), det, np.random.default_rng(1), zv)
    b = _simulate_hits_fast(ev.mu_minus, -np.ones(n), det, np.random.default_rng(1), zv)
    return a, b


@pytest.fixture
def rust_then_restore():
    yield
    _backend.use("auto")


@pytest.mark.parametrize("label", list(CONFIGS))
@pytest.mark.parametrize("proc", ["J/psi", "Z"])
def test_hits_match_numpy(label, proc):
    a, b = _hits(CONFIGS[label], proc)
    assert a.x.shape == b.x.shape
    assert np.array_equal(a.crossed, b.crossed) and np.array_equal(a.detected, b.detected)
    cr = a.crossed
    assert np.array_equal(a.surface[cr], b.surface[cr]) and np.array_equal(a.is_disk[cr], b.is_disk[cr])
    for u, v in ((a.x, b.x), (a.y, b.y), (a.z, b.z)):
        assert np.max(np.abs(u - v)[cr]) < 1e-12          # metres
    assert np.allclose(a.s[cr], b.s[cr], rtol=1e-12)
    assert np.array_equal(a.sigma_rphi[cr], b.sigma_rphi[cr])


@pytest.mark.parametrize("label", list(CONFIGS))
def test_fits_match_numpy(label, rust_then_restore):
    det = CONFIGS[label]
    h, _ = _hits(det)
    ref = _fit_tracks_numpy(h, det)
    _backend.use("rust")
    out = fit_tracks(h, det)
    ok = ref.ok
    assert np.array_equal(ref.ok, out.ok) and np.array_equal(ref.n_hits, out.n_hits)
    assert np.array_equal(ref.q[ok], out.q[ok])
    assert np.max(np.abs(ref.p[ok] - out.p[ok]) / np.linalg.norm(ref.p[ok], axis=1)[:, None]) < 1e-8
    assert np.allclose(ref.z0[ok], out.z0[ok], atol=1e-10) and np.allclose(ref.d0[ok], out.d0[ok], atol=1e-10)
    assert np.allclose(ref.chi2_ndf[ok], out.chi2_ndf[ok], rtol=1e-6, atol=1e-9)
    # σ(pT) comes from inverting a nearly singular matrix for stiff tracks: equal to rounding of that inverse
    assert np.allclose(ref.sigma_pt[ok], out.sigma_pt[ok], rtol=1e-2)


def test_full_simulation_identical(rust_then_restore):
    from particlelab.analysis import simulate
    _backend.use("numpy")
    a = simulate("Upsilon(1S)", 8000, seed=4, preselect=True)
    _backend.use("rust")
    b = simulate("Upsilon(1S)", 8000, seed=4, preselect=True)
    assert a.cutflow == b.cutflow
    assert np.allclose(a.reco_mass, b.reco_mass, rtol=1e-10)


def test_extreme_settings_agree_statistically(rust_then_restore):
    """With B = 0.5 T, hit errors x20 and material x10 many fits do not converge in the
    allowed iterations; rounding then changes individual fits in both versions, so the
    comparison is of distributions."""
    from particlelab.analysis import simulate
    det = _b.with_(B=0.5).scaled_resolution(20).scaled_material(10)
    _backend.use("numpy")
    a = simulate("Z", 6000, det, seed=2)
    _backend.use("rust")
    b = simulate("Z", 6000, det, seed=2)
    from scipy.stats import ks_2samp
    na, nb = a.reco_mass.size, b.reco_mass.size
    assert abs(na - nb) < 0.01 * na
    # the peak is 70 GeV wide here; the two mass distributions must be statistically the same
    assert ks_2samp(a.reco_mass, b.reco_mass).pvalue > 0.05


def test_empty_input_and_backend_switch(rust_then_restore):
    det = _b
    h = _simulate_hits_fast(np.zeros((0, 3)), np.zeros(0), det, np.random.default_rng(0), np.zeros(0))
    assert h.x.shape[0] == 0
    _backend.use("rust")
    assert _backend.active() == "rust" and fit_tracks(h, det).p.shape == (0, 3)
    _backend.use("numpy")
    assert _backend.active() == "numpy"
    with pytest.raises(ValueError):
        _backend.use("fortran")

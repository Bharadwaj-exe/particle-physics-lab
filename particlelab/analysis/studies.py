"""Detector-performance studies and toy Monte Carlo validation of the fitter."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detector import DetectorConfig, measure
from .fitting import FitResult, fit_counts


def robust_sigma(x):
    """Half-width of the central 68.3% interval (insensitive to rare tails)."""
    lo, hi = np.percentile(x, [15.865, 84.135])
    return 0.5 * (hi - lo)


def pt_resolution(det: DetectorConfig, pt: float, eta: float = 0.0, n: int = 20000, rng=None):
    """Return (rms, robust σ, efficiency) of pT_reco/pT_true − 1 for single muons."""
    rng = rng or np.random.default_rng()
    phi = rng.uniform(-np.pi, np.pi, n)
    p = np.column_stack([pt * np.cos(phi), pt * np.sin(phi), np.full(n, pt * np.sinh(eta))])
    q = rng.choice([-1.0, 1.0], n)
    zv = rng.normal(0, det.vertex_sigma_z, n) if det.vertex_sigma_z > 0 else None
    f = measure(p, q, det, rng, zv)
    r = np.hypot(f.p[f.ok, 0], f.p[f.ok, 1]) / pt - 1
    if r.size < 10:
        return np.nan, np.nan, f.ok.mean()
    return float(np.std(r)), float(robust_sigma(r)), float(f.ok.mean())


def resolution_curve(det, pts, eta=0.0, n=20000, rng=None):
    rng = rng or np.random.default_rng()
    return np.array([pt_resolution(det, pt, eta, n, rng) for pt in pts])


@dataclass
class ToyStudy:
    parameter: str
    truth: float
    pulls: np.ndarray
    n_failed: int

    @property
    def pull_mean(self): return float(np.mean(self.pulls))
    @property
    def pull_std(self): return float(np.std(self.pulls, ddof=1))
    @property
    def pull_mean_err(self): return self.pull_std / np.sqrt(len(self.pulls))


def toy_study(fit: FitResult, n_toys: int = 200, parameter: str = "mu", rng=None) -> ToyStudy:
    """Generate Poisson pseudo-datasets from a fitted model, refit, and collect pulls.

    Unbiased estimates with correct errors give pulls with mean 0 and width 1.
    """
    rng = rng or np.random.default_rng()
    mu = fit.expected()
    truth = fit.params[parameter]
    pulls, failed = [], 0
    for _ in range(n_toys):
        counts = rng.poisson(mu)
        r = fit_counts(counts, fit.edges, fit.name, fit.signal, fit.background, init=dict(fit.params),
                       extra_peaks=fit.extra_peaks)
        e = r.errors[parameter]
        if not r.converged or not np.isfinite(e) or e <= 0:
            failed += 1
            continue
        pulls.append((r.params[parameter] - truth) / e)
    return ToyStudy(parameter, truth, np.array(pulls), failed)

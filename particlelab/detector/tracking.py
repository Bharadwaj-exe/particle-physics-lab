"""Track reconstruction from hits: weighted circle fit + weighted s–z line fit.

Transverse plane: weighted algebraic (Kåsa) fit as a starting point, refined by
Gauss–Newton iterations of the weighted geometric fit
    minimise Σ w_i (|h_i − c| − R)²,  w_i = 1/σ_rφ,i²,
optionally with a beam-spot pseudo-hit at (0, 0).

Longitudinal: z = z0 + cot θ · s, weighted least squares, where s is the arc
length to each hit. Disk hits have exact z and an uncertain radius; their
equivalent z error is σ_r |cot θ|.

The covariance returned (sigma_pt) contains hit errors only. Multiple-scattering
correlations are not in the fit weights, so for low-momentum tracks the true
spread is larger than sigma_pt; a Kalman filter would be the full solution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import C_TRACK
from .geometry import DetectorConfig
from .simulation import Hits


@dataclass
class TrackFit:
    p: np.ndarray          # (n,3) momentum at the vertex
    q: np.ndarray          # (n,) charge ±1
    ok: np.ndarray         # (n,) fit succeeded and enough hits
    n_hits: np.ndarray
    sigma_pt: np.ndarray   # hit-error-only uncertainty on pT
    d0: np.ndarray         # signed transverse impact parameter
    z0: np.ndarray
    chi2_ndf: np.ndarray   # transverse fit chi2 / ndf (hit errors only)


def _normal(cols, w):
    """Weighted normal matrix M_ij = Σ_k w c_i c_j for a list of (n, K) columns."""
    k = len(cols)
    M = np.empty((cols[0].shape[0], k, k))
    wc = [w * c for c in cols]
    for i in range(k):
        for j in range(i, k):
            M[:, i, j] = M[:, j, i] = np.sum(wc[i] * cols[j], 1)
    return M, wc


def _wsolve(cols, rhs, w):
    """Batched weighted least squares: argmin Σ w (Σ_i x_i c_i − rhs)²."""
    M, wc = _normal(cols, w)
    v = np.stack([np.sum(c * rhs, 1) for c in wc], -1)
    return np.linalg.solve(M, v[..., None])[..., 0], M


def fit_tracks(h: Hits, det: DetectorConfig, iterations: int = 6) -> TrackFit:
    n = h.x.shape[0]
    nh = h.n_detected
    ok = nh >= max(det.min_hits, 3)
    out = TrackFit(np.full((n, 3), np.nan), np.zeros(n), ok.copy(), nh, np.full(n, np.nan),
                   np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    x, y, z, s_true = h.x[idx], h.y[idx], h.z[idx], h.s[idx]
    det_ = h.detected[idx]
    w = np.where(det_, 1.0 / h.sigma_rphi[idx]**2, 0.0)
    if det.beamspot_constraint > 0:
        x = np.hstack([np.zeros((len(idx), 1)), x])
        y = np.hstack([np.zeros((len(idx), 1)), y])
        w = np.hstack([np.full((len(idx), 1), 1.0 / det.beamspot_constraint**2), w])
    xs, ys = np.where(w > 0, x, 0.0), np.where(w > 0, y, 0.0)

    # algebraic start: x² + y² = 2a x + 2b y + c
    a, b, c = _wsolve([2 * xs, 2 * ys, np.ones_like(xs)], xs**2 + ys**2, w)[0].T
    R = np.sqrt(np.maximum(a**2 + b**2 + c, 1e-12))
    ones = np.ones_like(xs)
    for _ in range(iterations):
        dx, dy = xs - a[:, None], ys - b[:, None]
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        step, M = _wsolve([-dx / d, -dy / d, -ones], d - R[:, None], w)
        a, b, R = a - step[:, 0], b - step[:, 1], R - step[:, 2]
        if np.max(np.abs(step[:, 2]) / R) < 1e-11:
            break
    dx, dy = xs - a[:, None], ys - b[:, None]
    d = np.maximum(np.hypot(dx, dy), 1e-12)
    M, _ = _normal([-dx / d, -dy / d, -ones], w)
    cov = np.linalg.inv(M)
    res = d - R[:, None]
    ndf = np.maximum((w > 0).sum(1) - 3, 1)
    chi2 = np.sum(w * res**2, 1) / ndf

    pt = C_TRACK * det.B * R
    # charge from the turning direction between the first and last detected hits
    first = np.argmax(det_, axis=1)
    last = det_.shape[1] - 1 - np.argmax(det_[:, ::-1], axis=1)
    r_ = np.arange(len(idx))
    x1, y1 = h.x[idx][r_, first], h.y[idx][r_, first]
    x2, y2 = h.x[idx][r_, last], h.y[idx][r_, last]
    k = -np.sign(x1 * y2 - y1 * x2)
    k[k == 0] = 1.0
    phi0 = np.arctan2(b, a) + k * np.pi / 2
    d0 = np.hypot(a, b) - R

    # longitudinal fit
    rho = np.hypot(h.x[idx], h.y[idx])
    s = 2 * R[:, None] * np.arcsin(np.clip(rho / (2 * R[:, None]), -1, 1))
    disk = h.is_disk[idx]
    sl = h.sigma_long[idx]
    cot = np.zeros(len(idx))
    z0 = np.zeros(len(idx))
    for it in range(2):
        sig_eq = np.where(disk, sl * np.maximum(np.abs(cot), 0.1)[:, None] if it else sl, sl)
        wz = np.where(det_ & np.isfinite(sig_eq) & (sig_eq > 0), 1.0 / np.where(np.isfinite(sig_eq), sig_eq, 1)**2, 0.0)
        enough = (wz > 0).sum(1) >= 2
        wz[~enough] = np.where(det_[~enough], 1.0, 0.0)   # fall back to unweighted
        sol, _ = _wsolve([np.ones_like(s), s], np.where(det_, h.z[idx], 0.0), wz)
        z0, cot = sol[:, 0], sol[:, 1]

    p = np.column_stack([pt * np.cos(phi0), pt * np.sin(phi0), pt * cot])
    good = np.isfinite(p).all(1) & (R > 0)
    out.p[idx] = p
    out.q[idx] = k
    out.sigma_pt[idx] = C_TRACK * det.B * np.sqrt(np.abs(cov[:, 2, 2]))
    out.d0[idx] = d0
    out.z0[idx] = z0
    out.chi2_ndf[idx] = chi2
    out.ok[idx] = good
    return out


def measure(p, q, det: DetectorConfig, rng, z_vertex=None) -> TrackFit:
    """Full chain for a set of true tracks: hits → fit."""
    from .simulation import simulate_hits
    return fit_tracks(simulate_hits(p, q, det, rng, z_vertex), det)


def gluckstern(pt, det: DetectorConfig, n_hits: int | None = None):
    """σ(pT)/pT from point resolution only (Gluckstern, NIM 24 (1963) 381), N equally spaced points."""
    N = len(det.barrels) if n_hits is None else n_hits
    return det.sigma_rphi * np.asarray(pt) / (C_TRACK * det.B * det.lever_arm**2) * np.sqrt(720 / (N + 4))


def ms_resolution(det: DetectorConfig, p, sin_theta=1.0):
    """Approximate multiple-scattering term of σ(pT)/pT for a uniformly distributed scatterer.

    σ(pT)/pT ≈ 0.0136 √(x/X0) / (0.3 B L β) · √(1.43)/ sinθ-dependence omitted for θ = 90°.
    (PDG tracking review, uniform-material approximation; order-of-magnitude check only.)
    """
    x = det.material_at_eta0() / sin_theta
    return 0.0136 * np.sqrt(x) * np.sqrt(1.43) / (C_TRACK * det.B * det.lever_arm) * np.ones_like(np.asarray(p, float))

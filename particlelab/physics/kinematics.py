"""Relativistic kinematics. Four-vectors are arrays of shape (n, 4): (E, px, py, pz) in GeV."""
from __future__ import annotations

import numpy as np

from ..domain.particles import muon_mass

M_MU = muon_mass()


def boost(p4: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Boost four-vectors from a frame moving with velocity ``beta`` (n,3) into the lab.

    E' = γ(E + β·p),  p' = p + [γ²/(γ+1) (β·p) + γE] β
    (γ²/(γ+1) equals (γ−1)/β², written this way so that β = 0 is safe.)
    """
    p4 = np.atleast_2d(p4)
    beta = np.atleast_2d(beta)
    b2 = np.sum(beta**2, axis=1)
    if np.any(b2 >= 1.0):
        raise ValueError("boost velocity must satisfy |beta| < 1")
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * p4[:, 1:], axis=1)
    k = gamma**2 / (gamma + 1.0) * bp + gamma * p4[:, 0]
    return np.column_stack([gamma * (p4[:, 0] + bp), p4[:, 1:] + k[:, None] * beta])


def minkowski_mass(p4: np.ndarray) -> np.ndarray:
    p4 = np.atleast_2d(p4)
    return np.sqrt(np.maximum(p4[:, 0]**2 - np.sum(p4[:, 1:]**2, 1), 0.0))


def inv_mass(p1: np.ndarray, p2: np.ndarray, m1: float = M_MU, m2: float = M_MU) -> np.ndarray:
    """Invariant mass of two particles from their 3-momenta (the last 3 columns are used).

    Uses a cancellation-free form:
      m² = m1² + m2² + 2(E1E2 − |p1||p2|) + 4|p1||p2| sin²(θ12/2),
      E1E2 − |p1||p2| = (m1²|p2|² + m2²|p1|² + m1²m2²)/(E1E2 + |p1||p2|),
      2 sin(θ12/2) = |û1 − û2|.
    This stays accurate for collinear, highly boosted pairs where (E1+E2)² − |p1+p2|²
    loses digits.
    """
    p1, p2 = np.atleast_2d(p1)[:, -3:], np.atleast_2d(p2)[:, -3:]
    a = np.linalg.norm(p1, axis=1)
    b = np.linalg.norm(p2, axis=1)
    E1, E2 = np.sqrt(a**2 + m1**2), np.sqrt(b**2 + m2**2)
    term1 = (m1**2 * b**2 + m2**2 * a**2 + m1**2 * m2**2) / (E1 * E2 + a * b)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = p1 / a[:, None] - p2 / b[:, None]
    chord2 = np.nan_to_num(np.sum(u**2, 1))         # = 4 sin²(θ/2)
    m2_ = m1**2 + m2**2 + 2 * term1 + a * b * chord2
    return np.sqrt(np.maximum(m2_, 0.0))


def pt_eta_phi(p: np.ndarray):
    p = np.atleast_2d(p)[:, -3:]
    pt = np.hypot(p[:, 0], p[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(pt > 0, np.arcsinh(p[:, 2] / np.where(pt > 0, pt, 1.0)), np.copysign(np.inf, p[:, 2]))
    return pt, eta, np.arctan2(p[:, 1], p[:, 0])


def cos_theta_cs(mu_minus: np.ndarray, mu_plus: np.ndarray) -> np.ndarray:
    """Collins–Soper cos θ* of the negative lepton (Collins & Soper, PRD 16, 2219 (1977)).

    cos θ_CS = sign(Q_z) · 2(p1⁺p2⁻ − p1⁻p2⁺) / (Q √(Q² + Q_T²)),  p± = (E ± p_z)/√2,
    with 1 = μ⁻, 2 = μ⁺, Q = p1 + p2. The sign(Q_z) factor is the usual choice for
    pp collisions, where the quark direction is unknown.
    """
    def full(p):
        p = np.atleast_2d(p)
        if p.shape[1] == 4:
            return p
        return np.column_stack([np.sqrt(np.sum(p**2, 1) + M_MU**2), p])
    a, b = full(mu_minus), full(mu_plus)
    Q = a + b
    Q2 = Q[:, 0]**2 - np.sum(Q[:, 1:]**2, 1)
    QT2 = Q[:, 1]**2 + Q[:, 2]**2
    ap, am = (a[:, 0] + a[:, 3]) / np.sqrt(2), (a[:, 0] - a[:, 3]) / np.sqrt(2)
    bp, bm = (b[:, 0] + b[:, 3]) / np.sqrt(2), (b[:, 0] - b[:, 3]) / np.sqrt(2)
    s = np.where(Q[:, 3] >= 0, 1.0, -1.0)
    return s * 2 * (ap * bm - am * bp) / np.sqrt(Q2 * (Q2 + QT2))

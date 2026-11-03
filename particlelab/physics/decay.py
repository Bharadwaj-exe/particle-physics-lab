"""Two-body decay X → μ⁺μ⁻ with frame-dependent angular distributions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import boost, M_MU
from .production import ProductionModel, default_model, sample_parents, SQRT_S


@dataclass
class DecayBatch:
    name: str
    parent: np.ndarray      # (n,4) lab
    mass: np.ndarray        # (n,)
    mu_minus: np.ndarray    # (n,4) lab
    mu_plus: np.ndarray     # (n,4) lab

    def __len__(self):
        return len(self.mass)


def sample_cos_theta(n: int, rng: np.random.Generator, lam: float = 1.0) -> np.ndarray:
    """cos θ drawn from 1 + λ cos²θ (−1 ≤ λ ≤ 1) by rejection."""
    if not -1.0 <= lam <= 1.0:
        raise ValueError("lambda_theta must be in [-1, 1]")
    fmax = 1 + max(lam, 0.0)
    out = np.empty(0)
    while out.size < n:
        c = rng.uniform(-1, 1, 2 * n)
        keep = rng.uniform(0, fmax, 2 * n) < 1 + lam * c**2
        out = np.concatenate([out, c[keep]])
    return out[:n]


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def frame_axes(parent: np.ndarray, frame: str, sqrt_s: float = SQRT_S):
    """Orthonormal axes (x, y, z) in the parent rest frame for the given frame.

    "cs":       Collins–Soper. z bisects the beam and minus-target directions,
                oriented along the parent's longitudinal motion.
    "helicity": z along the parent's lab flight direction.
    """
    n = len(parent)
    beta = parent[:, 1:] / parent[:, :1]
    if frame == "cs":
        Eb = sqrt_s / 2
        b1 = boost(np.tile([Eb, 0, 0, Eb], (n, 1)), -beta)[:, 1:]
        b2 = boost(np.tile([Eb, 0, 0, -Eb], (n, 1)), -beta)[:, 1:]
        u1, u2 = _unit(b1), _unit(b2)
        s = np.where(parent[:, 3] >= 0, 1.0, -1.0)[:, None]
        z = s * _unit(u1 - u2)
        x = -_unit(u1 + u2)
        # parents with pT ≈ 0: u1 + u2 vanishes; any perpendicular axis will do
        bad = ~np.isfinite(x).all(1)
        x[bad] = np.array([1.0, 0, 0])
        x = _unit(x - np.sum(x * z, 1, keepdims=True) * z)
    elif frame == "helicity":
        z = _unit(parent[:, 1:])
        ref = np.where(np.abs(z[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        x = _unit(ref - np.sum(ref * z, 1, keepdims=True) * z)
    else:
        raise ValueError(f"unknown frame {frame}")
    return x, np.cross(z, x), z


def decay_rest(m: np.ndarray, rng: np.random.Generator, axes=None, lam: float = 0.0):
    """μ⁻ and μ⁺ four-momenta in the parent rest frame.

    |p*| = √(m²/4 − m_μ²). The μ⁻ direction has polar angle θ about the frame's z
    axis with density ∝ 1 + λ cos²θ; λ = 0 is isotropic.
    """
    n = m.size
    if np.any(m <= 2 * M_MU):
        raise ValueError("parent mass below the dimuon threshold")
    p = np.sqrt(m**2 / 4 - M_MU**2)
    c = sample_cos_theta(n, rng, lam) if lam != 0 else rng.uniform(-1, 1, n)
    s = np.sqrt(1 - c**2)
    phi = rng.uniform(0, 2 * np.pi, n)
    if axes is None:
        d = np.column_stack([s * np.cos(phi), s * np.sin(phi), c])
    else:
        x, y, z = axes
        d = (s * np.cos(phi))[:, None] * x + (s * np.sin(phi))[:, None] * y + c[:, None] * z
    d *= p[:, None]
    E = (m / 2)[:, None]
    return np.hstack([E, d]), np.hstack([E, -d])


def generate(name: str, n: int, seed=None, model: ProductionModel | None = None) -> DecayBatch:
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    model = model or default_model(name)
    parent, m = sample_parents(name, n, rng, model)
    axes = frame_axes(parent, "cs") if model.angular == "cs" else None
    lam = model.lambda_theta if model.angular != "isotropic" else 0.0
    mum, mup = decay_rest(m, rng, axes, lam)
    beta = parent[:, 1:] / parent[:, :1]
    return DecayBatch(name, parent, m, boost(mum, beta), boost(mup, beta))

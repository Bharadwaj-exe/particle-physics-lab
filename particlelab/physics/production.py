"""Parent production: mass line shapes and (pT, y) spectra.

The (pT, y) spectra are phenomenological shapes, NOT derived from parton
distributions or fitted to the CMS data used for comparison. They give the
qualitative features (soft peak with a power-law tail; rapidity plateau) so that
acceptance and kinematic distributions are not dominated by an arbitrary
exponential. See docs/physics/PRODUCTION_AND_DECAY.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.particles import load

SQRT_S = 7000.0  # GeV, LHC 2011


@dataclass(frozen=True)
class ProductionModel:
    lineshape: str          # "bw" (non-relativistic, fixed width) or "rbw" (relativistic, s-dependent width)
    mass_window: tuple      # generated mass range (GeV)
    pt_p0: float            # dN/dpT ∝ pT (1 + (pT/p0)²)^(−n)
    pt_n: float
    y_flat: float           # dN/dy flat for |y| < y_flat, falls linearly to 0 at y_max
    y_max: float
    angular: str            # "isotropic" or "cs" (1 + λ cos²θ in the Collins–Soper frame)
    lambda_theta: float = 0.0


def default_model(name: str) -> ProductionModel:
    p = load().get(name)
    if name == "Z":
        return ProductionModel("rbw", (60.0, 120.0), 10.0, 3.0, 2.0, 3.6, "cs", 1.0)
    w = max(10 * p.width, 1e-4)
    if name == "J/psi":
        # Quarkonium polarisation measured at the LHC is compatible with zero
        # (CMS, PLB 727 (2013) 381; PRL 110 (2013) 081802), so isotropic decays.
        return ProductionModel("bw", (p.mass - w, p.mass + w), 4.5, 3.5, 3.0, 4.5, "isotropic", 0.0)
    if name == "Upsilon(1S)":
        return ProductionModel("bw", (p.mass - w, p.mass + w), 8.0, 3.5, 3.0, 4.5, "isotropic", 0.0)
    raise KeyError(f"No production model for {name}")


# ---- mass line shapes -------------------------------------------------------

def bw_pdf(m, M, G):
    """Non-relativistic Breit–Wigner (Cauchy) density."""
    return (G / (2 * np.pi)) / ((m - M)**2 + G**2 / 4)


def rbw_pdf_unnorm(m, M, G):
    """Relativistic Breit–Wigner with s-dependent width, as a density in m.

    dσ/ds ∝ s / ((s − M²)² + s² Γ²/M²)  (the convention in which PDG quotes M_Z),
    times the Jacobian ds/dm = 2m. Parton-luminosity effects are not included.
    """
    s = m**2
    return 2 * m * s / ((s - M**2)**2 + s**2 * G**2 / M**2)


def sample_mass(name: str, n: int, rng: np.random.Generator, model: ProductionModel | None = None):
    model = model or default_model(name)
    p = load().get(name)
    lo, hi = model.mass_window
    if model.lineshape == "bw":
        F = lambda m: np.arctan(2 * (m - p.mass) / p.width) / np.pi + 0.5
        u = rng.uniform(F(lo), F(hi), n)
        return p.mass + 0.5 * p.width * np.tan(np.pi * (u - 0.5))
    if model.lineshape == "rbw":
        grid = np.linspace(lo, hi, 200_001)
        pdf = rbw_pdf_unnorm(grid, p.mass, p.width)
        cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
        cdf /= cdf[-1]
        return np.interp(rng.uniform(0, 1, n), cdf, grid)
    raise ValueError(f"unknown lineshape {model.lineshape}")


# ---- kinematic spectra ------------------------------------------------------

def sample_pt(n, p0, nexp, rng):
    """Inverse CDF of dN/dpT ∝ pT (1+(pT/p0)²)^(−n):  F = 1 − (1+pT²/p0²)^(1−n)."""
    u = rng.uniform(0, 1, n)
    return p0 * np.sqrt((1 - u) ** (1 / (1 - nexp)) - 1)


def sample_y(n, y_flat, y_max, rng):
    """Trapezoid: flat for |y| < y_flat, linear fall to zero at |y| = y_max."""
    a_flat, a_tail = y_flat, 0.5 * (y_max - y_flat)
    in_flat = rng.uniform(0, 1, n) < a_flat / (a_flat + a_tail)
    y = np.where(in_flat, rng.uniform(0, y_flat, n),
                 y_flat + (y_max - y_flat) * (1 - np.sqrt(rng.uniform(0, 1, n))))
    return y * rng.choice([-1.0, 1.0], n)


def sample_parents(name: str, n: int, rng: np.random.Generator, model: ProductionModel | None = None):
    model = model or default_model(name)
    m = sample_mass(name, n, rng, model)
    pt = sample_pt(n, model.pt_p0, model.pt_n, rng)
    y = sample_y(n, model.y_flat, model.y_max, rng)
    phi = rng.uniform(-np.pi, np.pi, n)
    mt = np.sqrt(m**2 + pt**2)
    p4 = np.column_stack([mt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y)])
    return p4, m

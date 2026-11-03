"""Signal and background line shapes (unnormalised densities in m).

Each shape is (function, parameter names, defaults builder). Normalisation over
the fit window is done numerically by the fitter, so these need not be normalised.
"""
from __future__ import annotations

import numpy as np
from scipy.special import voigt_profile

from ..physics.production import rbw_pdf_unnorm


def gauss(m, mu, sigma):
    return np.exp(-0.5 * ((m - mu) / sigma) ** 2) / sigma


def double_gauss(m, mu, sigma, frac, ratio):
    return frac * gauss(m, mu, sigma) + (1 - frac) * gauss(m, mu, sigma * ratio)


def crystal_ball(m, mu, sigma, alpha, n):
    """Gaussian core with a power-law low-mass tail (Oreglia/Gaiser, SLAC-R-236 (1980))."""
    t = (m - mu) / sigma
    B = n / alpha - alpha
    # evaluate the tail in log space to avoid overflow; only used where t <= -alpha
    with np.errstate(over="ignore", under="ignore"):
        tail = np.exp(n * np.log(n / alpha) - 0.5 * alpha**2 - n * np.log(np.maximum(B - t, n / alpha)))
    return np.where(t > -alpha, np.exp(-0.5 * t**2), tail) / sigma


def cb_gauss(m, mu, sigma, alpha, n, frac, ratio):
    """Crystal Ball plus a wider Gaussian with a common mean (per-event resolution spread)."""
    cb = crystal_ball(m, mu, sigma, alpha, n)
    cb_norm = _cb_integral(sigma, alpha, n)
    return frac * cb / cb_norm + (1 - frac) * gauss(m, mu, sigma * ratio) / np.sqrt(2 * np.pi)


def _cb_integral(sigma, alpha, n):
    """∫ crystal_ball dm over the real line (n > 1)."""
    from scipy.special import erf
    core = np.sqrt(np.pi / 2) * (1 + erf(alpha / np.sqrt(2)))
    tail = n / (alpha * (n - 1)) * np.exp(-0.5 * alpha**2)
    return core + tail


def voigt_fixed(m, mu, sigma, width):
    """Non-relativistic fixed-width Breit–Wigner ⊗ Gaussian."""
    return voigt_profile(m - mu, sigma, width / 2)


def _convolve_rbw(m, mu, width, kernel_fn, left, right, step):
    """Numerically convolve the relativistic Breit–Wigner with a resolution kernel.

    The kernel is sampled on offsets k·step, k = −K_left … K_right, so offset 0 sits
    exactly on a grid point and the convolution introduces no shift.
    """
    m = np.asarray(m, float)
    kl, kr = int(np.ceil(left / step)), int(np.ceil(right / step))
    offs = step * np.arange(-kl, kr + 1)
    kern = kernel_fn(offs)
    kern = kern / kern.sum()
    lo, hi = m.min() - right - 1.0, m.max() + left + 1.0
    grid = lo + step * np.arange(int(np.ceil((hi - lo) / step)) + 1)
    f = rbw_pdf_unnorm(np.maximum(grid, 1e-6), mu, width)
    # g(m_i) = Σ_k f(m_i − offs_k) K(offs_k); with full convolution index i + kl
    conv = np.convolve(f, kern, mode="full")[kl:kl + grid.size]
    return np.interp(m, grid, conv)


def rbw_gauss(m, mu, sigma, width):
    """Relativistic s-dependent-width Breit–Wigner ⊗ Gaussian."""
    step = min(sigma, width) / 20
    return _convolve_rbw(m, mu, width, lambda x: np.exp(-0.5 * (x / sigma) ** 2), 8 * sigma, 8 * sigma, step)


def rbw_cb(m, mu, sigma, alpha, n, width):
    """Relativistic Breit–Wigner ⊗ Crystal Ball (the CB tail absorbs final-state radiation)."""
    step = min(sigma, width) / 20
    tail = min(max(8 * sigma, 10 * sigma * n / alpha), 15.0)
    return _convolve_rbw(m, mu, width, lambda x: crystal_ball(x, 0.0, sigma, alpha, n), tail, 8 * sigma, step)


def exponential(m, lam, lo):
    return np.exp(-lam * (m - lo))


SIGNALS = {
    "gauss": (gauss, ("mu", "sigma")),
    "double_gauss": (double_gauss, ("mu", "sigma", "frac", "ratio")),
    "crystal_ball": (crystal_ball, ("mu", "sigma", "alpha", "n")),
    "cb_gauss": (cb_gauss, ("mu", "sigma", "alpha", "n", "frac", "ratio")),
    "voigt": (voigt_fixed, ("mu", "sigma")),        # width fixed from the PDG
    "rbw_gauss": (rbw_gauss, ("mu", "sigma")),      # width fixed from the PDG
    "rbw_cb": (rbw_cb, ("mu", "sigma", "alpha", "n")),
}
WIDTH_SHAPES = ("voigt", "rbw_gauss", "rbw_cb")

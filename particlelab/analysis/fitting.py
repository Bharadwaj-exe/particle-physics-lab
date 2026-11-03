"""Extended binned maximum-likelihood fits of dimuon mass peaks.

    −ln L = Σ_i [ μ_i − n_i ln μ_i ],   μ_i = N_s ∫_bin S + N_b ∫_bin B

Shapes are normalised numerically on the fit window, so N_s and N_b are event
counts in the window. Uncertainties come from the inverse of the numerical
Hessian of −ln L at the minimum. Goodness of fit uses the Poisson deviance
(Baker & Cousins, NIM 221 (1984) 437), compared with χ²(ndf).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from ..domain.particles import load
from .shapes import SIGNALS, WIDTH_SHAPES, exponential

WINDOWS = {"J/psi": (2.9, 3.3), "Upsilon(1S)": (9.0, 9.8), "Z": (70.0, 110.0)}
DEFAULT_SIGNAL = {"J/psi": "cb_gauss", "Upsilon(1S)": "double_gauss", "Z": "rbw_gauss"}
_SUB = 5  # sub-points per bin for bin integration


@dataclass
class FitResult:
    name: str
    signal: str
    background: str
    params: dict
    errors: dict
    cov: np.ndarray
    names: list
    nll: float
    deviance: float
    ndf: int
    pvalue: float
    window: tuple
    bins: int
    edges: np.ndarray
    counts: np.ndarray
    converged: bool
    at_bound: list = field(default_factory=list)
    _model: object = None
    extra_peaks: tuple = ()
    optimizer_ok: bool = True      # the minimiser reported success
    hessian_ok: bool = True        # the Hessian at the minimum is positive definite
    optimizer_message: str = ""

    # compatibility / convenience ------------------------------------------------
    @property
    def mass(self): return self.params["mu"]
    @property
    def mass_err(self): return self.errors["mu"]
    @property
    def n_signal(self): return self.params["ns"]
    @property
    def n_signal_err(self): return self.errors["ns"]
    @property
    def n_background(self): return self.params.get("nb", 0.0)
    @property
    def chi2_ndf(self): return self.deviance / max(self.ndf, 1)

    @property
    def sigma(self):
        """Detector-resolution width: Gaussian σ for Z shapes, FWHM/2.355 of the peak otherwise."""
        return self._width()[0]

    @property
    def sigma_err(self):
        return self._width()[1]

    def fwhm(self):
        x = np.array([self.params[k] for k in self.names])
        return _fwhm(self._model.signal_density, x, self.window)

    def _width(self):
        if self.signal in WIDTH_SHAPES or self.signal == "gauss":
            return self.params["sigma"], self.errors["sigma"]
        f = lambda x: _fwhm(self._model.signal_density, x, self.window) / 2.3548
        x0 = np.array([self.params[k] for k in self.names])
        w0 = f(x0)
        grad = np.zeros(len(x0))
        for i in range(len(x0)):
            h = 1e-4 * max(abs(x0[i]), 1e-3)
            xp, xm = x0.copy(), x0.copy(); xp[i] += h; xm[i] -= h
            grad[i] = (f(xp) - f(xm)) / (2 * h)
        return w0, float(np.sqrt(max(grad @ self.cov @ grad, 0.0)))

    def curve(self, m, component="total"):
        """Expected events per bin at masses m (for plotting)."""
        x = np.array([self.params[k] for k in self.names])
        w = (self.window[1] - self.window[0]) / self.bins
        return w * self._model.density(np.asarray(m, float), x, component)

    def expected(self):
        x = np.array([self.params[k] for k in self.names])
        return self._model.expected(x)

    def summary(self) -> dict:
        return {"process": self.name, "signal_model": self.signal, "background_model": self.background,
                "mass": self.mass, "mass_err": self.mass_err, "width": self.sigma, "width_err": self.sigma_err,
                "n_signal": self.n_signal, "n_signal_err": self.n_signal_err, "n_background": self.n_background,
                "deviance": self.deviance, "ndf": self.ndf, "pvalue": self.pvalue, "converged": self.converged,
                "optimizer_ok": self.optimizer_ok, "hessian_ok": self.hessian_ok,
                "at_bound": self.at_bound, "window": list(self.window), "bins": self.bins,
                "n_events": int(self.counts.sum())}


def _fwhm(dens, x, window):
    """Full width at half maximum with linear interpolation of the crossings."""
    m = np.linspace(window[0], window[1], 8001)
    f = dens(m, x)
    half = f.max() / 2
    above = np.flatnonzero(f >= half)
    i, j = above[0], above[-1]
    left = m[i] if i == 0 else m[i - 1] + (half - f[i - 1]) / (f[i] - f[i - 1]) * (m[i] - m[i - 1])
    right = m[j] if j == len(m) - 1 else m[j] + (f[j] - half) / (f[j] - f[j + 1]) * (m[j + 1] - m[j])
    return right - left


class _Model:
    def __init__(self, name, signal, background, window, bins, extra_peaks=()):
        self.lo, self.hi = window
        self.edges = np.linspace(self.lo, self.hi, bins + 1)
        w = self.edges[1] - self.edges[0]
        offs = (np.arange(_SUB) + 0.5) / _SUB * w
        self.sub = (self.edges[:-1, None] + offs[None, :]).ravel()
        self.subw = w / _SUB
        self.bins = bins
        self.norm_grid = np.linspace(self.lo, self.hi, 4001)
        self.width = load().get(name).width
        self.fn, sig_names = SIGNALS[signal]
        self.signal, self.background = signal, background
        self.nextra = len(extra_peaks)
        extra = [n for i in range(self.nextra) for n in (f"mu{i + 2}", f"f{i + 2}")]
        self.names = ["ns", *sig_names, *extra] + (["nb", "lam"] if background == "exp" else [])
        self.nsig = len(sig_names)

    def _sig(self, m, sp):
        if self.signal in WIDTH_SHAPES:
            return self.fn(m, *sp, self.width)
        return self.fn(m, *sp)

    def signal_density(self, m, x):
        sp = x[1:1 + self.nsig]
        return self._sig(m, sp)

    def _one(self, m, sp):
        if self.signal in ("rbw_gauss", "rbw_cb"):
            allm = np.concatenate([m, self.norm_grid])
            v = self._sig(allm, sp)
            s, sg = v[:m.size], v[m.size:]
        else:
            s, sg = self._sig(m, sp), self._sig(self.norm_grid, sp)
        return s / np.trapezoid(sg, self.norm_grid)

    def _normed(self, m, x):
        sp = np.array(x[1:1 + self.nsig], float)
        s = self._one(m, sp)
        # extra peaks share the shape; their width scales with mass (resolution ∝ p)
        for i in range(self.nextra):
            mu_i, f_i = x[1 + self.nsig + 2 * i], x[2 + self.nsig + 2 * i]
            spi = sp.copy()
            spi[1] = sp[1] * mu_i / sp[0]
            spi[0] = mu_i
            s = s + f_i * self._one(m, spi)
        if self.background == "exp":
            lam = x[-1]
            b = exponential(m, lam, self.lo)
            b = b / np.trapezoid(exponential(self.norm_grid, lam, self.lo), self.norm_grid)
        else:
            b = np.zeros_like(m)
        return s, b

    def density(self, m, x, component="total"):
        s, b = self._normed(m, x)
        ns = x[0]
        nb = x[1 + self.nsig + 2 * self.nextra] if self.background == "exp" else 0.0
        return {"signal": ns * s, "background": nb * b, "total": ns * s + nb * b}[component]

    def expected(self, x):
        d = self.density(self.sub, x)
        return d.reshape(self.bins, _SUB).sum(1) * self.subw


def _start(name, signal, background, counts, edges):
    p = load().get(name)
    centres = 0.5 * (edges[1:] + edges[:-1])
    w = edges[1] - edges[0]
    lo, hi = edges[0], edges[-1]
    N = counts.sum()
    ipk = np.argmax(counts)
    mu0 = centres[ipk] if abs(centres[ipk] - p.mass) < 0.05 * p.mass else p.mass
    base = np.median(counts) if background == "exp" else 0.0
    above = counts > (counts[ipk] + base) / 2
    sig0 = max(above.sum() * w / 2.355, w / 2)
    if name == "Z":
        sig0 = max(sig0 / 2, 0.3)
    nb0 = base * len(counts) if background == "exp" else 0.0
    ns0 = max(N - nb0, 1.0)
    span = hi - lo
    init = {"ns": ns0, "mu": mu0, "sigma": sig0, "frac": 0.7, "ratio": 2.0, "alpha": 1.5, "n": 3.0,
            "nb": max(nb0, 1.0), "lam": 0.5 / span}
    bounds = {"ns": (0, 10 * N + 10), "mu": (lo, hi), "sigma": (w / 20, span), "frac": (0.0, 1.0),
              "ratio": (1.0, 20.0), "alpha": (0.2, 10.0), "n": (1.05, 100.0), "nb": (0, 10 * N + 10),
              "lam": (-50.0 / span, 50.0 / span)}
    return init, bounds


def hessian(f, x, scale):
    k = len(x)
    H = np.zeros((k, k))
    h = 1e-4 * np.maximum(np.abs(x), scale)
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                xp, xm = x.copy(), x.copy(); xp[i] += h[i]; xm[i] -= h[i]
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i] ** 2
            else:
                vals = []
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    xx = x.copy(); xx[i] += si * h[i]; xx[j] += sj * h[j]
                    vals.append(f(xx))
                H[i, j] = H[j, i] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h[i] * h[j])
    return H


def fit_counts(counts, edges, name, signal=None, background="exp", init=None, extra_peaks=()):
    """Fit histogram counts. ``extra_peaks`` gives starting masses of additional peaks
    (e.g. Υ(2S), Υ(3S)) that share the main peak's shape, with free masses and yields."""
    signal = signal or DEFAULT_SIGNAL[name]
    bins = len(counts)
    model = _Model(name, signal, background, (edges[0], edges[-1]), bins, tuple(extra_peaks))
    names = model.names
    i0, b0 = _start(name, signal, background, counts, edges)
    for i, mu_i in enumerate(extra_peaks):
        i0[f"mu{i + 2}"], b0[f"mu{i + 2}"] = mu_i, (mu_i - 0.15, mu_i + 0.15)
        i0[f"f{i + 2}"], b0[f"f{i + 2}"] = 0.3, (0.0, 3.0)
    if init:
        i0.update(init)
    x0 = np.array([i0[k] for k in names], float)
    bnds = [b0[k] for k in names]
    x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])
    n = np.asarray(counts, float)

    lo_b = np.array([b[0] for b in bnds]); hi_b = np.array([b[1] for b in bnds])
    BAD = 1e300

    def nll(x):
        # Every minimiser below respects the bounds; this guard only protects the
        # Hessian steps and keeps a bad point from raising inside a minimiser.
        try:
            with np.errstate(all="ignore"):
                mu = np.maximum(model.expected(x), 1e-300)
                v = float(np.sum(mu - n * np.log(mu)))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return BAD
        return v if np.isfinite(v) else BAD

    scale = np.array([max(abs(v), 1e-3) for v in x0])
    fs = lambda z: nll(z * scale)
    sbnds = [(b[0] / s, b[1] / s) for b, s in zip(bnds, scale)]
    lbfgs = dict(method="L-BFGS-B", bounds=sbnds, options={"maxiter": 5000, "ftol": 1e-11, "gtol": 1e-7})
    res = optimize.minimize(fs, x0 / scale, **lbfgs)
    best = res
    # Fallbacks are all bounded: two L-BFGS-B restarts from the best point, then a
    # bounded Nelder–Mead polish followed by a final L-BFGS-B step.
    for _ in range(2):
        if best.success:
            break
        r = optimize.minimize(fs, best.x, **lbfgs)
        if r.fun <= best.fun:
            best = r
    if not best.success:
        nm = optimize.minimize(fs, best.x, method="Nelder-Mead", bounds=sbnds,
                               options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        r = optimize.minimize(fs, nm.x, **lbfgs)
        for cand in (nm, r):
            if cand.fun < best.fun or (cand.fun <= best.fun and cand.success):
                best = cand
    optimizer_ok = bool(best.success and best.fun < BAD)
    message = str(best.message)
    x = np.clip(best.x * scale, lo_b, hi_b)
    H = hessian(nll, x, scale * 1e-2)
    try:
        cov = np.linalg.inv(H)
        posdef = bool(np.all(np.isfinite(H)) and np.all(np.linalg.eigvalsh(H) > 0))
    except np.linalg.LinAlgError:
        cov, posdef = np.full_like(H, np.nan), False
    err = np.sqrt(np.abs(np.diag(cov)))
    at_bound = [k for k, v, b in zip(names, x, bnds) if np.isclose(v, b[0], rtol=1e-6, atol=1e-12) or np.isclose(v, b[1], rtol=1e-6)]
    mu = np.maximum(model.expected(x), 1e-300)
    dev = float(2 * np.sum(mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / mu), 0.0)))
    ndf = int(max(bins - len(names), 1))
    return FitResult(name, signal, background, dict(zip(names, map(float, x))), dict(zip(names, map(float, err))),
                     cov, names, nll(x), dev, ndf, float(stats.chi2.sf(dev, ndf)), (edges[0], edges[-1]), bins,
                     edges, n, bool(posdef and optimizer_ok), at_bound, model, tuple(extra_peaks),
                     optimizer_ok, posdef, message)


def fit_peak(masses, name, bins=80, window=None, background=True, signal=None, init=None,
             extra_peaks=()) -> FitResult:
    lo, hi = window or WINDOWS[name]
    counts, edges = np.histogram(masses, bins=bins, range=(lo, hi))
    return fit_counts(counts, edges, name, signal, "exp" if background else "none", init, extra_peaks)

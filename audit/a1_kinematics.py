"""Audit A1: two-body kinematics, boosts, invariant mass, angular distributions, spectra.

Independent of the production code wherever possible: symbolic derivation (sympy),
4x4 Lorentz matrices, 50-digit arithmetic (mpmath), and an angular analysis that
builds its own rest frame and Collins-Soper axes.
"""
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlelab.physics import kinematics as K                      # noqa: E402
from particlelab.physics import decay as D, production as P            # noqa: E402

OUT = {}
MMU = 0.1056583755

# ---- 1. symbolic two-body decay --------------------------------------------------
M, m1, m2 = sp.symbols("M m1 m2", positive=True)
E1 = (M**2 + m1**2 - m2**2) / (2 * M)
E2 = (M**2 + m2**2 - m1**2) / (2 * M)
lam = M**4 + m1**4 + m2**4 - 2 * M**2 * m1**2 - 2 * M**2 * m2**2 - 2 * m1**2 * m2**2
p = sp.sqrt(lam) / (2 * M)
OUT["sym_energy_conservation"] = sp.simplify(E1 + E2 - M) == 0
OUT["sym_onshell_1"] = sp.simplify(E1**2 - p**2 - m1**2) == 0
OUT["sym_onshell_2"] = sp.simplify(E2**2 - p**2 - m2**2) == 0
OUT["sym_equal_mass_p"] = sp.simplify(p.subs(m2, m1) - sp.sqrt(M**2 / 4 - m1**2)) == 0
# lambda factorises as (M^2-(m1+m2)^2)(M^2-(m1-m2)^2): zero at threshold
OUT["sym_lambda_threshold"] = sp.expand(lam - (M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) == 0

# ---- 2. decay_rest against the derivation -----------------------------------------
rng = np.random.default_rng(11)
for mass in (2 * MMU * (1 + 1e-9), 3.0969, 9.4604, 91.1879):
    mm = np.full(1000, mass)
    a, b = D.decay_rest(mm, rng)
    pexp = np.sqrt(mass**2 / 4 - MMU**2)
    OUT[f"rest_{mass:.6g}"] = dict(
        max_dp_sum=float(np.abs(a[:, 1:] + b[:, 1:]).max()),
        max_dE=float(np.abs(a[:, 0] + b[:, 0] - mass).max()),
        max_dp_rel=float(np.abs(np.linalg.norm(a[:, 1:], axis=1) / max(pexp, 1e-300) - 1).max()) if pexp > 0 else None,
        max_mass_err=float(np.abs(K.minkowski_mass(a) - MMU).max()))
try:
    D.decay_rest(np.array([2 * MMU * 0.999]), rng)
    OUT["below_threshold_raises"] = False
except ValueError:
    OUT["below_threshold_raises"] = True


# ---- 3. boost vs explicit 4x4 matrix --------------------------------------------
def lorentz_matrix(beta):
    b2 = beta @ beta
    g = 1 / np.sqrt(1 - b2)
    L = np.eye(4)
    L[0, 0] = g
    L[0, 1:] = L[1:, 0] = g * beta
    L[1:, 1:] += (g - 1) * np.outer(beta, beta) / b2
    return L


errs = []
for _ in range(2000):
    beta = rng.normal(size=3)
    beta *= rng.uniform(0, 0.999999) / np.linalg.norm(beta)
    p4 = np.r_[0, rng.normal(size=3) * 10]
    p4[0] = np.sqrt(p4[1:] @ p4[1:] + MMU**2)
    ref = lorentz_matrix(beta) @ p4
    errs.append(np.abs(K.boost(p4, beta)[0] - ref).max() / max(abs(ref[0]), 1))
OUT["boost_vs_matrix_max_rel_err"] = float(max(errs))

# ---- 4. invariant mass vs 50-digit reference --------------------------------------
mp.mp.dps = 50


def mass_ref(pa, pb):
    pa = [mp.mpf(float(v)) for v in pa]
    pb = [mp.mpf(float(v)) for v in pb]
    Ea = mp.sqrt(sum(v * v for v in pa) + mp.mpf(MMU)**2)
    Eb = mp.sqrt(sum(v * v for v in pb) + mp.mpf(MMU)**2)
    return mp.sqrt((Ea + Eb)**2 - sum((x + y)**2 for x, y in zip(pa, pb)))


cases = {"random": [], "collinear_1TeV": [], "back_to_back": [], "low_p": []}
for _ in range(300):
    a, b = rng.normal(size=3) * 20, rng.normal(size=3) * 20
    cases["random"].append((a, b))
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    cases["collinear_1TeV"].append((1000 * d, 800 * (d + 1e-7 * rng.normal(size=3))))
    cases["back_to_back"].append((45 * d, -45 * d))
    cases["low_p"].append((0.01 * rng.normal(size=3), 0.01 * rng.normal(size=3)))
for k, v in cases.items():
    A = np.array([x[0] for x in v]); B = np.array([x[1] for x in v])
    got = K.inv_mass(A, B)
    naive = np.sqrt(np.maximum((np.sqrt((A**2).sum(1) + MMU**2) + np.sqrt((B**2).sum(1) + MMU**2))**2 - ((A + B)**2).sum(1), 0))
    ref = np.array([float(mass_ref(a, b)) for a, b in v])
    OUT[f"invmass_{k}"] = dict(stable_max_rel=float(np.max(np.abs(got / ref - 1))),
                               naive_max_rel=float(np.max(np.abs(naive / ref - 1))))

# ---- 5. Lorentz invariance of the generated mass ----------------------------------
ev = D.generate("Z", 20000, seed=5)
OUT["generated_mass_preserved_max_rel"] = float(np.max(np.abs(K.inv_mass(ev.mu_minus, ev.mu_plus) / ev.mass - 1)))
OUT["four_momentum_conservation_max"] = float(np.abs(ev.mu_minus + ev.mu_plus - ev.parent).max())


# ---- 6. angular distributions, independent frame construction -------------------------
def to_rest(p4, parent):
    """Boost p4 into the parent rest frame with an explicit matrix per event."""
    out = np.empty_like(p4)
    for i in range(len(p4)):
        beta = parent[i, 1:] / parent[i, 0]
        out[i] = lorentz_matrix(-beta) @ p4[i]
    return out


def cs_cos_independent(ev, sqrt_s=7000.0):
    """cos theta_CS of the mu- : CS z axis bisects beam1 and -beam2 in the dimuon rest frame."""
    n = len(ev)
    mu = to_rest(ev.mu_minus, ev.parent)
    b1 = to_rest(np.tile([sqrt_s / 2, 0, 0, sqrt_s / 2], (n, 1)), ev.parent)[:, 1:]
    b2 = to_rest(np.tile([sqrt_s / 2, 0, 0, -sqrt_s / 2], (n, 1)), ev.parent)[:, 1:]
    z = b1 / np.linalg.norm(b1, axis=1, keepdims=True) - b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z *= np.sign(ev.parent[:, 3])[:, None]
    return np.sum(mu[:, 1:] * z, 1) / np.linalg.norm(mu[:, 1:], axis=1)


def helicity_cos(ev):
    mu = to_rest(ev.mu_minus, ev.parent)
    z = ev.parent[:, 1:] / np.linalg.norm(ev.parent[:, 1:], axis=1, keepdims=True)
    return np.sum(mu[:, 1:] * z, 1) / np.linalg.norm(mu[:, 1:], axis=1)


def fit_lambda(c):
    """Moments: <cos^2> = (1/3 + lam/5)/(1 + lam/3) -> solve for lam; A_FB from <cos>."""
    m2 = np.mean(c**2)
    lam = (m2 - 1 / 3) / (1 / 5 - m2 / 3)
    # error by bootstrap
    bs = []
    r = np.random.default_rng(1)
    for _ in range(200):
        s = c[r.integers(0, len(c), len(c))]
        m = np.mean(s**2)
        bs.append((m - 1 / 3) / (1 / 5 - m / 3))
    return float(lam), float(np.std(bs)), float(np.mean(c)), float(np.std(c) / np.sqrt(len(c)))


ang = {}
for name in ("Z", "J/psi", "Upsilon(1S)"):
    e = D.generate(name, 20000, seed=21)
    for frame, fn in (("CS", cs_cos_independent), ("helicity", helicity_cos)):
        lam_, lerr, mean_c, merr = fit_lambda(fn(e))
        ang[f"{name} {frame}"] = dict(lambda_theta=lam_, err=lerr, mean_cos=mean_c, mean_cos_err=merr)
# cross-check the production CS formula against the independent construction
e = D.generate("Z", 3000, seed=3)
ang["Z CS formula vs independent max diff"] = float(np.abs(K.cos_theta_cs(e.mu_minus, e.mu_plus) - cs_cos_independent(e)).max())
OUT["angular"] = ang

# ---- 7. spectra and line shapes ----------------------------------------------------
r = np.random.default_rng(8)
mdl = P.default_model("J/psi")
pt = P.sample_pt(200000, mdl.pt_p0, mdl.pt_n, r)
cdf = lambda x: 1 - (1 + (x / mdl.pt_p0)**2)**(1 - mdl.pt_n)
OUT["pt_spectrum_KS_p"] = float(stats.kstest(pt, cdf).pvalue)
y = np.abs(P.sample_y(200000, mdl.y_flat, mdl.y_max, r))
a_f, a_t = mdl.y_flat, 0.5 * (mdl.y_max - mdl.y_flat)
def ycdf(x):
    x = np.asarray(x)
    L = mdl.y_max - mdl.y_flat
    t = np.clip(x - mdl.y_flat, 0, L)
    return np.where(x < mdl.y_flat, x, mdl.y_flat + t - t**2 / (2 * L)) / (a_f + a_t)
OUT["y_spectrum_KS_p"] = float(stats.kstest(y, ycdf).pvalue)

# Z mass sampling vs direct numerical integration of the s-dependent BW (independent quad)
from scipy import integrate
Mz, Gz = 91.1879, 2.4955
f = lambda m: 2 * m * m**2 / ((m**2 - Mz**2)**2 + m**4 * Gz**2 / Mz**2)
norm = integrate.quad(f, 60, 120, points=[Mz], limit=400)[0]
mean_ref = integrate.quad(lambda m: m * f(m), 60, 120, points=[Mz], limit=400)[0] / norm
ms = P.sample_mass("Z", 400000, r)
OUT["Z_mass_sampling"] = dict(sample_mean=float(ms.mean()), ref_mean=mean_ref,
                              pull=float((ms.mean() - mean_ref) / (ms.std() / np.sqrt(ms.size))),
                              KS_p=float(stats.kstest(ms, lambda x: np.array([integrate.quad(f, 60, xi, points=[Mz] if xi > Mz else None, limit=400)[0] for xi in np.atleast_1d(x)]) / norm).pvalue) if False else None)
# peak of s-dependent BW in m vs M: differs from M by O(Gamma^2/M)
grid = np.linspace(89, 93, 400001)
OUT["Z_rbw_mode_minus_M_MeV"] = float(1000 * (grid[np.argmax(f(grid))] - Mz))
# fixed-width (PDG "running width" vs fixed) conversion: M_fixed = M_run - Gamma^2/(2M) approx
OUT["Z_fixed_vs_running_shift_MeV_theory"] = float(1000 * Gz**2 / (2 * Mz))

Path(__file__).with_name("results").mkdir(exist_ok=True)
Path(__file__).with_name("results").joinpath("a1_kinematics.json").write_text(json.dumps(OUT, indent=2, default=str))
print(json.dumps(OUT, indent=2, default=str))

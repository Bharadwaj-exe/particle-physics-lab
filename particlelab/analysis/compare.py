"""Data–simulation comparison with a common selection.

For each resonance: fit data (signal + exponential background) and simulation
(signal only) with the same signal family, then compare peak position, width
and yield, and compare muon pT and η shapes using sideband-subtracted data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..domain.particles import load
from ..physics.kinematics import pt_eta_phi
from .fitting import FitResult, fit_peak, WINDOWS, DEFAULT_SIGNAL
from .pipeline import SimResult

# Data fit settings. Υ: the 2S and 3S states are fitted together with the 1S (free
# masses and yields, shared resolution shape scaled with mass). Z: Breit–Wigner ⊗
# Crystal Ball, whose tail absorbs final-state radiation present in data.
DATA_FIT = {
    "J/psi": dict(window=(2.9, 3.3), bins=80, signal="cb_gauss"),
    "Upsilon(1S)": dict(window=(8.5, 11.5), bins=120, signal="double_gauss", extra_peaks=(10.02, 10.35)),
    "Z": dict(window=(70.0, 110.0), bins=80, signal="rbw_cb"),
}
# Simulation has no radiative tail, so Crystal Ball tail parameters are unconstrained
# (65 % of toy fits had a singular Hessian); a double Gaussian is well defined.
SIM_FIT = {
    "J/psi": dict(window=(2.9, 3.3), bins=80, signal="double_gauss"),
    "Upsilon(1S)": dict(window=(9.0, 9.8), bins=80, signal="double_gauss"),
    "Z": dict(window=(70.0, 110.0), bins=80, signal="rbw_gauss"),
}


@dataclass
class KinematicComparison:
    variable: str
    edges: np.ndarray
    data: np.ndarray           # sideband-subtracted, normalised to unit area
    data_err: np.ndarray
    sim: np.ndarray
    sim_err: np.ndarray
    chi2: float
    ndf: int


@dataclass
class Comparison:
    name: str
    data_fit: FitResult
    sim_fit: FitResult
    pdg_mass: float
    n_data_signal_region: int
    n_sideband: int
    sideband_scale: float
    kinematics: list = field(default_factory=list)

    def summary(self) -> dict:
        d, s = self.data_fit, self.sim_fit
        return {
            "process": self.name, "pdg_mass": self.pdg_mass,
            "data_mass": d.mass, "data_mass_err": d.mass_err, "sim_mass": s.mass, "sim_mass_err": s.mass_err,
            "data_minus_pdg_MeV": 1000 * (d.mass - self.pdg_mass), "sim_minus_pdg_MeV": 1000 * (s.mass - self.pdg_mass),
            "data_fwhm_MeV": 1000 * d.fwhm(), "sim_fwhm_MeV": 1000 * s.fwhm(),
            "fwhm_ratio_sim_over_data": s.fwhm() / d.fwhm(),
            "data_resolution_param_MeV": 1000 * d.sigma, "sim_resolution_param_MeV": 1000 * s.sigma,
            "data_signal": d.n_signal, "data_signal_err": d.n_signal_err, "data_background": d.n_background,
            "data_fit_pvalue": d.pvalue, "sim_fit_pvalue": s.pvalue,
            "data_signal_model": d.signal, "sim_signal_model": s.signal,
            "kinematics_chi2": {k.variable: [k.chi2, k.ndf] for k in self.kinematics},
        }


def _shape(values, weights, edges):
    h, _ = np.histogram(values, edges, weights=weights)
    h2, _ = np.histogram(values, edges, weights=weights**2)
    total = h.sum()
    return h / total, np.sqrt(h2) / total


def compare(name: str, data_events, sim: SimResult) -> Comparison:
    """``data_events``: DataFrame of selected CMS events (pt/eta recomputed columns)."""
    m = data_events["M_recomputed"].to_numpy()
    dfit = fit_peak(m, name, **DATA_FIT[name])
    sfit = fit_peak(sim.reco_mass, name, background=False, **SIM_FIT[name])

    # signal region ±1 FWHM around the fitted peak; sidebands 2–3 FWHM away
    mu, w = dfit.mass, dfit.fwhm()
    lo, hi = dfit.window
    sr = (mu - w, mu + w)
    sbs = [(max(lo, mu - 3 * w), mu - 2 * w), (mu + 2 * w, min(hi, mu + 3 * w))]
    if name == "Upsilon(1S)":
        sbs = [(max(lo, mu - 3 * w), mu - 2 * w)]       # upper side is occupied by Υ(2S)
    grid = np.linspace(lo, hi, 20001)
    bkg = dfit.curve(grid, "background")
    integ = lambda a, b: np.trapezoid(bkg[(grid >= a) & (grid <= b)], grid[(grid >= a) & (grid <= b)])
    sb_int = sum(integ(a, b) for a, b in sbs if b > a)
    scale = integ(*sr) / sb_int if sb_int > 0 else 0.0
    in_sr = (m > sr[0]) & (m < sr[1])
    in_sb = np.zeros_like(in_sr)
    for a, b in sbs:
        in_sb |= (m > a) & (m < b)

    ev = data_events
    pt_d = np.concatenate([ev["pt1_re"].to_numpy(), ev["pt2_re"].to_numpy()])
    eta_d = np.concatenate([ev["eta1_re"].to_numpy(), ev["eta2_re"].to_numpy()])
    wsr = np.concatenate([in_sr, in_sr]).astype(float)
    wsb = np.concatenate([in_sb, in_sb]).astype(float) * scale
    wd = wsr - wsb
    pt_s, eta_s = sim.muon_kinematics()
    sim_sr = np.concatenate([(sim.reco_mass > mu - sfit.fwhm()) & (sim.reco_mass < mu + sfit.fwhm())] * 2)
    pmax = {"J/psi": 30, "Upsilon(1S)": 40, "Z": 100}[name]
    kin = []
    for var, dv, sv, edges in (("muon pT [GeV]", pt_d, pt_s, np.linspace(3, pmax, 28)),
                               ("muon eta", eta_d, eta_s, np.linspace(-2.4, 2.4, 25))):
        hd, ed = _shape(dv[wd != 0], wd[wd != 0], edges)
        # data variance: SR counts + scale² × SB counts
        v_sr, _ = np.histogram(dv[wsr > 0], edges)
        v_sb, _ = np.histogram(dv[wsb > 0], edges)
        tot, _ = np.histogram(dv[wd != 0], edges, weights=wd[wd != 0])
        ed = np.sqrt(v_sr + scale**2 * v_sb) / max(tot.sum(), 1e-12)
        hs, es = _shape(sv[sim_sr], np.ones(sim_sr.sum()), edges)
        ok = (ed > 0) | (es > 0)
        chi2 = float(np.sum(((hd - hs)[ok]) ** 2 / (ed[ok] ** 2 + es[ok] ** 2)))
        kin.append(KinematicComparison(var, edges, hd, ed, hs, es, chi2, int(ok.sum() - 1)))
    return Comparison(name, dfit, sfit, load().get(name).mass, int(in_sr.sum()), int(in_sb.sum()), scale, kin)

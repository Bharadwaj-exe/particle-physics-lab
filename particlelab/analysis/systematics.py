"""Systematic uncertainties of the fitted data masses, and masses by detector region.

Each variation refits the data with one fit choice changed from the nominal
settings (compare.DATA_FIT):

* window  – narrower and wider fit windows (same bin width),
* binning – half and double the number of bins,
* model   – alternative signal shapes.

A variation counts only if the minimiser converged and the fit describes the data
(p > 0.01). A positive-definite Hessian is not required: the CB-type shapes have
nearly flat directions in their tail parameters (correlations up to 0.96), which
make the error matrix unreliable without affecting the fitted mass. A model that
fails the goodness-of-fit test is reported but does not define an uncertainty. The systematic for each category is the largest accepted shift from
the nominal mass; the categories are added in quadrature. Window variations change
the sample, so their shifts include a statistical part; the estimate is
conservative.

Masses by region use the nominal settings on events where both muons are central
(|η| < 0.9), one is central, or neither is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..domain.particles import load
from .fitting import fit_peak

P_MIN = 0.01
BARREL_ETA = 0.9

VARIATIONS = {
    "J/psi": {
        "window": [(2.95, 3.25), (2.85, 3.35)],
        "model": ["double_gauss", "crystal_ball"],
    },
    "Upsilon(1S)": {
        "window": [(8.7, 11.2), (8.4, 11.8)],
        "model": ["cb_gauss"],
    },
    "Z": {
        "window": [(75.0, 105.0), (80.0, 100.0)],
        "model": ["rbw_gauss"],
    },
}


@dataclass
class Variation:
    category: str
    label: str
    mass: float
    mass_err: float
    pvalue: float
    converged: bool
    shift_MeV: float
    accepted: bool
    note: str = ""


@dataclass
class MassSystematics:
    name: str
    nominal_mass: float
    nominal_err: float
    variations: list = field(default_factory=list)

    def by_category(self) -> dict:
        out = {}
        for v in self.variations:
            if v.accepted:
                out[v.category] = max(out.get(v.category, 0.0), abs(v.shift_MeV))
        return out

    @property
    def total_MeV(self) -> float:
        return float(np.sqrt(sum(s**2 for s in self.by_category().values())))

    def summary(self) -> dict:
        return {"process": self.name, "nominal_mass": self.nominal_mass, "stat_err_MeV": 1000 * self.nominal_err,
                "syst_by_category_MeV": self.by_category(), "syst_total_MeV": self.total_MeV,
                "variations": [v.__dict__ for v in self.variations]}


def _try_fit(m, name, kw):
    try:
        return fit_peak(m, name, **kw), ""
    except Exception as e:   # a failed variation is reported, never silently dropped
        return None, f"fit failed: {e}"


def mass_systematics(name: str, masses: np.ndarray, nominal: dict) -> MassSystematics:
    """``nominal``: keyword arguments of the nominal fit_peak call (compare.DATA_FIT[name])."""
    f0 = fit_peak(masses, name, **nominal)
    out = MassSystematics(name, f0.mass, f0.mass_err)
    lo, hi = nominal["window"]
    width = (hi - lo) / nominal["bins"]
    trials = []
    for w in VARIATIONS[name]["window"]:
        trials.append(("window", f"window {w[0]:g}–{w[1]:g} GeV", dict(nominal, window=w, bins=int(round((w[1] - w[0]) / width)))))
    for factor in (0.5, 2.0):
        b = int(round(nominal["bins"] * factor))
        trials.append(("binning", f"{b} bins", dict(nominal, bins=b)))
    for s in VARIATIONS[name]["model"]:
        trials.append(("model", f"signal {s}", dict(nominal, signal=s)))
    for cat, label, kw in trials:
        f, note = _try_fit(masses, name, kw)
        if f is None:
            out.variations.append(Variation(cat, label, np.nan, np.nan, np.nan, False, np.nan, False, note))
            continue
        ok = bool(f.optimizer_ok and f.pvalue > P_MIN)
        if not ok:
            note = "minimiser did not converge" if not f.optimizer_ok else f"rejected: p = {f.pvalue:.1e} < {P_MIN}"
        elif not f.hessian_ok:
            note = "Hessian not positive definite (mass used, its error not)"
        out.variations.append(Variation(cat, label, f.mass, f.mass_err, f.pvalue, bool(f.converged),
                                        1000 * (f.mass - f0.mass), ok, note))
    return out


def masses_by_region(name: str, events, nominal: dict) -> dict:
    """Nominal fit in three detector regions; ``events`` has eta1_re, eta2_re, M_recomputed."""
    e1 = np.abs(events["eta1_re"].to_numpy()); e2 = np.abs(events["eta2_re"].to_numpy())
    m = events["M_recomputed"].to_numpy()
    c1, c2 = e1 < BARREL_ETA, e2 < BARREL_ETA
    regions = {"both central (|η| < 0.9)": c1 & c2, "one central": c1 ^ c2, "both forward (|η| > 0.9)": ~c1 & ~c2}
    pdg = load().get(name).mass
    out = {}
    for label, mask in regions.items():
        f, note = _try_fit(m[mask], name, nominal)
        if f is None:
            out[label] = {"n_events": int(mask.sum()), "note": note}
            continue
        out[label] = {"n_events_window": int(f.counts.sum()), "mass": f.mass, "mass_err": f.mass_err,
                      "offset_from_pdg_pct": 100 * (f.mass / pdg - 1), "pvalue": f.pvalue,
                      "converged": f.converged, "optimizer_ok": f.optimizer_ok, "hessian_ok": f.hessian_ok}
    return out

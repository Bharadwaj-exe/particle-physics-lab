"""Speed and accuracy against a published fast-simulation baseline.

    python benchmarks/compare_baselines.py [--quick]

1. Stage profile: where the time goes (generation, hits, track fits, selection).
2. Interactive latency: change a setting -> new fitted mass plot, at the GUI's default size.
3. Accuracy against CMS 2011 data: fitted resolution width of each peak, for the whole
   sample and for central and forward muon pairs, compared with
   - this hit-level simulation, and
   - a Delphes-style parametric simulation: the muon momentum resolution formula of
     the Delphes CMS card (cards/delphes_card_CMS.tcl, MuonMomentumSmearing),
     applied with Delphes' log-normal pT smearing to the SAME generated muons.
   Both are fitted with the same code, so only the detector model differs.
4. Generator-level pre-selection: how much time is saved by skipping events whose true
   muons cannot pass the cuts, and how many finally selected events it would lose.

Writes outputs/baseline_comparison.json. Times are single-process on the machine in
the environment record.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from particlelab.analysis import simulate, fit_peak  # noqa: E402
from particlelab.analysis.compare import DATA_FIT, SIM_FIT  # noqa: E402
from particlelab.data import cms  # noqa: E402
from particlelab.detector import cms_inspired, simulate_hits, fit_tracks  # noqa: E402
from particlelab.physics import generate, inv_mass, pt_eta_phi  # noqa: E402
from particlelab.provenance import environment, save_json  # noqa: E402

PROCS = ("J/psi", "Upsilon(1S)", "Z")
REGIONS = {  # (name, condition on the larger and smaller |eta| of the two muons)
    "all": lambda hi, lo: np.ones_like(hi, bool),
    "central (both |eta| < 0.9)": lambda hi, lo: hi < 0.9,
    "forward (both |eta| > 1.2)": lambda hi, lo: lo > 1.2,
}
PRE_PT, PRE_ETA = 0.8, 0.1    # pre-selection margins: true pT > 0.8 pT_min, |eta| < eta_max + 0.1


def delphes_cms_resolution(pt, eta):
    """Relative pT resolution of muons in the Delphes CMS card."""
    a = np.abs(eta)
    r = np.where(a <= 0.5, np.sqrt(0.01**2 + pt**2 * 1.0e-4**2),
        np.where(a <= 1.5, np.sqrt(0.015**2 + pt**2 * 1.5e-4**2),
        np.where(a <= 2.5, np.sqrt(0.025**2 + pt**2 * 3.5e-4**2), 0.0)))
    return np.where(pt > 0.1, np.minimum(r, 1.0), 0.0)


def delphes_smear(p, rng):
    """Delphes MomentumSmearing: pT drawn from a log-normal of mean pT and width res*pT; eta, phi kept."""
    p = p[:, -3:]
    pt, eta, _ = pt_eta_phi(p)
    sig = delphes_cms_resolution(pt, eta) * pt
    b = np.sqrt(np.log1p((sig / pt) ** 2))
    a = np.log(pt) - 0.5 * b**2
    pt_new = np.exp(a + b * rng.standard_normal(len(pt)))
    return p * (pt_new / pt)[:, None]


def delphes_tracking_eff(pt, eta):
    a = np.abs(eta)
    return np.where(a <= 1.5, np.where(pt > 1, 0.99, np.where(pt > 0.1, 0.75, 0)),
           np.where(a <= 2.5, np.where(pt > 1, 0.98, np.where(pt > 0.1, 0.70, 0)), 0))


def passes(p, det):
    pt, eta, _ = pt_eta_phi(p)
    return (pt > det.pt_min) & (np.abs(eta) < det.eta_max)


def eta_pair(pm, pp):
    _, e1, _ = pt_eta_phi(pm)
    _, e2, _ = pt_eta_phi(pp)
    return np.maximum(np.abs(e1), np.abs(e2)), np.minimum(np.abs(e1), np.abs(e2))


def width(fit):
    return dict(fwhm_MeV=1000 * fit.fwhm(), sigma_MeV=1000 * fit.sigma, sigma_err_MeV=1000 * fit.sigma_err,
                mass=fit.mass, pvalue=fit.pvalue, converged=bool(fit.converged))


def safe_fit(m, name, **kw):
    try:
        if len(m) < 200:
            return None
        return width(fit_peak(m, name, **kw))
    except Exception as e:   # a sparse region can fail to fit: record it, do not stop
        return {"error": str(e)[:120]}


# ---- 1 and 4: stage profile and pre-selection --------------------------------------------

def stage_profile(name, n, det, seed):
    rng = np.random.default_rng(seed)
    t = {}
    t0 = time.perf_counter()
    ev = generate(name, n, seed=rng)
    t["generate"] = time.perf_counter() - t0
    zv = rng.normal(0, det.vertex_sigma_z, n)
    t0 = time.perf_counter()
    hm = simulate_hits(ev.mu_minus, -np.ones(n), det, rng, zv)
    hp = simulate_hits(ev.mu_plus, np.ones(n), det, rng, zv)
    t["hits"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    fm, fp = fit_tracks(hm, det), fit_tracks(hp, det)
    t["track fits"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok = (hm.n_detected >= det.min_hits) & (hp.n_detected >= det.min_hits) & fm.ok & fp.ok & (fm.q * fp.q < 0)
    sel = ok & passes(np.nan_to_num(fm.p), det) & passes(np.nan_to_num(fp.p), det)
    inv_mass(fm.p[sel], fp.p[sel])
    t["selection + mass"] = time.perf_counter() - t0
    total = sum(t.values())

    # pre-selection on the true muons, judged on the same events
    def pre(p):
        pt, eta, _ = pt_eta_phi(p)
        return (pt > PRE_PT * det.pt_min) & (np.abs(eta) < det.eta_max + PRE_ETA)
    keep = pre(ev.mu_minus) & pre(ev.mu_plus)
    lost = int(np.sum(sel & ~keep))
    m = int(keep.sum())
    rng2 = np.random.default_rng(seed + 1)
    t0 = time.perf_counter()
    h1 = simulate_hits(ev.mu_minus[keep], -np.ones(m), det, rng2, zv[keep])
    h2 = simulate_hits(ev.mu_plus[keep], np.ones(m), det, rng2, zv[keep])
    fit_tracks(h1, det), fit_tracks(h2, det)
    t_pre = time.perf_counter() - t0 + t["generate"]
    return {
        "events": n, "seconds": t, "total_seconds": total, "events_per_s": n / total,
        "share": {k: v / total for k, v in t.items()}, "selected": int(sel.sum()),
        "preselection": {"kept_fraction": m / n, "selected_events_lost": lost, "selected_events": int(sel.sum()),
                         "seconds": t_pre, "speedup": total / t_pre},
    }


# ---- 2: interactive latency -------------------------------------------------------------

def latency(name, n, det, seed):
    t0 = time.perf_counter()
    sim = simulate(name, n, det, seed=seed)
    t1 = time.perf_counter()
    if sim.reco_mass.size > 50:
        fit_peak(sim.reco_mass, name, background=False, **SIM_FIT[name])
    t2 = time.perf_counter()
    return {"events": n, "selected": int(sim.reco_mass.size), "simulate_s": t1 - t0, "fit_s": t2 - t1,
            "total_s": t2 - t0}


# ---- 3: accuracy against data -----------------------------------------------------------

def accuracy(name, n, det, seed, data):
    out = {}
    # this simulation (official pipeline)
    t0 = time.perf_counter()
    sim = simulate(name, n, det, seed=seed)
    t_ours = time.perf_counter() - t0
    # Delphes-style, on muons from the same generator
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    ev = generate(name, n, seed=rng)
    pm, pp = delphes_smear(ev.mu_minus, rng), delphes_smear(ev.mu_plus, rng)
    pt1, e1, _ = pt_eta_phi(ev.mu_minus); pt2, e2, _ = pt_eta_phi(ev.mu_plus)
    tracked = (rng.random(n) < delphes_tracking_eff(pt1, e1)) & (rng.random(n) < delphes_tracking_eff(pt2, e2))
    sel = tracked & passes(pm, det) & passes(pp, det)
    m_del = inv_mass(pm[sel], pp[sel])
    t_del = time.perf_counter() - t0
    # the full Delphes CMS muon chain also requires pT > 10 GeV for muon identification
    pt_m, _, _ = pt_eta_phi(pm); pt_p, _, _ = pt_eta_phi(pp)
    out["delphes_muon_id_keeps_fraction_of_selected"] = float(np.mean((pt_m[sel] > 10) & (pt_p[sel] > 10)))
    out["seconds"] = {"this simulation": t_ours, "delphes-style": t_del}
    out["events_per_s"] = {"this simulation": n / t_ours, "delphes-style": n / t_del}

    d = data.events
    m_data = d["M_recomputed"].to_numpy()
    a1, a2 = d["eta1_re"].abs().to_numpy(), d["eta2_re"].abs().to_numpy()
    d_hi, d_lo = np.maximum(a1, a2), np.minimum(a1, a2)
    s_hi, s_lo = eta_pair(sim.p_minus, sim.p_plus)
    x_hi, x_lo = eta_pair(pm[sel], pp[sel])
    regions = {}
    for reg, cond in REGIONS.items():
        r = {"data": safe_fit(m_data[cond(d_hi, d_lo)], name, **DATA_FIT[name]),
             "this simulation": safe_fit(sim.reco_mass[cond(s_hi, s_lo)], name, background=False, **SIM_FIT[name]),
             "delphes-style": safe_fit(m_del[cond(x_hi, x_lo)], name, background=False, **SIM_FIT[name])}
        dd = r["data"]
        for k in ("this simulation", "delphes-style"):
            s = r[k]
            if dd and s and "sigma_MeV" in dd and "sigma_MeV" in s:
                ratio = s["sigma_MeV"] / dd["sigma_MeV"]
                s["sigma_ratio_to_data"] = ratio
                s["ratio_err_from_data"] = ratio * dd["sigma_err_MeV"] / dd["sigma_MeV"]
                s["pulls_from_data"] = (ratio - 1) / s["ratio_err_from_data"]
        regions[reg] = r
    out["regions"] = regions
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer events (smoke test)")
    a = ap.parse_args()
    scale = 0.1 if a.quick else 1.0
    det = cms_inspired()
    n_acc = {"J/psi": 1_000_000, "Upsilon(1S)": 200_000, "Z": 100_000}   # as configs/default.json
    out = {"environment": environment(), "delphes_card": "delphes/cards/delphes_card_CMS.tcl (master), "
           "MuonMomentumSmearing + MuonTrackingEfficiency", "preselection_margins": [PRE_PT, PRE_ETA],
           "profile": {}, "latency": {}, "accuracy": {}}
    data = cms.load(selection=cms.Selection(pt_min=det.pt_min, eta_max=det.eta_max), check_file=False) \
        if cms.available() else None
    for i, name in enumerate(PROCS):
        seed = 20261102 + i
        out["profile"][name] = p = stage_profile(name, int(100_000 * scale), det, seed)
        print(name, "profile", round(p["events_per_s"]), "ev/s", {k: round(v, 2) for k, v in p["share"].items()},
              "pre-selection x%.1f, lost %d" % (p["preselection"]["speedup"], p["preselection"]["selected_events_lost"]),
              flush=True)
        out["latency"][name] = L = latency(name, int(100_000 * scale), det, seed)
        print(name, "latency", {k: round(v, 2) if isinstance(v, float) else v for k, v in L.items()}, flush=True)
        if data is not None:
            out["accuracy"][name] = A = accuracy(name, int(n_acc[name] * scale), det, seed, data)
            for reg, r in A["regions"].items():
                print(name, reg, {k: (round(v["sigma_MeV"], 1), round(v.get("sigma_ratio_to_data", 1), 3))
                                  if v and "sigma_MeV" in v else v for k, v in r.items()}, flush=True)
    path = ROOT / ("outputs_quick" if a.quick else "outputs") / "baseline_comparison.json"
    path.parent.mkdir(exist_ok=True)
    save_json(out, path)
    print("wrote", path)


if __name__ == "__main__":
    main()

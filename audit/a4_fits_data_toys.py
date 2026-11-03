"""Audit A4: fit-model dependence of the width comparison, model-independent widths,
CMS-data mass offsets by region, pT-column inconsistency, kinematic discrepancies,
and independent toy / subsample validation of the mass uncertainties.

The simulated samples are regenerated with the seeds used by run_analysis.py
(seed 20261102 + i) and cached in results/sim_cache.npz.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlelab.analysis import simulate                                     # noqa: E402
from particlelab.analysis.compare import DATA_FIT, SIM_FIT                     # noqa: E402
from particlelab.analysis.fitting import fit_peak                              # noqa: E402
from particlelab.data import cms                                               # noqa: E402
from particlelab.provenance import RunConfig                                   # noqa: E402

RES = Path(__file__).with_name("results")
OUT = {}
t_start = time.time()
cfg = RunConfig.load(Path(__file__).resolve().parents[1] / "configs" / "default.json")
det = cfg.build_detector()
NAMES = ("J/psi", "Upsilon(1S)", "Z")

# ---- simulated samples ---------------------------------------------------------------
cache = RES / "sim_cache.npz"
if cache.exists():
    z = np.load(cache)
    sim = {n: dict(reco=z[f"{i}_reco"], true=z[f"{i}_true"], pm=z[f"{i}_pm"], pp=z[f"{i}_pp"]) for i, n in enumerate(NAMES)}
else:
    sim, store = {}, {}
    for i, n in enumerate(NAMES):
        r = simulate(n, cfg.events[n], det, seed=cfg.seed + i)
        sim[n] = dict(reco=r.reco_mass, true=r.true_mass, pm=r.p_minus, pp=r.p_plus)
        for k, v in sim[n].items():
            store[f"{i}_{k}"] = v
    np.savez_compressed(cache, **store)
print("sim ready", time.time() - t_start, flush=True)

ds = cms.load(selection=cms.Selection(pt_min=cfg.pt_min, eta_max=cfg.eta_max))
ev = ds.events
m_data = ev["M_recomputed"].to_numpy()


def summ(f):
    return dict(model=f.signal, mass=f.mass, mass_err=f.mass_err, fwhm_MeV=1000 * f.fwhm(), sigma_prop_MeV=1000 * f.sigma,
                pvalue=f.pvalue, converged=f.converged, at_bound=f.at_bound)


# ---- 1. model dependence of masses and FWHM ----------------------------------------------
alt_data = {"J/psi": ["cb_gauss", "double_gauss"], "Upsilon(1S)": ["double_gauss", "cb_gauss"], "Z": ["rbw_cb", "rbw_gauss"]}
alt_sim = {"J/psi": ["double_gauss", "cb_gauss"], "Upsilon(1S)": ["double_gauss", "cb_gauss"], "Z": ["rbw_gauss", "rbw_cb"]}
md = {}
for n in NAMES:
    md[n] = {"data": {}, "sim": {}}
    for s in alt_data[n]:
        kw = dict(DATA_FIT[n]); kw["signal"] = s
        try:
            md[n]["data"][s] = summ(fit_peak(m_data, n, **kw))
        except Exception as e:
            md[n]["data"][s] = f"failed: {e}"
    for s in alt_sim[n]:
        kw = dict(SIM_FIT[n]); kw["signal"] = s
        try:
            md[n]["sim"][s] = summ(fit_peak(sim[n]["reco"], n, background=False, **kw))
        except Exception as e:
            md[n]["sim"][s] = f"failed: {e}"
    # every data-model / sim-model combination of the FWHM ratio
    md[n]["fwhm_ratio_sim_over_data"] = {f"{a}|{b}": md[n]["sim"][a]["fwhm_MeV"] / md[n]["data"][b]["fwhm_MeV"]
                                         for a in md[n]["sim"] for b in md[n]["data"]
                                         if isinstance(md[n]["sim"][a], dict) and isinstance(md[n]["data"][b], dict)}
OUT["model_dependence"] = md
print("model dependence done", time.time() - t_start, flush=True)

# correlation matrix of the published data fits
corr = {}
for n in NAMES:
    f = fit_peak(m_data, n, **DATA_FIT[n])
    sd = np.sqrt(np.diag(f.cov))
    C = f.cov / np.outer(sd, sd)
    big = {f"{f.names[i]}-{f.names[j]}": float(C[i, j]) for i in range(len(f.names)) for j in range(i + 1, len(f.names)) if abs(C[i, j]) > 0.7}
    corr[n] = dict(params=f.names, strong_correlations=big, at_bound=f.at_bound,
                   max_abs_offdiag=float(np.max(np.abs(C - np.eye(len(C))))))
OUT["data_fit_correlations"] = corr


# ---- 2. model-independent widths ----------------------------------------------------------
def emp_widths(x, lo, hi, bins=400):
    """FWHM of a lightly smoothed histogram and the half-width of the central 68.3 %."""
    x = x[(x > lo) & (x < hi)]
    h, e = np.histogram(x, bins, (lo, hi))
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0)**2); k /= k.sum()
    hs = np.convolve(h, k, mode="same"); c = 0.5 * (e[1:] + e[:-1])
    half = hs.max() / 2; above = np.flatnonzero(hs >= half)
    i, j = above[0], above[-1]
    left = np.interp(half, [hs[i - 1], hs[i]], [c[i - 1], c[i]])
    right = np.interp(half, [hs[j + 1], hs[j]], [c[j + 1], c[j]])
    q = np.percentile(x, [15.865, 84.135])
    return dict(fwhm_MeV=1000 * (right - left), central68_half_MeV=1000 * 0.5 * (q[1] - q[0]), n=int(x.size))


def subtracted_data_widths(m, name, bins=400):
    """Sideband-subtract an exponential background fitted to the sidebands only."""
    mu = {"J/psi": 3.093, "Upsilon(1S)": 9.444, "Z": 90.95}[name]
    half = {"J/psi": 0.10, "Upsilon(1S)": 0.30, "Z": 10.0}[name]
    lo, hi = mu - 2 * half, mu + 2 * half
    if name == "Upsilon(1S)":
        lo, hi = 8.8, 9.75      # upper side touches the 2S
    h, e = np.histogram(m, bins, (lo, hi)); c = 0.5 * (e[1:] + e[:-1])
    sb = (np.abs(c - mu) > half) & (c < (9.75 if name == "Upsilon(1S)" else hi))
    if name == "Upsilon(1S)":
        sb = c < mu - half
    a, b = np.polyfit(c[sb], np.log(np.maximum(h[sb], 0.5)), 1)
    sig = h - np.exp(a * c + b)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0)**2); k /= k.sum()
    hs = np.convolve(sig, k, mode="same")
    hm = hs.max() / 2; above = np.flatnonzero(hs >= hm)
    i, j = above[0], above[-1]
    left = np.interp(hm, [hs[i - 1], hs[i]], [c[i - 1], c[i]]); right = np.interp(hm, [hs[j + 1], hs[j]], [c[j + 1], c[j]])
    return dict(fwhm_MeV=1000 * (right - left), window=[lo, hi])


wid = {}
for n in NAMES:
    lo, hi = SIM_FIT[n]["window"]
    w_sim = emp_widths(sim[n]["reco"], lo, hi)
    w_res = 1000 * float(0.5 * np.subtract(*np.percentile(sim[n]["reco"] - sim[n]["true"], [84.135, 15.865])))
    w_dat = subtracted_data_widths(m_data, n)
    wid[n] = dict(sim=w_sim, sim_resolution_central68_MeV=w_res, data_subtracted=w_dat,
                  empirical_fwhm_ratio_sim_over_data=w_sim["fwhm_MeV"] / w_dat["fwhm_MeV"])
OUT["empirical_widths"] = wid

# ---- 3. data mass offsets by region / muon type / pT --------------------------------------
eta1, eta2 = ev["eta1_re"].abs().to_numpy(), ev["eta2_re"].abs().to_numpy()
pt1, pt2 = ev["pt1_re"].to_numpy(), ev["pt2_re"].to_numpy()
gg = ((ev["type1"] == "G") & (ev["type2"] == "G")).to_numpy()
cats = {"both |eta|<0.9": (eta1 < 0.9) & (eta2 < 0.9),
        "both 0.9<|eta|<2.4 ": (eta1 > 0.9) & (eta2 > 0.9),
        "mixed barrel/forward": ((eta1 < 0.9) ^ (eta2 < 0.9)),
        "both global": gg, "one tracker-only": ~gg}
reg = {}
for n in NAMES:
    reg[n] = {}
    for cname, mask in cats.items():
        kw = dict(DATA_FIT[n])
        if n == "Upsilon(1S)":
            kw["window"], kw["bins"] = (8.5, 11.5), 120
        try:
            f = fit_peak(m_data[mask], n, **kw)
            reg[n][cname] = dict(n_events_window=int(f.counts.sum()), mass=f.mass, mass_err=f.mass_err,
                                 rel_offset_from_pdg_pct=100 * (f.mass / {"J/psi": 3.0969, "Upsilon(1S)": 9.4604, "Z": 91.1879}[n] - 1),
                                 converged=f.converged)
        except Exception as e:
            reg[n][cname] = f"failed: {e}"
OUT["data_mass_by_category"] = reg
# Z window / model sensitivity
zs = {}
for w in ((70.0, 110.0), (80.0, 100.0), (75.0, 105.0)):
    for s in ("rbw_cb", "rbw_gauss"):
        f = fit_peak(m_data, "Z", window=w, bins=int(2 * (w[1] - w[0])), signal=s)
        zs[f"{s} {w}"] = dict(mass=f.mass, mass_err=f.mass_err, pvalue=f.pvalue)
OUT["Z_data_mass_window_model"] = zs
print("data categories done", time.time() - t_start, flush=True)

# ---- 4. pT-column inconsistency -------------------------------------------------------------
raw = cms.load_raw()
pc = {}
for i in (1, 2):
    pt_re = np.hypot(raw[f"px{i}"], raw[f"py{i}"])
    d = raw[f"pt{i}"] - pt_re
    bad = np.abs(d) > 2e-3 + 1e-3 * pt_re
    rel = (d / pt_re)[bad]
    other = np.hypot(raw[f"px{3 - i}"], raw[f"py{3 - i}"])
    pc[f"muon{i}"] = dict(
        n_bad=int(bad.sum()), frac=float(bad.mean()),
        rel_diff_percentiles={str(p): float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), np.percentile(rel, (1, 5, 25, 50, 75, 95, 99)))},
        frac_published_larger=float(np.mean(rel > 0)),
        bad_frac_by_type={t: float(bad[raw[f"type{i}"] == t].mean()) for t in ("G", "T")},
        bad_frac_by_abseta={k: float(bad[m].mean()) for k, m in {
            "<0.9": np.abs(raw[f"eta{i}"]) < 0.9, "0.9-1.6": (np.abs(raw[f"eta{i}"]) >= 0.9) & (np.abs(raw[f"eta{i}"]) < 1.6),
            ">1.6": np.abs(raw[f"eta{i}"]) >= 1.6}.items()},
        bad_frac_by_pt={k: float(bad[m].mean()) for k, m in {
            "<5": pt_re < 5, "5-20": (pt_re >= 5) & (pt_re < 20), ">20": pt_re >= 20}.items()},
        matches_other_muon_pt=int(np.sum(bad & (np.abs(raw[f"pt{i}"] - other) < 2e-3 + 1e-3 * other))),
        E_consistent_with_components_frac=float(np.mean(np.abs(np.sqrt(np.maximum(raw[f"E{i}"]**2 - pt_re**2 - raw[f"pz{i}"]**2, 0)) - 0.10566) < 0.01)),
        phi_consistent_frac_in_bad=float(np.mean((np.abs(np.angle(np.exp(1j * (np.arctan2(raw[f"py{i}"], raw[f"px{i}"]) - raw[f"phi{i}"])))) < 2e-3)[bad])))
both_bad = None
# downstream: selection with published pt instead of recomputed
v, _, _ = cms.validate(raw)
sel_pub = (v.pt1 > 3) & (v.pt2 > 3) & (v.eta1.abs() < 2.4) & (v.eta2.abs() < 2.4)
sel_re = (v.pt1_re > 3) & (v.pt2_re > 3) & (v.eta1_re.abs() < 2.4) & (v.eta2_re.abs() < 2.4)
pc["selection_published_vs_recomputed"] = dict(n_published=int(sel_pub.sum()), n_recomputed=int(sel_re.sum()),
                                               n_only_published=int((sel_pub & ~sel_re).sum()), n_only_recomputed=int((~sel_pub & sel_re).sum()))
mp_ = v["M_recomputed"].to_numpy()
for n in NAMES:
    a = fit_peak(mp_[sel_pub.to_numpy()], n, **DATA_FIT[n]); b = fit_peak(mp_[sel_re.to_numpy()], n, **DATA_FIT[n])
    pc[f"{n}_mass_published_sel_minus_recomputed_sel_MeV"] = 1000 * (a.mass - b.mass)
    pc[f"{n}_mass_err_MeV"] = 1000 * b.mass_err
# mass from published (pt, eta, phi) vs from components, for bad rows
def m_from(pt1, eta1, phi1, pt2, eta2, phi2):
    p1 = np.column_stack([pt1 * np.cos(phi1), pt1 * np.sin(phi1), pt1 * np.sinh(eta1)])
    p2 = np.column_stack([pt2 * np.cos(phi2), pt2 * np.sin(phi2), pt2 * np.sinh(eta2)])
    E1 = np.sqrt((p1**2).sum(1) + 0.10566**2); E2 = np.sqrt((p2**2).sum(1) + 0.10566**2)
    return np.sqrt(np.maximum((E1 + E2)**2 - ((p1 + p2)**2).sum(1), 0))
mpub = m_from(raw.pt1, raw.eta1, raw.phi1, raw.pt2, raw.eta2, raw.phi2)
bad_any = (np.abs(raw.pt1 - np.hypot(raw.px1, raw.py1)) > 2e-3 + 1e-3 * np.hypot(raw.px1, raw.py1)) | \
          (np.abs(raw.pt2 - np.hypot(raw.px2, raw.py2)) > 2e-3 + 1e-3 * np.hypot(raw.px2, raw.py2))
pc["published_M_agrees_with_components_in_bad_rows_frac"] = float(np.mean((np.abs(raw.M - v_m) < 2e-3 + 1e-3 * raw.M)[bad_any])) if (v_m := cms.inv_mass(raw[["px1", "py1", "pz1"]].to_numpy(), raw[["px2", "py2", "pz2"]].to_numpy())) is not None else None
pc["published_M_agrees_with_pt_eta_phi_in_bad_rows_frac"] = float(np.mean((np.abs(raw.M - mpub) < 2e-3 + 1e-3 * raw.M)[bad_any]))
pc["rows_with_any_bad_pt"] = int(bad_any.sum())
dup = raw.duplicated(["Run", "Event"]).sum()
pc["duplicate_run_event_pairs"] = int(dup)
pc["runs"] = [int(raw.Run.min()), int(raw.Run.max())]
pc["dtypes"] = {k: str(t) for k, t in raw.dtypes.items()}
OUT["cms_pt_inconsistency"] = pc

# ---- 5. kinematic discrepancies (quantified) -----------------------------------------------
kin = {}
for n in NAMES:
    lo, hi = {"J/psi": (3.0, 3.2), "Upsilon(1S)": (9.3, 9.6), "Z": (81, 101)}[n]
    md_ = (m_data > lo) & (m_data < hi)
    ms_ = (sim[n]["reco"] > lo) & (sim[n]["reco"] < hi)
    ptd = np.r_[pt1[md_], pt2[md_]]; etd = np.r_[ev["eta1_re"].to_numpy()[md_], ev["eta2_re"].to_numpy()[md_]]
    pts = np.r_[np.hypot(*sim[n]["pm"][ms_, :2].T), np.hypot(*sim[n]["pp"][ms_, :2].T)]
    ets = np.r_[np.arcsinh(sim[n]["pm"][ms_, 2] / np.hypot(*sim[n]["pm"][ms_, :2].T)), np.arcsinh(sim[n]["pp"][ms_, 2] / np.hypot(*sim[n]["pp"][ms_, :2].T))]
    kin[n] = dict(window=[lo, hi], note="data not background-subtracted here",
                  data_mean_pt=float(ptd.mean()), sim_mean_pt=float(pts.mean()),
                  data_median_pt=float(np.median(ptd)), sim_median_pt=float(np.median(pts)),
                  data_frac_abseta_gt_1p2=float(np.mean(np.abs(etd) > 1.2)), sim_frac_abseta_gt_1p2=float(np.mean(np.abs(ets) > 1.2)),
                  ks_pt=float(stats.ks_2samp(ptd, pts).statistic), ks_eta=float(stats.ks_2samp(etd, ets).statistic))
OUT["kinematics"] = kin
print("kinematics done", time.time() - t_start, flush=True)

# ---- 6. toy and subsample validation ------------------------------------------------------
toy = {}
# 6a. statistical precision of the published pull widths
for n, w in (("J/psi", 0.9456), ("Upsilon(1S)", 0.9343), ("Z", 0.9279)):
    nt = 98 if n == "Z" else 100
    se = w / np.sqrt(2 * (nt - 1))
    toy[f"published_{n}"] = dict(width=w, width_se=float(se), deviation_from_1_in_se=float((1 - w) / se))


# 6b. independent truth (not the fitted model): unbinned sampling from known shapes
def run_toys(sampler, name, signal, mu_true, n_toys, window, seed):
    rng = np.random.default_rng(seed); pulls, fails, errs = [], 0, []
    for _ in range(n_toys):
        x = sampler(rng)
        f = fit_peak(x, name, window=window, bins=80, background=False, signal=signal)
        if not f.converged or not np.isfinite(f.mass_err) or f.mass_err <= 0:
            fails += 1; continue
        pulls.append((f.mass - mu_true) / f.mass_err); errs.append(f.mass_err)
    p = np.array(pulls)
    return dict(n=int(p.size), failed=fails, pull_mean=float(p.mean()), pull_mean_se=float(p.std(ddof=1) / np.sqrt(p.size)),
                pull_width=float(p.std(ddof=1)), pull_width_se=float(p.std(ddof=1) / np.sqrt(2 * (p.size - 1))),
                mean_err=float(np.mean(errs)))


Mj = 3.0973
def jpsi_sampler(rng, n=12184):
    core = rng.random(n) < 0.7
    return np.where(core, rng.normal(Mj, 0.018, n), rng.normal(Mj, 0.035, n))
toy["independent_jpsi_double_gauss"] = run_toys(jpsi_sampler, "J/psi", "double_gauss", Mj, 300, (2.9, 3.3), 1)

Mz, Gz = 91.1879, 2.4955
grid = np.linspace(60, 125, 130001)
pdf = 2 * grid**3 / ((grid**2 - Mz**2)**2 + grid**4 * Gz**2 / Mz**2)
cdf = integrate.cumulative_trapezoid(pdf, grid, initial=0); cdf /= cdf[-1]
def z_sampler(rng, n=56948):
    m = np.interp(rng.random(int(n * 1.1)), cdf, grid) + rng.normal(0, 1.2, int(n * 1.1))
    m = m[(m > 70) & (m < 110)]
    return m[:n]
toy["independent_Z_rbw_gauss"] = run_toys(z_sampler, "Z", "rbw_gauss", Mz, 150, (70.0, 110.0), 2)
print("independent toys done", time.time() - t_start, flush=True)

# 6c. real simulated shape: disjoint subsamples, spread of fitted mass vs quoted error
sub = {}
for n, k in (("J/psi", 6), ("Upsilon(1S)", 30), ("Z", 20)):
    x = np.random.default_rng(3).permutation(sim[n]["reco"])
    parts = np.array_split(x, k)
    full = fit_peak(sim[n]["reco"], n, background=False, **SIM_FIT[n])
    ms, es = [], []
    for part in parts:
        f = fit_peak(part, n, background=False, **SIM_FIT[n])
        if f.converged and np.isfinite(f.mass_err):
            ms.append(f.mass); es.append(f.mass_err)
    ms, es = np.array(ms), np.array(es)
    ratio = ms.std(ddof=1) / es.mean()
    sub[n] = dict(n_subsamples=int(ms.size), events_each=int(len(parts[0])), spread_over_mean_error=float(ratio),
                  ratio_se=float(ratio / np.sqrt(2 * (ms.size - 1))),
                  mean_fit_minus_full_fit_MeV=float(1000 * (ms.mean() - full.mass)),
                  se_MeV=float(1000 * ms.std(ddof=1) / np.sqrt(ms.size)))
toy["simulated_subsamples"] = sub
OUT["toys"] = toy
OUT["runtime_s"] = time.time() - t_start
(RES / "a4_fits_data_toys.json").write_text(json.dumps(OUT, indent=2, default=str))
print(json.dumps(OUT, indent=2, default=str))

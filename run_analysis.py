"""Full analysis: simulation, CMS data comparison, detector studies, fit validation.

    python run_analysis.py                          # configs/default.json
    python run_analysis.py --config configs/quick.json

Writes figures (PNG + PDF), tables (JSON, CSV), a provenance record and a
Markdown report to the configured output directory.
"""
from __future__ import annotations

import argparse
import sys
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particlelab.provenance import RunConfig, run_record, save_json
from particlelab.domain.particles import load as load_particles, RESONANCES
from particlelab.detector import uniform_barrel, gluckstern, simulate_hits, fit_tracks
from particlelab.detector.simulation import helix_at
from particlelab.analysis import simulate, toy_study
from particlelab.analysis.compare import compare, SIM_FIT, DATA_FIT
from particlelab.analysis.systematics import mass_systematics, masses_by_region
from particlelab.analysis.studies import resolution_curve, pt_resolution
from particlelab.analysis.fitting import fit_peak
from particlelab.data import cms
from particlelab.physics import generate
from particlelab.units import C_TRACK

ROOT = Path(__file__).resolve().parent
LABEL = {k: load_particles().get(k).label for k in RESONANCES}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _safe_stdout():
    """Redirected output on Windows uses the ANSI code page, which cannot encode
    characters such as μ; replace them instead of crashing at the end of a run."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def savefig(fig, outdir, name, pdf=False):
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", dpi=140)
    if pdf:
        fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def mass_comparison_figure(comps, figdir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for j, c in enumerate(comps):
        ax, rax = axes[0, j], axes[1, j]
        d = c.data_fit
        centres = 0.5 * (d.edges[1:] + d.edges[:-1])
        ax.errorbar(centres, d.counts, np.sqrt(d.counts), fmt="k.", ms=4, label="CMS 2011 data (DoubleMu)")
        fine = np.linspace(*d.window, 1500)
        ax.plot(fine, d.curve(fine), "C3-", lw=1.5, label=f"fit: {d.signal} + exp")
        ax.plot(fine, d.curve(fine, "background"), "C3--", lw=1, label="fitted background")
        # simulation shape scaled to the fitted data signal (main peak only)
        s = c.sim_fit
        hs, _ = np.histogram([], d.edges)
        simm = c._sim_masses
        hs, _ = np.histogram(simm, d.edges)
        n_in = np.sum((simm > d.window[0]) & (simm < d.window[1]))
        scale = d.n_signal / max(n_in, 1)
        bkg_bins = d.curve(centres, "background")
        ax.step(centres, hs * scale + bkg_bins, where="mid", color="C0", lw=1.5,
                label="simulation (scaled to data yield) + data bkg")
        ax.set_title(f"{LABEL[c.name]} → μ⁺μ⁻:  data M = {d.mass:.4f} ± {d.mass_err:.4f} GeV,  "
                     f"sim M = {s.mass:.4f} GeV", fontsize=9)
        ax.set_ylabel(f"events / {1000 * (d.edges[1] - d.edges[0]):.0f} MeV")
        ax.legend(fontsize=7)
        pull = (d.counts - d.expected()) / np.sqrt(np.maximum(d.expected(), 1))
        rax.bar(centres, pull, width=d.edges[1] - d.edges[0], color="0.5")
        rax.axhline(0, color="k", lw=0.8)
        rax.set_ylim(-4, 4)
        rax.set_ylabel("(data−fit)/σ")
        rax.set_xlabel("m(μμ) [GeV]")
    savefig(fig, figdir, "mass_comparison", pdf=True)


def kinematics_figure(comps, figdir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, c in enumerate(comps):
        for i, k in enumerate(c.kinematics):
            ax = axes[i, j]
            ctr = 0.5 * (k.edges[1:] + k.edges[:-1])
            ax.errorbar(ctr, k.data, k.data_err, fmt="k.", label="data (sideband-subtracted)")
            ax.step(ctr, k.sim, where="mid", color="C0", label="simulation")
            ax.set(xlabel=k.variable, ylabel="fraction of muons",
                   title=f"{LABEL[c.name]}: χ²/ndf = {k.chi2:.0f}/{k.ndf}")
            if "pT" in k.variable:
                ax.set_yscale("log")
            ax.legend(fontsize=7)
    savefig(fig, figdir, "kinematics_comparison", pdf=True)


def spectrum_figure(data_all, figdir):
    m = data_all["M_recomputed"].to_numpy()
    edges = np.logspace(np.log10(0.3), np.log10(300), 400)
    h, _ = np.histogram(m, edges)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.step(0.5 * (edges[1:] + edges[:-1]), h / np.diff(edges), where="mid", color="k", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="m(μμ) [GeV]", ylabel="events / GeV",
           title="CMS 2011 open data, Dimuon_DoubleMu (record 545), both muons pT > 3 GeV, |η| < 2.4")
    for mm, lab in [(0.78, "ρ/ω"), (1.02, "φ"), (3.097, "J/ψ"), (3.686, "ψ(2S)"), (9.46, "Υ(nS)"), (91.19, "Z")]:
        ax.annotate(lab, (mm, h[np.searchsorted(edges, mm) - 1] / np.diff(edges)[np.searchsorted(edges, mm) - 1]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    savefig(fig, figdir, "dimuon_spectrum")


def resolution_figures(det, figdir, rng, n=8000):
    pts = np.array([3, 5, 10, 20, 50, 100, 200.0])
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for eta, col in [(0.0, "C0"), (1.0, "C1"), (2.0, "C2")]:
        full = resolution_curve(det, pts, eta, n, rng)
        noms = resolution_curve(det.with_(multiple_scattering=False), pts, eta, n, rng)
        res[f"eta={eta}"] = {"pt": pts, "robust_sigma_full": full[:, 1], "robust_sigma_no_ms": noms[:, 1],
                             "efficiency": full[:, 2]}
        ax[0].plot(pts, 100 * full[:, 1], "o-", color=col, label=f"|η| = {eta}: full")
        ax[0].plot(pts, 100 * noms[:, 1], "s--", color=col, mfc="none", label=f"|η| = {eta}: no multiple scattering")
    ax[0].set(xscale="log", yscale="log", xlabel="pT [GeV]", ylabel="σ(pT)/pT [%] (central 68%)",
              title="CMS-inspired tracker: momentum resolution")
    ax[0].legend(fontsize=7)
    uni = uniform_barrel()
    ures = resolution_curve(uni, pts, 0.0, n, rng)
    ax[1].plot(pts, 100 * ures[:, 0], "o", label="simulation (rms)")
    xx = np.logspace(np.log10(2), np.log10(250), 100)
    ax[1].plot(xx, 100 * gluckstern(xx, uni), "-", label="Gluckstern formula")
    ax[1].set(xscale="log", yscale="log", xlabel="pT [GeV]", ylabel="σ(pT)/pT [%]",
              title="Reference uniform barrel (no material): fitter validation")
    ax[1].legend()
    res["uniform_reference"] = {"pt": pts, "rms": ures[:, 0], "gluckstern": gluckstern(pts, uni)}
    savefig(fig, figdir, "resolution_vs_pt", pdf=True)

    # parameter scans at η = 0
    scans = {}
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for pt, mk in [(10.0, "o-"), (100.0, "s-")]:
        Bs = [1.0, 2.0, 3.0, 3.8, 5.0, 7.0]
        v = [pt_resolution(det.with_(B=B), pt, 0.0, n, rng)[1] for B in Bs]
        axs[0, 0].plot(Bs, 100 * np.array(v), mk, label=f"pT = {pt:.0f} GeV")
        scans.setdefault("B", {})[pt] = (Bs, v)
        fs = [0.5, 1, 2, 4]
        v = [pt_resolution(det.scaled_resolution(f), pt, 0.0, n, rng)[1] for f in fs]
        axs[0, 1].plot(fs, 100 * np.array(v), mk, label=f"pT = {pt:.0f} GeV")
        scans.setdefault("hit_resolution_scale", {})[pt] = (fs, v)
        ms = [0.0, 0.5, 1, 2, 4]
        v = [pt_resolution(det.scaled_material(f) if f > 0 else det.with_(multiple_scattering=False), pt, 0.0, n, rng)[1] for f in ms]
        axs[1, 0].plot(ms, 100 * np.array(v), mk, label=f"pT = {pt:.0f} GeV")
        scans.setdefault("material_scale", {})[pt] = (ms, v)
    Ns = [5, 8, 10, 15, 20, 30]
    v = [pt_resolution(uniform_barrel(n_layers=N), 50.0, 0.0, n, rng)[0] for N in Ns]
    axs[1, 1].plot(Ns, 100 * np.array(v), "o", label="simulation, uniform barrel, pT = 50 GeV")
    axs[1, 1].plot(Ns, [100 * gluckstern(50.0, uniform_barrel(n_layers=N)) for N in Ns], "-", label="Gluckstern")
    scans["n_layers_uniform"] = (Ns, v)
    axs[0, 0].set(xlabel="B [T]", ylabel="σ(pT)/pT [%]", title="Field strength (η = 0)")
    axs[0, 1].set(xlabel="hit-resolution scale factor", ylabel="σ(pT)/pT [%]", title="Hit resolution (η = 0)")
    axs[1, 0].set(xlabel="material scale factor (0 = no scattering)", ylabel="σ(pT)/pT [%]", title="Material (η = 0)")
    axs[1, 1].set(xlabel="number of layers", ylabel="σ(pT)/pT [%]", title="Number of layers (reference model)")
    for a in axs.ravel():
        a.legend(fontsize=8)
    savefig(fig, figdir, "resolution_scans", pdf=True)
    return res, scans


def cms_reference_figures(det, figdir, rng, n=8000):
    """Model against the digitised CMS references (JINST 9 (2014) P10009): material
    budget X0(η) (Fig. 2) and single-muon σ(pT)/pT versus η (Fig. 14). The CMS tracks
    are unconstrained, so the beam-spot point is switched off for this comparison."""
    import json as _json
    from particlelab.detector import material_along
    ref_x0 = _json.loads((ROOT / "references" / "cms_tracker_material_x0.json").read_text(encoding="utf-8"))
    ref_pt = _json.loads((ROOT / "references" / "cms_muon_pt_resolution.json").read_text(encoding="utf-8"))["pt_GeV"]
    eta = np.array([b["eta"] for b in ref_x0["bins"]])
    cms_x0 = np.array([b["cumulative"]["tec"] for b in ref_x0["bins"]])
    by_sub = [material_along(det, e) for e in eta]
    model_x0 = np.array([sum(d.values()) for d in by_sub])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bottom = np.zeros_like(eta)
    for k in ("beam pipe", "pixel", "TIB/TID", "TOB", "TEC"):
        v = np.array([d[k] for d in by_sub])
        ax[0].bar(eta, v, 0.1, bottom=bottom, label=f"model: {k}", alpha=0.8)
        bottom += v
    ax[0].step(eta, cms_x0, where="mid", color="k", lw=2, label="CMS (tracker, without support tube)")
    ax[0].set(xlabel="|η|", ylabel="x/X0", title="Material budget: model vs CMS")
    ax[0].legend(fontsize=7)
    det0 = det.with_(beamspot_constraint=0.0)
    etas = np.round(np.arange(0.05, 2.5, 0.2), 2)
    comp = {"material": {"eta": eta, "cms": cms_x0, "model": model_x0}, "resolution": {}}
    for (pt, col) in ((1, "k"), (10, "C0"), (100, "C3")):
        model = np.array([100 * pt_resolution(det0, float(pt), float(e), n, rng)[1] for e in etas])
        ref = ref_pt[str(pt)]
        re = np.array([float(k) for k in ref]); rv = np.array(list(ref.values()))
        ax[1].plot(re, rv, "o", color=col, mfc="none", label=f"CMS, pT = {pt} GeV")
        ax[1].plot(etas, model, "-", color=col, label=f"model, pT = {pt} GeV")
        comp["resolution"][str(pt)] = {"eta": etas, "model_pct": model,
                                       "cms_pct": [ref.get(f"{e:.2f}") for e in etas]}
    ax[1].set(yscale="log", xlabel="|η|", ylabel="σ(pT)/pT [%] (68 % half-width)",
              title="Single-muon resolution: model (no beam-spot point) vs CMS")
    ax[1].legend(fontsize=7, ncol=2)
    savefig(fig, figdir, "cms_reference_comparison", pdf=True)
    return comp


def event_display(det, figdir, rng):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.6]})
    t = np.linspace(0, 2 * np.pi, 300)
    for b in det.barrels:
        a1.plot(b.radius * np.cos(t), b.radius * np.sin(t), color="0.8", lw=0.6)
        for sgn in (1, -1):
            a2.plot([-b.half_length, b.half_length], [sgn * b.radius] * 2, color="0.75", lw=0.8)
    for d in det.disks:
        for sz in (1, -1):
            for sr in (1, -1):
                a2.plot([sz * d.z] * 2, [sr * d.r_min, sr * d.r_max], color="0.75", lw=0.8)
    cols = {"J/psi": "C0", "Upsilon(1S)": "C2", "Z": "C3"}
    for name in RESONANCES:
        ev = generate(name, 60, seed=rng)
        for i in range(60):
            mus = [(ev.mu_minus[i], -1), (ev.mu_plus[i], 1)]
            ok = all(np.hypot(p[1], p[2]) > 3 and abs(np.arcsinh(p[3] / np.hypot(p[1], p[2]))) < 2.4 for p, _ in mus)
            if not ok:
                continue
            for p, q in mus:
                h = simulate_hits(p[None, 1:], np.array([float(q)]), det, rng)
                pt = np.hypot(p[1], p[2]); R = pt / (C_TRACK * det.B)
                s_end = np.nanmax(np.where(h.crossed[0], h.s[0], np.nan))
                ss = np.linspace(0, s_end, 200)
                x, y, z = helix_at(ss, R, q, np.arctan2(p[2], p[1]), p[3] / pt)
                a1.plot(x, y, color=cols[name], lw=1)
                sgn = np.sign(y[-1]) or 1
                a2.plot(z, sgn * np.hypot(x, y), color=cols[name], lw=1)
                dmask = h.detected[0]
                a1.plot(h.x[0][dmask], h.y[0][dmask], "k.", ms=3)
                a2.plot(h.z[0][dmask], sgn * np.hypot(h.x[0][dmask], h.y[0][dmask]), "k.", ms=3)
            a1.plot([], [], color=cols[name], label=LABEL[name])
            break
    a1.set(aspect="equal", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xlabel="x [m]", ylabel="y [m]",
           title="Transverse view (B = 3.8 T along z)")
    a2.set(xlim=(-3, 3), ylim=(-1.2, 1.2), xlabel="z [m]", ylabel="±r [m]",
           title="Longitudinal view: barrel layers and endcap disks (CMS-inspired)")
    a1.legend(fontsize=8)
    savefig(fig, figdir, "event_display")


def toys_figure(toys, figdir):
    fig, axes = plt.subplots(1, len(toys), figsize=(5 * len(toys), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, t) in zip(axes, toys.items()):
        ax.hist(t.pulls, bins=20, range=(-4, 4), density=True, color="0.6", label=f"{len(t.pulls)} toys")
        xx = np.linspace(-4, 4, 200)
        ax.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "C3-", label="N(0,1)")
        ax.set(xlabel="pull (μ_fit − μ_true)/σ_fit", title=f"{LABEL[name]}: mean {t.pull_mean:+.2f} ± {t.pull_mean_err:.2f}, "
                                                         f"width {t.pull_std:.2f}")
        ax.legend(fontsize=8)
    savefig(fig, figdir, "fit_validation_pulls")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    args = ap.parse_args()
    _safe_stdout()
    cfg = RunConfig.load(args.config)
    out = ROOT / cfg.output_dir
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    det = cfg.build_detector()
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    # data
    if not cms.available():
        log("downloading CMS data …")
        cms.fetch()
    sel = cms.Selection(pt_min=cfg.pt_min, eta_max=cfg.eta_max)
    data = cms.load(selection=sel)
    log("data:", dict(data.cutflow))
    save_json(data.report.to_dict(), out / "data_validation.json")
    save_json(dict(data.cutflow), out / "data_cutflow.json")
    if len(data.rejected):
        data.rejected.to_csv(out / "data_rejected_rows.csv", index=False)
    spectrum_figure(data.events, figdir)

    # simulation + comparison
    comps, sims, results = [], {}, {}
    for i, name in enumerate(RESONANCES):
        log(f"simulating {name}: {cfg.events[name]} events")
        ts = time.time()
        sim = simulate(name, cfg.events[name], det, seed=cfg.seed + i)
        log(f"  {sim.reco_mass.size} selected ({sim.acceptance:.2%}) in {time.time() - ts:.0f} s")
        sims[name] = sim
        c = compare(name, data.events, sim)
        c._sim_masses = sim.reco_mass
        comps.append(c)
        results[name] = {**c.summary(), "sim_cutflow": sim.cutflow, "sim_acceptance": sim.acceptance,
                         "sim_seed": sim.seed, "sim_events_generated": sim.n_generated,
                         "sim_mean_reco_minus_true_MeV": 1000 * float(np.mean(sim.reco_mass - sim.true_mass)),
                         "sim_mean_reco_minus_true_err_MeV": 1000 * float(np.std(sim.reco_mass - sim.true_mass)
                                                                          / np.sqrt(sim.reco_mass.size))}
        log("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.summary().items()
                   if k in ("data_mass", "sim_mass", "data_fwhm_MeV", "sim_fwhm_MeV", "data_fit_pvalue")})
    mass_comparison_figure(comps, figdir)
    kinematics_figure(comps, figdir)

    # systematic uncertainties of the data masses, and masses by detector region
    m_data = data.events["M_recomputed"].to_numpy()
    for name in RESONANCES:
        log(f"mass systematics {name} …")
        syst = mass_systematics(name, m_data, DATA_FIT[name])
        results[name]["data_mass_syst_MeV"] = syst.total_MeV
        results[name]["data_mass_systematics"] = syst.summary()
        results[name]["data_mass_by_region"] = masses_by_region(name, data.events, DATA_FIT[name])

    # detector studies
    log("resolution studies …")
    res, scans = resolution_figures(det, figdir, rng)
    cms_ref = cms_reference_figures(det, figdir, rng)
    event_display(det, figdir, rng)

    # fit validation with toys (simulation-fit models)
    toys = {}
    if cfg.toys > 0:
        for name in RESONANCES:
            log(f"toys {name} …")
            fs = fit_peak(sims[name].reco_mass, name, background=False, **SIM_FIT[name])
            toys[name] = toy_study(fs, cfg.toys, "mu", rng)
        toys_figure(toys, figdir)

    # tables
    save_json(results, out / "results.json")
    keys = ["process", "pdg_mass", "data_mass", "data_mass_err", "data_mass_syst_MeV", "sim_mass", "sim_mass_err", "data_minus_pdg_MeV",
            "sim_minus_pdg_MeV", "data_fwhm_MeV", "sim_fwhm_MeV", "fwhm_ratio_sim_over_data", "data_signal",
            "data_background", "data_fit_pvalue", "sim_fit_pvalue", "sim_acceptance"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in results.values():
            w.writerow(r)
    toy_summary = {k: {"n_toys": len(t.pulls), "failed": t.n_failed, "pull_mean": t.pull_mean,
                       "pull_mean_err": t.pull_mean_err, "pull_width": t.pull_std} for k, t in toys.items()}
    save_json({"resolution": res, "scans": scans, "cms_reference_comparison": cms_ref}, out / "detector_studies.json")
    save_json(toy_summary, out / "fit_validation.json")
    prov = cms.PROVENANCE_FILE.read_text(encoding="utf-8") if cms.PROVENANCE_FILE.exists() else None
    import json as _json
    rec = run_record(cfg, det, data={"record": cms.RECORD, "file_check": data.report.file_check,
                                      "provenance": _json.loads(prov) if prov else None,
                                      "selection": sel.to_dict(), "cutflow": dict(data.cutflow)},
                     results={"comparison": results, "fit_validation": toy_summary},
                     extra={"runtime_s": time.time() - t0})
    save_json(rec, out / "run_record.json")
    write_report(out, results, toy_summary, data, det, cfg)
    log(f"done in {time.time() - t0:.0f} s -> {out}")


def write_report(out, results, toys, data, det, cfg):
    L = ["# Analysis report", "",
         f"Generated by `run_analysis.py` (seed {cfg.seed}). Detector: {det.name}. "
         f"Selection: both muons pT > {cfg.pt_min} GeV, |η| < {cfg.eta_max}.", "",
         "## Data", "", f"File `{cms.RECORD['file']}` (CERN Open Data record 545), adler32 "
         f"{data.report.file_check.get('adler32')} (expected {cms.RECORD['adler32']}). "
         f"Rows: {data.report.n_rows}, rejected: {data.report.n_rejected}.", "",
         "| step | events |", "|---|---|"] + [f"| {k} | {v} |" for k, v in data.cutflow.items()] + [
         "", "Warnings (rows kept):", ""] + [f"- {k}: {v}" for k, v in data.report.warnings.items()] + [
         "", "## Peak comparison", "",
         "| | PDG M (GeV) | data M (GeV) | sim M (GeV) | data FWHM (MeV) | sim FWHM (MeV) | sim/data FWHM | data signal | data fit p |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in results.values():
        L.append(f"| {r['process']} | {r['pdg_mass']:.4f} | {r['data_mass']:.4f} ± {r['data_mass_err']:.4f} (stat) "
                 f"± {r['data_mass_syst_MeV'] / 1000:.4f} (syst) | "
                 f"{r['sim_mass']:.4f} ± {r['sim_mass_err']:.4f} | {r['data_fwhm_MeV']:.0f} | {r['sim_fwhm_MeV']:.0f} | "
                 f"{r['fwhm_ratio_sim_over_data']:.2f} | {r['data_signal']:.0f} ± {r['data_signal_err']:.0f} | {r['data_fit_pvalue']:.2f} |")
    L += ["", "## Systematic uncertainty of the data masses", "",
          "Each row refits the data with one fit choice changed. A variation counts only if the minimiser "
          "converged and p > 0.01; the largest accepted shift per category is used, and categories are added "
          "in quadrature.", "", "| | variation | shift (MeV) | p | used |", "|---|---|---|---|---|"]
    for r in results.values():
        sy = r["data_mass_systematics"]
        for v in sy["variations"]:
            L.append(f"| {r['process']} | {v['label']} | {v['shift_MeV']:+.2f} | {v['pvalue']:.3g} | "
                     f"{'yes' if v['accepted'] else 'no'}{(' — ' + v['note']) if v['note'] else ''} |")
        cats = ", ".join(f"{k} {v:.2f}" for k, v in sy["syst_by_category_MeV"].items())
        L.append(f"| {r['process']} | **total** ({cats}) | ±{sy['syst_total_MeV']:.2f} | | stat ±{sy['stat_err_MeV']:.2f} |")
    L += ["", "## Data masses by detector region (statistical errors only)", "",
          "| | region | events in window | M (GeV) | offset from PDG | fit p | error matrix OK |", "|---|---|---|---|---|---|---|"]
    for r in results.values():
        for k, v in r["data_mass_by_region"].items():
            if "mass" not in v:
                L.append(f"| {r['process']} | {k} | {v.get('n_events', '')} | {v.get('note', '')} | | | |")
                continue
            L.append(f"| {r['process']} | {k} | {v['n_events_window']} | {v['mass']:.4f} ± {v['mass_err']:.4f} | "
                     f"{v['offset_from_pdg_pct']:+.3f} % | {v['pvalue']:.3g} | {'yes' if v['hessian_ok'] else 'no'} |")
    L += ["", "## Simulation cut flow", ""]
    for r in results.values():
        L.append(f"- {r['process']}: " + ", ".join(f"{k}: {v}" for k, v in r["sim_cutflow"].items()))
    L += ["", "## Kinematic shape comparison (sideband-subtracted data vs simulation)", ""]
    for r in results.values():
        L.append(f"- {r['process']}: " + ", ".join(f"{k}: χ²/ndf = {v[0]:.0f}/{v[1]}" for k, v in r["kinematics_chi2"].items()))
    if toys:
        L += ["", "## Fit validation (toy Monte Carlo, mass parameter)", "", "| | toys | pull mean | pull width |", "|---|---|---|---|"]
        for k, t in toys.items():
            L.append(f"| {k} | {t['n_toys']} (+{t['failed']} failed) | {t['pull_mean']:+.2f} ± {t['pull_mean_err']:.2f} | {t['pull_width']:.2f} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

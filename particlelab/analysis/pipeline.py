"""Simulation driver: generate → detector → reconstruct → select.

Events are processed in chunks. Chunk i always uses the i-th child of
``SeedSequence(seed)``, so a run gives identical results for any number of
worker processes (reproducibility does not depend on parallelism).

Optional pre-selection (``preselect=True``): events whose TRUE muons are too far
below the pT cut or outside the |eta| cut to be selected are not passed through the
detector. The margins grow with the estimated pT resolution at the cut (see
``preselection_cuts``); they are twice the largest migration measured over the
GUI's extreme settings. A random fraction of the skipped events is simulated
anyway, and the number of those that would have been selected is reported, so
every run measures its own loss. Off by default: the published outputs use the
full chain.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..physics import generate, inv_mass, pt_eta_phi, ProductionModel
from ..detector import DetectorConfig, cms_inspired, simulate_hits, fit_tracks
from ..detector.tracking import gluckstern, ms_resolution

PRESELECT_STEP = "true muons can pass the cuts (pre-selection)"
PRESELECT_CHECK = 0.01      # fraction of skipped events simulated anyway, to measure the loss
CUTFLOW_STEPS = ("generated", "both tracks reconstructable", "both fits ok", "opposite charge",
                 "both pass pT and |eta| cuts")
_FIELDS = ("true_mass", "reco_mass", "p_minus", "p_plus", "true_minus", "true_plus", "true_parent",
           "sigma_pt_minus", "sigma_pt_plus")


@dataclass
class SimResult:
    name: str
    detector: DetectorConfig
    n_generated: int
    true_mass: np.ndarray
    reco_mass: np.ndarray
    p_minus: np.ndarray      # reconstructed momenta, labelled by reconstructed charge
    p_plus: np.ndarray
    true_minus: np.ndarray
    true_plus: np.ndarray
    true_parent: np.ndarray
    sigma_pt_minus: np.ndarray
    sigma_pt_plus: np.ndarray
    cutflow: dict = field(default_factory=dict)
    seed: object = None
    preselection: dict | None = None   # pre-selection cuts and measured loss, if used

    @property
    def q_minus(self): return -np.ones(len(self.reco_mass))
    @property
    def q_plus(self): return np.ones(len(self.reco_mass))

    @property
    def acceptance(self) -> float:
        return self.reco_mass.size / self.n_generated

    def muon_kinematics(self):
        pt, eta, _ = pt_eta_phi(np.vstack([self.p_minus, self.p_plus]))
        return pt, eta


def _passes(p, det):
    pt, eta, _ = pt_eta_phi(p)
    return (pt > det.pt_min) & (np.abs(eta) < det.eta_max)


@lru_cache(maxsize=1)
def _reference_sigma_rphi():
    return cms_inspired().sigma_rphi


def preselection_cuts(det: DetectorConfig):
    """(true pT threshold, true |eta| threshold) below/beyond which an event cannot be selected.

    σ is the estimated relative pT resolution at the cut (point-resolution term ⊕
    multiple-scattering term). Over the GUI's extreme settings the largest observed
    deficit (1 − true pT / pT cut) of a selected muon was 10 σ and the largest |eta|
    excess 0.16; the margins are 20 σ and 0.05 + 0.02 r + σ (r: hit resolution relative
    to the CMS-inspired detector), at least twice those. A margin of 100 % disables the pT test."""
    sig = float(np.hypot(gluckstern(det.pt_min, det), ms_resolution(det, det.pt_min)))
    r = det.sigma_rphi / _reference_sigma_rphi()
    return det.pt_min * max(0.0, 1.0 - 20.0 * sig), det.eta_max + 0.05 + 0.02 * r + sig


def _detector_chain(p_minus, p_plus, zv, det, rng):
    m = len(zv)
    hm = simulate_hits(p_minus, -np.ones(m), det, rng, zv)
    hp = simulate_hits(p_plus, np.ones(m), det, rng, zv)
    ok = (hm.n_detected >= det.min_hits) & (hp.n_detected >= det.min_hits)
    reco = ok.copy()
    fm, fp = fit_tracks(hm, det), fit_tracks(hp, det)
    ok &= fm.ok & fp.ok
    fits = ok.copy()
    ok &= fm.q * fp.q < 0
    sel = ok & _passes(np.nan_to_num(fm.p), det) & _passes(np.nan_to_num(fp.p), det)
    return fm, fp, reco, fits, ok, sel


def _run_chunk(args):
    name, m, det, model, seedseq, preselect = args
    rng = np.random.default_rng(seedseq)
    ev = generate(name, m, seed=rng, model=model)
    zv = rng.normal(0, det.vertex_sigma_z, m) if det.vertex_sigma_z > 0 else np.zeros(m)
    flow = {"generated": m}
    pre = None
    if preselect:
        pt_cut, eta_cut = preselection_cuts(det)
        def can_pass(p):
            pt, eta, _ = pt_eta_phi(p)
            return (pt > pt_cut) & (np.abs(eta) < eta_cut)
        keep = can_pass(ev.mu_minus) & can_pass(ev.mu_plus)
        check = ~keep & (rng.random(m) < PRESELECT_CHECK)
        n_lost = 0
        if check.any():
            n_lost = int(_detector_chain(ev.mu_minus[check], ev.mu_plus[check], zv[check], det, rng)[-1].sum())
        pre = {"skipped": int((~keep).sum()), "checked": int(check.sum()), "checked_selected": n_lost}
        flow[PRESELECT_STEP] = int(keep.sum())
        ev = type(ev)(ev.name, ev.parent[keep], ev.mass[keep], ev.mu_minus[keep], ev.mu_plus[keep])
        zv = zv[keep]
    fm, fp, reco, fits, ok, sel = _detector_chain(ev.mu_minus, ev.mu_plus, zv, det, rng)
    flow["both tracks reconstructable"] = int(reco.sum())
    flow["both fits ok"] = int(fits.sum())
    flow["opposite charge"] = int(ok.sum())
    flow["both pass pT and |eta| cuts"] = int(sel.sum())
    flow = {k: flow[k] for k in ("generated", PRESELECT_STEP, *CUTFLOW_STEPS[1:]) if k in flow}
    swap = fm.q > 0
    pm = np.where(swap[:, None], fp.p, fm.p)[sel]
    pp = np.where(swap[:, None], fm.p, fp.p)[sel]
    out = dict(true_mass=ev.mass[sel], reco_mass=inv_mass(pm, pp), p_minus=pm, p_plus=pp,
               true_minus=ev.mu_minus[sel], true_plus=ev.mu_plus[sel], true_parent=ev.parent[sel],
               sigma_pt_minus=np.where(swap, fp.sigma_pt, fm.sigma_pt)[sel],
               sigma_pt_plus=np.where(swap, fm.sigma_pt, fp.sigma_pt)[sel])
    return out, flow, pre


def simulate(name: str, n: int, det: DetectorConfig | None = None, seed=None, smear: bool = True,
             model: ProductionModel | None = None, chunk: int = 25_000, workers: int | None = None,
             progress=None, preselect: bool = False) -> SimResult:
    """Simulate n generated decays. ``workers`` > 1 uses a process pool (call from a
    ``if __name__ == "__main__":`` block on Windows). ``progress(done, total)`` is optional.
    ``preselect`` skips events that cannot pass the cuts (see the module docstring)."""
    det = det or cms_inspired()
    if not smear:
        det = det.ideal()
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**63)
    sizes = [chunk] * (n // chunk) + ([n % chunk] if n % chunk else [])
    seqs = np.random.SeedSequence(seed).spawn(len(sizes))
    jobs = [(name, m, det, model, s, preselect) for m, s in zip(sizes, seqs)]
    workers = workers or 1   # a process pool was slower on the reference laptop (see docs/validation)
    results = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_run_chunk, jobs)):
                results.append(r)
                if progress: progress(min((i + 1) * chunk, n), n)
    else:
        for i, j in enumerate(jobs):
            results.append(_run_chunk(j))
            if progress: progress(min((i + 1) * chunk, n), n)
    flow = dict.fromkeys(results[0][1], 0)
    for _, f, _ in results:
        for k in flow:
            flow[k] += f[k]
    pre = None
    if preselect:
        pt_cut, eta_cut = preselection_cuts(det)
        tot = {k: sum(r[2][k] for r in results) for k in ("skipped", "checked", "checked_selected")}
        frac = tot["checked_selected"] / tot["checked"] if tot["checked"] else 0.0
        pre = {"true_pt_min": pt_cut, "true_abs_eta_max": eta_cut, "check_fraction": PRESELECT_CHECK, **tot,
               # estimated selected events lost = skipped × (selected fraction of the checked ones)
               "estimated_lost": frac * tot["skipped"]}
    cat = {k: np.concatenate([r[0][k] for r in results]) for k in _FIELDS}
    return SimResult(name, det, n, cutflow=flow, seed=seed, preselection=pre, **cat)

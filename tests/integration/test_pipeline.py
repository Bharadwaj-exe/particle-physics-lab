import json

import numpy as np
import pytest

from particlelab.analysis import simulate, CUTFLOW_STEPS
from particlelab.analysis.compare import compare
from particlelab.data import cms
from particlelab.provenance import RunConfig, run_record, save_json


def test_same_seed_same_result_and_chunking_independent():
    a = simulate("Upsilon(1S)", 6000, seed=21, chunk=2000)
    b = simulate("Upsilon(1S)", 6000, seed=21, chunk=2000)
    c = simulate("Upsilon(1S)", 6000, seed=22, chunk=2000)
    assert np.array_equal(a.reco_mass, b.reco_mass)
    assert not np.array_equal(a.reco_mass[:50], c.reco_mass[:50])


def test_cutflow_is_monotonic_and_consistent():
    r = simulate("Z", 5000, seed=3)
    counts = [r.cutflow[k] for k in CUTFLOW_STEPS]
    assert counts[0] == 5000 and counts == sorted(counts, reverse=True)
    assert counts[-1] == r.reco_mass.size
    assert np.all(r.p_minus[:, 0] ** 2 + r.p_minus[:, 1] ** 2 > 9.0)


def test_parallel_matches_serial():
    if __import__("os").cpu_count() < 2:
        pytest.skip("single core")
    a = simulate("Z", 6000, seed=4, chunk=2000, workers=1)
    b = simulate("Z", 6000, seed=4, chunk=2000, workers=2)
    assert np.array_equal(a.reco_mass, b.reco_mass)


def test_config_roundtrip_and_run_record(tmp_path):
    cfg = RunConfig(seed=7, detector_overrides={"B": 3.0}, resolution_scale=2.0)
    cfg.save(tmp_path / "c.json")
    cfg2 = RunConfig.load(tmp_path / "c.json")
    assert cfg2 == cfg
    det = cfg2.build_detector()
    assert det.B == 3.0 and det.barrels[0].sigma_rphi == pytest.approx(20e-6)
    rec = run_record(cfg2, det, results={"x": np.float64(1.5)})
    save_json(rec, tmp_path / "r.json")
    back = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert back["config"]["seed"] == 7 and back["particle_data"]["source"].startswith("Particle Data Group")
    assert back["environment"]["packages"]["numpy"]


@pytest.mark.skipif(not cms.available(), reason="CMS file not downloaded")
def test_data_simulation_comparison_runs():
    data = cms.load()
    sim = simulate("Upsilon(1S)", 20000, seed=5)
    c = compare("Upsilon(1S)", data.events, sim)
    s = c.summary()
    assert abs(s["data_mass"] - 9.46) < 0.05 and s["data_fit_pvalue"] > 1e-3
    assert 0 < s["fwhm_ratio_sim_over_data"] < 2
    assert {k.variable for k in c.kinematics} == {"muon pT [GeV]", "muon eta"}


def test_run_analysis_logging_survives_a_non_utf8_redirect():
    """Audit A16: with output redirected to a cp1252 stream, logging μ or → must not raise."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONIOENCODING="cp1252", MPLBACKEND="Agg")
    code = "import run_analysis as r; r._safe_stdout(); r.log('J/psi → μ⁺μ⁻ done')"
    p = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True)
    assert p.returncode == 0, p.stderr.decode("cp1252", "replace")
    assert b"J/psi" in p.stdout


# ---- pre-selection (events whose true muons cannot pass the cuts are skipped) ----------

def _selected_true_muons(name, det, n, seed):
    """Full chain on n events; true momenta of the selected events."""
    from particlelab.analysis.pipeline import _detector_chain
    from particlelab.physics import generate
    rng = np.random.default_rng(seed)
    ev = generate(name, n, seed=rng)
    zv = rng.normal(0, det.vertex_sigma_z, n)
    sel = _detector_chain(ev.mu_minus, ev.mu_plus, zv, det, rng)[-1]
    return ev.mu_minus[sel], ev.mu_plus[sel]


@pytest.mark.parametrize("label", ["default", "B 0.5 T, resolution x20, material x10",
                                   "same with pT cut 0.5 GeV", "resolution x20, eta cut 1.0"])
def test_preselection_never_drops_a_selectable_event_at_extreme_settings(label):
    from particlelab.analysis.pipeline import preselection_cuts
    from particlelab.detector import cms_inspired
    from particlelab.physics import pt_eta_phi
    b = cms_inspired()
    det = {"default": b,
           "B 0.5 T, resolution x20, material x10": b.with_(B=0.5).scaled_resolution(20).scaled_material(10),
           "same with pT cut 0.5 GeV": b.with_(B=0.5, pt_min=0.5).scaled_resolution(20).scaled_material(10),
           "resolution x20, eta cut 1.0": b.with_(eta_max=1.0).scaled_resolution(20)}[label]
    pt_cut, eta_cut = preselection_cuts(det)
    for name in ("J/psi", "Upsilon(1S)"):
        for p in _selected_true_muons(name, det, 15000, 11):
            pt, eta, _ = pt_eta_phi(p)
            assert len(pt) > 50
            assert np.all(pt > pt_cut) and np.all(np.abs(eta) < eta_cut), (label, name)


def test_preselected_run_matches_full_run_statistically():
    full = simulate("Upsilon(1S)", 30000, seed=8)
    pre = simulate("Upsilon(1S)", 30000, seed=8, preselect=True)
    assert pre.preselection["checked"] > 0 and pre.preselection["checked_selected"] == 0
    flow = list(pre.cutflow.values())
    assert flow[0] == 30000 and flow == sorted(flow, reverse=True) and flow[-1] == pre.reco_mass.size
    # same physics: selected counts within 3 binomial sigma, peak mean and spread within errors
    n1, n2 = full.reco_mass.size, pre.reco_mass.size
    assert abs(n1 - n2) < 3 * np.sqrt(2 * n1 * (1 - n1 / 30000))
    core = lambda m: m[(m > 9.0) & (m < 9.9)]
    a, c = core(full.reco_mass), core(pre.reco_mass)
    assert abs(a.mean() - c.mean()) < 4 * a.std() / np.sqrt(len(a))
    assert abs(a.std() / c.std() - 1) < 0.05
    # without pre-selection nothing changed
    assert full.preselection is None and "true muons can pass the cuts (pre-selection)" not in full.cutflow

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

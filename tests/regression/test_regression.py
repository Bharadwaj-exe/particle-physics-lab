"""Regression: fixed-seed results must not change silently.

If a deliberate physics change alters these numbers, regenerate the baseline with
    python tests/regression/make_baseline.py
and record the reason in docs/engineering/DESIGN_DECISIONS.md.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from particlelab.domain.particles import RESONANCES

BASELINE = Path(__file__).with_name("baseline.json")


def current():
    from particlelab.analysis import simulate
    out = {}
    for i, name in enumerate(RESONANCES):
        r = simulate(name, 4000, seed=100 + i, chunk=4000)
        out[name] = {"selected": int(r.reco_mass.size),
                     "mean_reco_mass": float(np.mean(r.reco_mass)),
                     "std_reco_mass": float(np.std(r.reco_mass))}
    return out


@pytest.mark.skipif(not BASELINE.exists(), reason="no baseline yet")
def test_fixed_seed_results_unchanged():
    base = json.loads(BASELINE.read_text())
    now = current()
    for name in RESONANCES:
        assert now[name]["selected"] == base[name]["selected"]
        assert now[name]["mean_reco_mass"] == pytest.approx(base[name]["mean_reco_mass"], rel=1e-7)
        assert now[name]["std_reco_mass"] == pytest.approx(base[name]["std_reco_mass"], rel=1e-5)

"""Audit A2b: is the forward curvature offset caused by the disk radial error that the
circle fit ignores? Smearing only (no scattering), 5 GeV muons at eta = 2.3."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlelab.detector import cms_inspired, simulate_hits, fit_tracks   # noqa: E402


def offset(det, eta, n=400000, seed=1, pt=5.0):
    r = np.random.default_rng(seed)
    ph = r.uniform(-np.pi, np.pi, n)
    p = np.column_stack([pt * np.cos(ph), pt * np.sin(ph), np.full(n, pt * np.sinh(eta))])
    q = r.choice([-1.0, 1.0], n)
    f = fit_tracks(simulate_hits(p, q, det, r, r.normal(0, det.vertex_sigma_z, n)), det)
    k = pt / np.hypot(f.p[f.ok, 0], f.p[f.ok, 1]) - 1
    return dict(mean=float(k.mean()), se=float(k.std() / np.sqrt(k.size)), pull=float(k.mean() / (k.std() / np.sqrt(k.size))))


base = cms_inspired().with_(multiple_scattering=False)
no_r = replace(base, disks=tuple(replace(d, sigma_r=float("inf")) for d in base.disks))   # inf -> no radial smearing
out = {"eta2.3_smear_only": offset(base, 2.3),
       "eta2.3_smear_only_no_disk_r_smear": offset(no_r, 2.3),
       "eta0_smear_only": offset(base, 0.0)}
Path(__file__).with_name("results").joinpath("a2b_curvature_cause.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

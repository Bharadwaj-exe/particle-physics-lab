"""Calibrate the material of cms_inspired() to the published CMS X0(η) curve.

Reference: references/cms_tracker_material_x0.json (JINST 9 (2014) P10009, Fig. 2),
digitised per subsystem. The material met by a straight track is linear in the
surface thicknesses, so each subsystem is a bounded linear least-squares problem
over 0.1-wide |η| bins from 0 to 2.5, minimising relative residuals
(weights 1/max(band, 0.02 X0)):

    pixel    target = (pixel + beam pipe) − beam pipe of the model (fixed 0.8 mm Be)
    TIB/TID  target = band height of TIB and TID
    TOB      target = band height of TOB
    TEC      target = band height of TEC

The support tube outside the tracker is not modelled: it lies beyond the last
measurement and cannot affect the momentum measurement.

Only the material reference is used. Nothing here looks at dimuon data or at
momentum resolution.

Usage (from the project root):  python calibration/fit_material.py
It prints the CMS_MATERIAL dictionary for detector/geometry.py and writes
calibration/material_fit.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from particlelab.detector.geometry import CMS_MATERIAL, cms_inspired, material_along  # noqa: E402

GROUPS = {
    "pixel": ["PXB", "FPIX", "PIX_flange_inner", "PIX_flange_outer", "PIX_service_cyl"],
    "TIB/TID": ["TIB", "TID", "TIB_end", "TIB_service_inner", "TIB_service_outer", "TID_end"],
    "TOB": ["TOB", "TOB_service_inner", "TOB_service_outer", "TOB_end"],
    "TEC": ["TEC"],
}
SENSITIVE = {"PXB", "FPIX", "TIB", "TID", "TOB", "TEC"}
BOUNDS_SENSITIVE = (0.01, 0.15)    # a silicon module with its hybrid and support
BOUNDS_PASSIVE = (0.0, 1.5)


def reference():
    ref = json.loads((ROOT / "references" / "cms_tracker_material_x0.json").read_text(encoding="utf-8"))
    eta = np.array([b["eta"] for b in ref["bins"]])
    c = {k: np.array([b["cumulative"][k] for b in ref["bins"]]) for k in ref["bins"][0]["cumulative"]}
    keep = eta < 2.5
    bp = np.array([material_along(cms_inspired({k: 0.0 for k in CMS_MATERIAL}), e)["beam pipe"] for e in eta])
    bands = {"pixel": c["pixel_and_beampipe"] - bp, "TIB/TID": c["tib_tid"] - c["pixel_and_beampipe"],
             "TOB": c["tob"] - c["tib_tid"], "TEC": c["tec"] - c["tob"]}
    return eta[keep], {k: v[keep] for k, v in bands.items()}, c["tec"][keep]


def column(group, eta, sub):
    t = {k: 0.0 for k in CMS_MATERIAL}
    t[group], t["BP"] = 1.0, 0.0
    det = cms_inspired(t)
    return np.array([material_along(det, e)[sub] for e in eta])


def fit():
    eta, bands, total_ref = reference()
    result, table = {}, {}
    for sub, groups in GROUPS.items():
        A = np.column_stack([column(g, eta, sub) for g in groups])
        lo = [BOUNDS_SENSITIVE[0] if g in SENSITIVE else BOUNDS_PASSIVE[0] for g in groups]
        hi = [BOUNDS_SENSITIVE[1] if g in SENSITIVE else BOUNDS_PASSIVE[1] for g in groups]
        # relative residuals: every |η| bin counts equally, so the thin central
        # region is not traded away against the thick forward region
        wgt = 1.0 / np.maximum(bands[sub], 0.02)
        r = lsq_linear(A * wgt[:, None], bands[sub] * wgt, bounds=(lo, hi))
        result.update({g: round(float(v), 4) for g, v in zip(groups, r.x)})
        table[sub] = dict(reference=bands[sub].round(3).tolist(), model=(A @ r.x).round(3).tolist(),
                          rms_residual=float(np.sqrt(np.mean((A @ r.x - bands[sub]) ** 2))))
    det = cms_inspired(result)
    model_total = np.array([sum(material_along(det, e).values()) for e in eta])
    return eta, result, table, total_ref, model_total


if __name__ == "__main__":
    eta, result, table, total_ref, model_total = fit()
    print("CMS_MATERIAL = {")
    for sub, groups in GROUPS.items():
        print("    " + ", ".join(f'"{g}": {result[g]}' for g in groups) + ",")
    print("}")
    print(f"\n{'|eta|':>6} {'CMS':>6} {'model':>6} {'ratio':>6}")
    for e, a, b in zip(eta, total_ref, model_total):
        print(f"{e:6.2f} {a:6.3f} {b:6.3f} {b / a:6.2f}")
    for sub, t in table.items():
        print(f"{sub:8s} rms residual {t['rms_residual']:.3f} X0")
    out = dict(fitted=result, eta=eta.round(2).tolist(), cms_tracker_total=total_ref.round(3).tolist(),
               model_total=model_total.round(3).tolist(), by_subsystem=table,
               reference="references/cms_tracker_material_x0.json")
    (Path(__file__).with_name("material_fit.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")

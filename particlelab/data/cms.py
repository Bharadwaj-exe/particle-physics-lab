"""CMS 2011 open-data dimuon pipeline (CERN Open Data record 545).

Dataset: "Datasets derived from the Run2011A SingleElectron, SingleMu, DoubleElectron,
and DoubleMu primary datasets", file Dimuon_DoubleMu.csv. Licence CC0-1.0.
Producer selection (from the record page): two muons with |η| < 2.4, at least one
global muon, opposite charge, 0.3 < M < 300 GeV.

Pipeline: fetch → verify checksum → validate schema → validate rows (rejections
counted and saved with reasons, never silently dropped) → select → masses.
The raw file is never modified.
"""
from __future__ import annotations

import json
import urllib.request
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..physics.kinematics import inv_mass, M_MU

RECORD = {
    "record": 545,
    "title": "Datasets derived from the Run2011A SingleElectron, SingleMu, DoubleElectron, and DoubleMu primary datasets",
    "record_url": "https://opendata.cern.ch/record/545",
    "file": "Dimuon_DoubleMu.csv",
    "url": "https://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv",
    "size": 13935840,
    "adler32": "5b87a72a",
    "license": "CC0-1.0",
    "collision": "pp, sqrt(s) = 7 TeV, LHC Run2011A",
    "producer_selection": "two muons |eta|<2.4, >=1 global, opposite charge, 0.3<M<300 GeV",
}
EXPECTED_COLUMNS = ["Run", "Event", "type1", "E1", "px1", "py1", "pz1", "pt1", "eta1", "phi1", "Q1",
                    "type2", "E2", "px2", "py2", "pz2", "pt2", "eta2", "phi2", "Q2", "M"]
NUMERIC = [c for c in EXPECTED_COLUMNS if not c.startswith("type")]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_FILE = DATA_DIR / "raw" / RECORD["file"]
PROVENANCE_FILE = DATA_DIR / "raw" / (RECORD["file"] + ".provenance.json")


def _display_path(path) -> str:
    """Path relative to the project folder when possible, so that records and reports
    do not contain machine-specific absolute paths."""
    p = Path(path).resolve()
    try:
        return p.relative_to(DATA_DIR.parent).as_posix()
    except ValueError:
        return str(p)


class SchemaError(ValueError):
    pass


def adler32(path: Path) -> str:
    v = 1
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            v = zlib.adler32(block, v)
    return format(v & 0xFFFFFFFF, "08x")


def verify(path: Path = RAW_FILE) -> dict:
    path = Path(path)
    size = path.stat().st_size
    chk = adler32(path)
    return {"path": _display_path(path), "size": size, "adler32": chk,
            "size_ok": size == RECORD["size"], "checksum_ok": chk == RECORD["adler32"]}


def fetch(dest: Path = RAW_FILE, force: bool = False) -> Path:
    """Download the CSV (if needed), verify it and write a provenance record."""
    dest = Path(dest)
    if dest.exists() and not force and verify(dest)["checksum_ok"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(RECORD["url"], tmp)
    v = verify(tmp)
    if not (v["size_ok"] and v["checksum_ok"]):
        tmp.unlink(missing_ok=True)
        raise IOError(f"Downloaded file failed verification: {v}")
    tmp.replace(dest)
    prov = {**RECORD, "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verified": verify(dest)}
    PROVENANCE_FILE.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return dest


def available(path: Path = RAW_FILE) -> bool:
    return Path(path).exists()


# ---- validation ---------------------------------------------------------------

@dataclass
class ValidationReport:
    path: str
    n_rows: int
    n_valid: int
    rejected: dict = field(default_factory=dict)       # reason -> count (a row can have several)
    n_rejected: int = 0
    warnings: dict = field(default_factory=dict)       # reason -> count (rows kept)
    mass_check: dict = field(default_factory=dict)
    file_check: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run particlelab.data.cms.fetch() or the download script")
    df = pd.read_csv(path, dtype={"type1": str, "type2": str})
    if list(df.columns) != EXPECTED_COLUMNS:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        raise SchemaError(f"Unexpected columns. Missing: {missing}; extra: {extra}")
    return df


def validate(df: pd.DataFrame, path: str = "") -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """Split rows into valid and rejected. Returns (valid, rejected_with_reason, report)."""
    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)
    def flag(mask, reason, counts):
        counts[reason] = int(mask.sum())
        reasons[mask] = reasons[mask] + reason + "; "

    rej: dict = OrderedDict()
    num = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    flag(~np.isfinite(num).all(axis=1), "non-finite or non-numeric value", rej)
    flag(~df["Q1"].isin([-1, 1]) | ~df["Q2"].isin([-1, 1]), "charge not ±1", rej)
    flag(~df["type1"].isin(["G", "T"]) | ~df["type2"].isin(["G", "T"]), "unknown muon type", rej)
    for i in (1, 2):
        p = np.sqrt(num[f"px{i}"]**2 + num[f"py{i}"]**2 + num[f"pz{i}"]**2)
        # values are rounded to 1e-4 GeV, hence the tolerances
        flag(num[f"E{i}"] < p - 2e-3 - 1e-4 * p, f"muon {i}: E < |p|", rej)
    flag(df["Q1"] * df["Q2"] >= 0, "not opposite charge", rej)

    bad = reasons != ""
    warn = OrderedDict()
    for i in (1, 2):
        # The published pt/eta columns disagree with (px, py, pz) for ~3% of muons, while
        # the published M agrees with (px, py, pz, E). They are kept; cuts use values
        # recomputed from the momentum components.
        pt = np.hypot(num[f"px{i}"], num[f"py{i}"])
        warn[f"muon {i}: published pt differs from hypot(px,py) by > 0.1%+2 MeV (row kept)"] = int(
            np.sum((np.abs(pt - num[f"pt{i}"]) > 2e-3 + 1e-3 * pt) & ~bad))
        eta = np.arcsinh(num[f"pz{i}"] / pt)
        warn[f"muon {i}: published eta differs from recomputed by > 0.002 (row kept)"] = int(
            np.sum((np.abs(eta - num[f"eta{i}"]) > 2e-3) & ~bad))
    m_re = inv_mass(num[["px1", "py1", "pz1"]].to_numpy(), num[["px2", "py2", "pz2"]].to_numpy())
    dm = np.abs(m_re - num["M"].to_numpy())
    tol = 1e-3 + 1e-3 * num["M"].to_numpy()
    warn["published M differs from recomputed M by > 0.1%+1 MeV (row kept; recomputed M used)"] = int(
        np.sum((dm > tol) & ~bad.to_numpy()))
    valid = df[~bad].copy()
    valid["M_recomputed"] = m_re[~bad.to_numpy()]
    for i in (1, 2):
        valid[f"pt{i}_re"] = np.hypot(valid[f"px{i}"], valid[f"py{i}"])
        valid[f"eta{i}_re"] = np.arcsinh(valid[f"pz{i}"] / valid[f"pt{i}_re"])
    rejected = df[bad].copy()
    rejected["reasons"] = reasons[bad].str.rstrip("; ")
    report = ValidationReport(path, len(df), int((~bad).sum()), {k: v for k, v in rej.items()}, int(bad.sum()),
                              dict(warn), {"median_abs_diff_GeV": float(np.median(dm)), "max_abs_diff_GeV": float(dm.max()),
                                           "muon_mass_GeV": M_MU})
    return valid, rejected, report


# ---- selection ----------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    pt_min: float = 3.0
    eta_max: float = 2.4
    require_both_global: bool = False
    window: tuple | None = None

    def to_dict(self):
        return asdict(self)


def select(valid: pd.DataFrame, sel: Selection) -> tuple[pd.DataFrame, OrderedDict]:
    flow = OrderedDict()
    d = valid
    flow["validated events"] = len(d)
    d = d[(d.pt1_re > sel.pt_min) & (d.pt2_re > sel.pt_min)]
    flow[f"both muons pT > {sel.pt_min} GeV"] = len(d)
    d = d[(d.eta1_re.abs() < sel.eta_max) & (d.eta2_re.abs() < sel.eta_max)]
    flow[f"both muons |eta| < {sel.eta_max}"] = len(d)
    if sel.require_both_global:
        d = d[(d.type1 == "G") & (d.type2 == "G")]
        flow["both global muons"] = len(d)
    if sel.window:
        lo, hi = sel.window
        d = d[(d.M_recomputed > lo) & (d.M_recomputed < hi)]
        flow[f"{lo} < M < {hi} GeV"] = len(d)
    return d, flow


@dataclass
class CMSDataset:
    events: pd.DataFrame          # validated and selected
    report: ValidationReport
    rejected: pd.DataFrame
    cutflow: OrderedDict
    selection: Selection

    @property
    def masses(self) -> np.ndarray:
        return self.events["M_recomputed"].to_numpy()

    def muon_momenta(self):
        e = self.events
        return (e[["px1", "py1", "pz1"]].to_numpy(), e[["px2", "py2", "pz2"]].to_numpy())


def load(path: Path = RAW_FILE, selection: Selection = Selection(), check_file: bool = True) -> CMSDataset:
    path = Path(path)
    df = load_raw(path)
    valid, rejected, report = validate(df, _display_path(path))
    if check_file:
        report.file_check = verify(path)
    events, flow = select(valid, selection)
    return CMSDataset(events, report, rejected, flow, selection)


def masses(path: Path = RAW_FILE, selection: Selection = Selection()) -> np.ndarray:
    return load(path, selection, check_file=False).masses

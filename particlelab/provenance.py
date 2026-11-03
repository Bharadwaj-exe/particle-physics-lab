"""Run configuration and provenance records (what was run, with what, from which data)."""
from __future__ import annotations

import json
import re
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from . import __version__
from .domain.particles import load as load_particles
from .detector import DetectorConfig, cms_inspired, uniform_barrel

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunConfig:
    """Everything needed to reproduce a run. Saved and loaded as JSON."""
    seed: int = 20261102
    events: dict = field(default_factory=lambda: {"J/psi": 1_000_000, "Upsilon(1S)": 200_000, "Z": 100_000})
    detector: str = "cms_inspired"           # or "uniform_barrel"
    detector_overrides: dict = field(default_factory=dict)   # e.g. {"B": 3.0, "hit_efficiency": 0.95}
    resolution_scale: float = 1.0
    material_scale: float = 1.0
    pt_min: float = 3.0
    eta_max: float = 2.4
    toys: int = 100
    output_dir: str = "outputs"

    def build_detector(self) -> DetectorConfig:
        det = {"cms_inspired": cms_inspired, "uniform_barrel": uniform_barrel}[self.detector]()
        det = det.with_(pt_min=self.pt_min, eta_max=self.eta_max, **self.detector_overrides)
        if self.resolution_scale != 1.0:
            det = det.scaled_resolution(self.resolution_scale)
        if self.material_scale != 1.0:
            det = det.scaled_material(self.material_scale)
        return det

    @classmethod
    def load(cls, path) -> "RunConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _git():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        # the run's own output folders are written before the record; they do not
        # make the code state dirty
        changed = [ln for ln in st.stdout.splitlines()
                   if ln.strip() and not re.search(r"(^|/)outputs(_quick)?(/|$)", ln[3:].strip('"'))]
        return {"commit": rev.stdout.strip() or None, "dirty": bool(changed), "changed_files": changed[:50]}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def environment() -> dict:
    vers = {}
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "PySide6", "pyqtgraph", "pyvista", "pyvistaqt", "vtk"):
        try:
            vers[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            vers[pkg] = None
    return {"particlelab": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": vers, "git": _git()}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def run_record(config: RunConfig | None = None, detector: DetectorConfig | None = None, data: dict | None = None,
               results: dict | None = None, extra: dict | None = None) -> dict:
    db = load_particles()
    rec = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        "config": asdict(config) if config else None,
        "detector": detector.to_dict() if detector else None,
        "particle_data": {"source": db.source, "retrieved": db.retrieved},
        "data_provenance": data,
        "results": results,
    }
    if extra:
        rec.update(extra)
    return _jsonable(rec)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")

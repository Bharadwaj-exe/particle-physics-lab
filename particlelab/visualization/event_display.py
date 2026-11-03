"""Display data for one simulated event, independent of any rendering library.

Turns the outputs of the existing physics and detector code (unchanged) into
plain arrays that the 2D (pyqtgraph) and 3D (PyVista) views draw. Coordinates
are in metres, with the beam along z, matching the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detector import DetectorConfig, simulate_hits, fit_tracks
from ..detector.simulation import helix_at
from ..physics import generate, inv_mass, pt_eta_phi
from ..units import C_TRACK, cm

BEAM_PIPE_RADIUS = 3.0 * cm      # drawn for orientation only; not simulated as material
BEAM_AXIS_HALF_LENGTH = 3.2


# ---- detector geometry ---------------------------------------------------------

@dataclass(frozen=True)
class CylinderShape:
    name: str
    group: str            # "pixel barrel", "strip barrel", "beam pipe"
    radius: float
    half_length: float


@dataclass(frozen=True)
class DiskShape:
    name: str
    group: str            # "pixel disks", "inner disks", "endcap disks"
    z: float              # signed position
    r_min: float
    r_max: float


@dataclass(frozen=True)
class DetectorGeometry:
    cylinders: tuple
    disks: tuple
    beam_axis: tuple      # ((0,0,-L), (0,0,+L))
    key: tuple            # identifies the geometry, for caching

    def groups(self):
        return sorted({s.group for s in self.cylinders + self.disks})


def _disk_group(name: str) -> str:
    return {"FPIX": "pixel disks", "TID": "inner disks", "TEC": "endcap disks"}.get(
        "".join(c for c in name if c.isalpha()), "endcap disks")


def detector_geometry(det: DetectorConfig, max_half_length: float = 3.0) -> DetectorGeometry:
    """Shapes taken directly from the detector configuration used by the simulation.

    Infinitely long reference layers are drawn with ``max_half_length``.
    """
    cyl = [CylinderShape("beam pipe", "beam pipe", BEAM_PIPE_RADIUS, max_half_length)]
    for b in det.barrels:
        group = "pixel barrel" if b.name.startswith("PX") else "strip barrel"
        cyl.append(CylinderShape(b.name, group, b.radius, min(b.half_length, max_half_length)))
    disks = []
    for d in det.disks:
        for sgn in (1.0, -1.0):
            disks.append(DiskShape(f"{d.name}{'+' if sgn > 0 else '-'}", _disk_group(d.name), sgn * d.z, d.r_min, d.r_max))
    key = (tuple((c.radius, c.half_length) for c in cyl), tuple((d.z, d.r_min, d.r_max) for d in disks))
    axis = ((0.0, 0.0, -BEAM_AXIS_HALF_LENGTH), (0.0, 0.0, BEAM_AXIS_HALF_LENGTH))
    return DetectorGeometry(tuple(cyl), tuple(disks), axis, key)


# ---- event -------------------------------------------------------------------------

@dataclass
class DisplayTrack:
    label: str                 # "μ⁻" / "μ⁺"
    charge_true: int
    charge_reco: int
    p_true: np.ndarray         # (3,) GeV
    p_reco: np.ndarray         # (3,) GeV
    true_path: np.ndarray      # (N,3) m
    reco_path: np.ndarray      # (N,3) m
    hits: np.ndarray           # (k,3) recorded hits, smeared (includes multiple scattering)
    hits_ideal: np.ndarray     # (k,3) where the true helix crosses those surfaces
    hits_disk: np.ndarray      # (k,) bool
    missed: np.ndarray         # (m,3) crossings that were not recorded (inefficiency)
    sigma_pt: float            # hit-error-only uncertainty
    n_hits: int

    @property
    def pt_true(self): return float(np.hypot(*self.p_true[:2]))
    @property
    def pt_reco(self): return float(np.hypot(*self.p_reco[:2]))
    @property
    def eta_reco(self): return float(np.arcsinh(self.p_reco[2] / self.pt_reco))

    def hits_exaggerated(self, factor: float) -> np.ndarray:
        """Hit positions with residuals from the true crossing scaled by ``factor`` (display only)."""
        return self.hits_ideal + factor * (self.hits - self.hits_ideal)


@dataclass
class DisplayEvent:
    process: str
    detector_name: str
    B: float
    vertex: np.ndarray         # (3,)
    true_mass: float
    reco_mass: float
    tracks: list = field(default_factory=list)
    passes_cuts: bool = True
    seed: object = None


def helix_path(p, q, B, z0, s_max, n=240):
    """Points (n,3) along the helix of momentum p (GeV) and charge q from (0, 0, z0)."""
    p = np.asarray(p, float)
    pt = np.hypot(p[0], p[1])
    R = pt / (C_TRACK * B)
    s = np.linspace(0.0, s_max, n)
    x, y, z = helix_at(s, R, float(np.sign(q)), np.arctan2(p[1], p[0]), p[2] / pt, z0)
    return np.column_stack([x, y, z])


def momentum_arrow(p, origin, base=0.15, per_decade=0.12):
    """Arrow along p from ``origin``; length grows with log10(|p| / 1 GeV) so all muons stay visible."""
    p = np.asarray(p, float)
    mag = np.linalg.norm(p)
    direction = p / mag
    return np.asarray(origin, float), direction, base + per_decade * max(np.log10(max(mag, 1.0)), 0.0)


def _passes(p, det):
    pt, eta, _ = pt_eta_phi(np.atleast_2d(p))
    return bool(pt[0] > det.pt_min and abs(eta[0]) < det.eta_max)


def make_event(name: str, det: DetectorConfig, rng: np.random.Generator, require_cuts: bool = True,
               max_tries: int = 2000) -> DisplayEvent:
    """Generate one decay and run it through the detector with the existing code.

    With ``require_cuts`` the event is redrawn until both muons are reconstructed with
    opposite charges and pass the analysis pT and |η| cuts, i.e. it would enter the
    mass spectrum.
    """
    for _ in range(max_tries):
        ev = generate(name, 1, seed=rng)
        z0 = float(rng.normal(0, det.vertex_sigma_z)) if det.vertex_sigma_z > 0 else 0.0
        zv = np.array([z0])
        parts = []
        for p4, q, lab in ((ev.mu_minus[0], -1, "μ⁻"), (ev.mu_plus[0], 1, "μ⁺")):
            h = simulate_hits(p4[None, 1:], np.array([float(q)]), det, rng, zv)
            f = fit_tracks(h, det)
            parts.append((p4, q, lab, h, f))
        ok = all(f.ok[0] for *_, f in parts)
        if ok:
            ok = parts[0][4].q[0] * parts[1][4].q[0] < 0
        passes = ok and all(_passes(f.p[0], det) for *_, f in parts)
        if (passes or not require_cuts) and ok:
            break
    else:
        raise RuntimeError(f"no reconstructable {name} event in {max_tries} tries")

    tracks = []
    for p4, q, lab, h, f in parts:
        p = p4[1:]
        crossed = h.crossed[0]
        s_end = float(h.s[0][crossed].max()) if crossed.any() else 1.0
        pt = np.hypot(p[0], p[1])
        R = pt / (C_TRACK * det.B)
        rec = h.detected[0]
        ideal_x, ideal_y, ideal_z = helix_at(h.s[0][rec], R, float(q), np.arctan2(p[1], p[0]), p[2] / pt, z0)
        tracks.append(DisplayTrack(
            label=lab, charge_true=int(q), charge_reco=int(f.q[0]), p_true=p.copy(), p_reco=f.p[0].copy(),
            true_path=helix_path(p, q, det.B, z0, s_end),
            reco_path=helix_path(f.p[0], f.q[0], det.B, float(f.z0[0]), s_end),
            hits=np.column_stack([h.x[0][rec], h.y[0][rec], h.z[0][rec]]),
            hits_ideal=np.column_stack([ideal_x, ideal_y, ideal_z]),
            hits_disk=h.is_disk[0][rec].copy(),
            missed=np.column_stack([h.x[0][crossed & ~rec], h.y[0][crossed & ~rec], h.z[0][crossed & ~rec]]),
            sigma_pt=float(f.sigma_pt[0]), n_hits=int(f.n_hits[0])))
    reco_m = float(inv_mass(tracks[0].p_reco[None], tracks[1].p_reco[None])[0])
    return DisplayEvent(name, det.name, det.B, np.array([0.0, 0.0, z0]), float(ev.mass[0]), reco_m, tracks, passes)

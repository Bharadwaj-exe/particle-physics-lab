"""Display data for one event, independent of any rendering library.

Turns the outputs of the existing physics and detector code (unchanged) into
plain arrays that the 2D (pyqtgraph) and 3D (PyVista) views draw. Coordinates
are in metres, with the beam along z, matching the simulation.

Every point on a path and every hit also carries its flight time in ns,
t = (path length) / (beta c), so the views can animate an event on one physical
clock. Events come from the simulation (``make_event``) or from the CMS 2011
open data (``data_event``), where only the measured momenta are known.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detector import DetectorConfig, simulate_hits, fit_tracks
from ..detector.simulation import helix_at
from ..physics import generate, inv_mass, pt_eta_phi
from ..physics.kinematics import M_MU
from ..units import C_TRACK, cm

BEAM_PIPE_RADIUS = 3.0 * cm      # drawn for orientation only; not simulated as material
BEAM_AXIS_HALF_LENGTH = 3.2
C_LIGHT = 0.299792458            # m / ns
DATA_B = 3.8                     # T, CMS solenoid field in 2011
# narrow mass windows for picking data events to display: mostly signal, some background
DATA_DISPLAY_WINDOWS = {"J/psi": (3.0, 3.2), "Upsilon(1S)": (9.3, 9.6), "Z": (85.0, 97.0)}


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
    true_t: np.ndarray = None  # (N,) ns, flight time at each point of true_path
    reco_t: np.ndarray = None  # (N,) ns, along reco_path
    hit_t: np.ndarray = None   # (k,) ns, when the muon crosses each recorded hit's surface
    missed_t: np.ndarray = None
    crossings: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))   # data events: computed layer crossings
    crossing_t: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def has_truth(self): return len(self.true_path) > 0

    @property
    def flight_path(self):
        """Path followed in the animation: the true one if known, else the measured one."""
        return (self.true_path, self.true_t) if self.has_truth else (self.reco_path, self.reco_t)

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
    source: str = "simulation"     # or "CMS 2011 data"
    run: int | None = None
    event_number: int | None = None
    muon_types: tuple = ()         # data: CMS muon type per track, G (global) or T (tracker)

    @property
    def is_data(self): return self.source != "simulation"

    @property
    def t_max(self):
        """Time in ns when the last muon leaves the drawn volume."""
        return max(float(t.flight_path[1][-1]) for t in self.tracks)


def helix_path(p, q, B, z0, s_max, n=240):
    """Points (n,3) along the helix of momentum p (GeV) and charge q from (0, 0, z0)."""
    p = np.asarray(p, float)
    pt = np.hypot(p[0], p[1])
    R = pt / (C_TRACK * B)
    s = np.linspace(0.0, s_max, n)
    x, y, z = helix_at(s, R, float(np.sign(q)), np.arctan2(p[1], p[0]), p[2] / pt, z0)
    return np.column_stack([x, y, z])


def flight_time(s, p):
    """Flight time in ns after transverse arc length ``s`` (m) for a muon of momentum ``p`` (GeV)."""
    p = np.asarray(p, float)
    pt, ptot = np.hypot(p[0], p[1]), np.linalg.norm(p)
    beta = ptot / np.hypot(ptot, M_MU)
    return np.asarray(s, float) * (ptot / pt) / (beta * C_LIGHT)


def path_until(path, t, t_cut):
    """Part of ``path`` (N,3) flown by time ``t_cut``, ending exactly at the interpolated position."""
    path, t = np.asarray(path), np.asarray(t)
    if t_cut >= t[-1]:
        return path
    if t_cut <= t[0]:
        return path[:1]
    i = int(np.searchsorted(t, t_cut, side="right"))
    f = (t_cut - t[i - 1]) / (t[i] - t[i - 1])
    return np.vstack([path[:i], path[i - 1] + f * (path[i] - path[i - 1])])


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
            sigma_pt=float(f.sigma_pt[0]), n_hits=int(f.n_hits[0]),
            true_t=flight_time(np.linspace(0.0, s_end, 240), p),
            reco_t=flight_time(np.linspace(0.0, s_end, 240), f.p[0]),
            hit_t=flight_time(h.s[0][rec], p),
            missed_t=flight_time(h.s[0][crossed & ~rec], p)))
    reco_m = float(inv_mass(tracks[0].p_reco[None], tracks[1].p_reco[None])[0])
    return DisplayEvent(name, det.name, det.B, np.array([0.0, 0.0, z0]), float(ev.mass[0]), reco_m, tracks, passes)


# ---- CMS open-data events -------------------------------------------------------------

def layer_crossings(p, q, det: DetectorConfig, z0: float = 0.0):
    """Where an ideal helix (no scattering, no smearing) crosses the sensitive layers of ``det``.

    Uses the simulation's own surface code with smearing, scattering and inefficiency off,
    so the points are exactly where the model would place unsmeared hits.
    Returns (points (k,3), transverse arc lengths (k,)).
    """
    ideal = det.with_(smear=False, multiple_scattering=False, hit_efficiency=1.0)
    h = simulate_hits(np.asarray(p, float)[None], np.array([float(q)]), ideal, np.random.default_rng(0),
                      np.array([z0]))
    m = h.detected[0]
    return np.column_stack([h.x[0][m], h.y[0][m], h.z[0][m]]), h.s[0][m]


def exit_arc_length(p, B, r_out, z_out, z0=0.0):
    """Transverse arc length at which a helix from (0, 0, z0) leaves r < r_out, |z| < z_out."""
    p = np.asarray(p, float)
    pt = np.hypot(p[0], p[1])
    R = pt / (C_TRACK * B)
    s_r = 2 * R * np.arcsin(r_out / (2 * R)) if r_out < 2 * R else np.pi * R     # a looper turns back at s = πR
    cot = p[2] / pt
    s_z = (np.sign(cot) * z_out - z0) / cot if abs(cot) > 1e-12 else np.inf
    return float(min(s_r, s_z))


def data_event(row, process: str, det: DetectorConfig) -> DisplayEvent:
    """Display event from one row of the CMS dimuon CSV (columns px1 … Q2, Run, Event, type1, type2).

    Only the measured momenta and charges are known: the vertex is drawn at the origin
    (it is not in the file) and the tracks are ideal helices in ``det.B`` (use 3.8 T), without
    scattering or energy loss. ``crossings`` are computed from these momenta in ``det``'s
    geometry; they are not CMS hits.
    """
    tracks = []
    r_out = max(b.radius for b in det.barrels)
    z_out = max(d.z for d in det.disks) if det.disks else max(min(b.half_length, 3.0) for b in det.barrels)
    for i in (1, 2):
        p = np.array([row[f"px{i}"], row[f"py{i}"], row[f"pz{i}"]], float)
        q = int(row[f"Q{i}"])
        s_end = exit_arc_length(p, det.B, r_out, z_out)
        pts, s_hit = layer_crossings(p, q, det)
        tracks.append(DisplayTrack(
            label="μ⁻" if q < 0 else "μ⁺", charge_true=q, charge_reco=q, p_true=p.copy(), p_reco=p.copy(),
            true_path=np.empty((0, 3)), reco_path=helix_path(p, q, det.B, 0.0, s_end), hits=np.empty((0, 3)),
            hits_ideal=np.empty((0, 3)), hits_disk=np.empty(0, bool), missed=np.empty((0, 3)),
            sigma_pt=float("nan"), n_hits=0, true_t=np.empty(0),
            reco_t=flight_time(np.linspace(0.0, s_end, 240), p), hit_t=np.empty(0), missed_t=np.empty(0),
            crossings=pts, crossing_t=flight_time(s_hit, p)))
    tracks.sort(key=lambda t: t.charge_reco)          # μ⁻ first, as in simulated events
    m = float(inv_mass(tracks[0].p_reco[None], tracks[1].p_reco[None])[0])
    return DisplayEvent(process, "CMS (2011 data)", det.B, np.zeros(3), float("nan"), m, tracks, True,
                        source="CMS 2011 data", run=int(row["Run"]), event_number=int(row["Event"]),
                        muon_types=tuple(str(row[f"type{i}"]) for i in (1, 2)))


def pick_data_row(events, process: str, rng: np.random.Generator):
    """A random selected CMS event inside the display window of ``process`` (a pandas row)."""
    lo, hi = DATA_DISPLAY_WINDOWS[process]
    m = events["M_recomputed"] if "M_recomputed" in events else events["M"]
    cand = events[(m > lo) & (m < hi)]
    if not len(cand):
        raise LookupError(f"no CMS events with {lo} < M < {hi} GeV")
    return cand.iloc[int(rng.integers(len(cand)))]

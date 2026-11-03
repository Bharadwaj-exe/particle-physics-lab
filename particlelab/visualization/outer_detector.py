"""Schematic outer detector for the event displays: calorimeters, solenoid, return yoke, muon stations.

DISPLAY MODEL ONLY. Nothing in the analysis (simulation, fits, data comparison) uses it.
It continues each muon beyond the tracker so the displays can show what CMS shows:
muons are the only charged particles that cross the calorimeters, the coil and the iron
and reach the muon chambers; outside the solenoid the field points the other way, so the
track bends back (the "S" shape in the transverse view); and muons that are too soft lose
all their energy in the calorimeters or the iron and never reach the outer stations.

Geometry: dimensions rounded from the CMS detector description (CMS Collaboration,
JINST 3 (2008) S08004): ECAL, HCAL, coil, three iron layers of the barrel yoke with four
muon stations MB1-MB4 between them, and endcap disks with stations ME1-ME4.

Field (simplified): B along +z inside the coil (|z| < 6 m), falling linearly across the
coil winding; in the barrel iron, B_return = -0.5 B (about -1.9 T at 3.8 T); zero elsewhere.
The endcap return field (mostly radial) is not modelled, so endcap muons go straight
outside the coil.

Energy loss: continuous, at the minimum-ionising rate of each material (PDG values,
1.229 MeV cm2/g for PbWO4, 1.403 for brass as Cu, 1.615 for Al, 1.451 for Fe), with the
fraction of each layer that is absorber. Neglected: the rise of dE/dx at low and high
momentum, radiative losses (important above ~300 GeV in iron), energy-loss fluctuations
and multiple scattering. With these values a muon at eta = 0 loses about 1.6 GeV before
MB1 and about 3.3 GeV before MB4, close to the 2-4 GeV usually quoted for CMS; muons
below that stop in the calorimeters or the iron.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics.kinematics import M_MU
from ..units import C_TRACK

C_LIGHT = 0.299792458          # m / ns

# minimum-ionising dE/dx in GeV/m: (MeV cm2/g) x density (g/cm3) / 10
DEDX = {"PbWO4": 1.229 * 8.30 / 10, "brass": 1.403 * 8.53 / 10, "Al": 1.615 * 2.699 / 10, "Fe": 1.451 * 7.874 / 10}

COIL_R_IN, COIL_R_WINDING, COIL_R_OUT, COIL_HALF_Z = 2.95, 3.25, 3.80, 6.0
RETURN_FRACTION = -0.5          # field in the barrel iron relative to the central field
BARREL_HALF_Z = 6.6             # barrel yoke and barrel muon stations
WORLD_R, WORLD_Z = 7.4, 10.6    # muons are followed up to here


@dataclass(frozen=True)
class Layer:
    """An absorber volume: barrel (r_in..r_out, |z| < z_max) or endcap disk (z_in..z_out, r_in..r_out)."""
    name: str
    group: str                  # "calorimeters", "solenoid", "return yoke"
    kind: str                   # "barrel" or "endcap"
    r_in: float
    r_out: float
    z_in: float                 # barrel: 0; endcap: |z| where the disk starts
    z_out: float                # barrel: half length; endcap: |z| where it ends
    material: str
    fraction: float = 1.0       # absorber fraction of the volume
    field: str = "none"         # "none" or "return" (the barrel iron)

    @property
    def dedx(self) -> float:
        return DEDX[self.material] * self.fraction

    def contains(self, r: float, az: float) -> bool:
        if self.kind == "barrel":
            return self.r_in <= r < self.r_out and az < self.z_out
        return self.z_in <= az < self.z_out and self.r_in <= r < self.r_out


@dataclass(frozen=True)
class Station:
    """A muon station: barrel cylinder at radius ``pos`` (|z| < extent) or endcap disk at |z| = ``pos``."""
    name: str
    kind: str
    pos: float
    lo: float                   # barrel: 0 ; endcap: inner radius
    hi: float                   # barrel: half length ; endcap: outer radius


@dataclass(frozen=True)
class OuterDetector:
    layers: tuple
    stations: tuple

    def dedx(self, r: float, z: float) -> float:
        az = abs(z)
        for layer in self.layers:
            if layer.contains(r, az):
                return layer.dedx
        return 0.0

    def region(self, r: float, z: float) -> str:
        az = abs(z)
        for layer in self.layers:
            if layer.contains(r, az):
                return layer.name
        return "a gap"

    def bz(self, r: float, z: float, B: float) -> float:
        """Axial field (T) at (r, z) for a central field B."""
        az = abs(z)
        if az < COIL_HALF_Z:
            if r < COIL_R_IN:
                return B
            if r < COIL_R_WINDING:          # across the winding the field falls to zero
                return B * (COIL_R_WINDING - r) / (COIL_R_WINDING - COIL_R_IN)
        for layer in self.layers:
            if layer.field == "return" and layer.contains(r, az):
                return RETURN_FRACTION * B
        return 0.0


def cms_outer() -> OuterDetector:
    """The schematic CMS-like outer detector (see the module docstring for sources)."""
    L = Layer
    layers = (
        L("ECAL barrel", "calorimeters", "barrel", 1.29, 1.52, 0.0, 3.0, "PbWO4"),
        L("HCAL barrel", "calorimeters", "barrel", 1.77, 2.95, 0.0, 4.3, "brass", 0.80),
        L("ECAL endcap", "calorimeters", "endcap", 0.32, 1.71, 3.15, 3.37, "PbWO4"),
        L("HCAL endcap", "calorimeters", "endcap", 0.40, 2.95, 3.88, 5.68, "brass", 0.80),
        L("solenoid", "solenoid", "barrel", COIL_R_IN, COIL_R_OUT, 0.0, 6.3, "Al", 0.55),
        L("yoke barrel 1", "return yoke", "barrel", 4.20, 4.50, 0.0, BARREL_HALF_Z, "Fe", 1.0, "return"),
        L("yoke barrel 2", "return yoke", "barrel", 5.05, 5.68, 0.0, BARREL_HALF_Z, "Fe", 1.0, "return"),
        L("yoke barrel 3", "return yoke", "barrel", 6.10, 6.73, 0.0, BARREL_HALF_Z, "Fe", 1.0, "return"),
        L("yoke endcap 1", "return yoke", "endcap", 1.20, 7.00, 6.66, 7.26, "Fe"),
        L("yoke endcap 2", "return yoke", "endcap", 1.20, 7.00, 7.86, 8.46, "Fe"),
        L("yoke endcap 3", "return yoke", "endcap", 1.20, 7.00, 9.06, 9.36, "Fe"),
    )
    stations = (
        Station("MB1", "barrel", 4.02, 0.0, BARREL_HALF_Z), Station("MB2", "barrel", 4.87, 0.0, BARREL_HALF_Z),
        Station("MB3", "barrel", 5.90, 0.0, BARREL_HALF_Z), Station("MB4", "barrel", 7.00, 0.0, BARREL_HALF_Z),
        Station("ME1", "endcap", 6.30, 1.0, 7.0), Station("ME2", "endcap", 7.56, 1.0, 7.0),
        Station("ME3", "endcap", 8.76, 1.0, 7.0), Station("ME4", "endcap", 9.90, 1.0, 7.0),
    )
    return OuterDetector(layers, stations)


@dataclass
class OuterTrack:
    """A muon followed from the end of its tracker path through the outer detector."""
    path: np.ndarray                    # (N,3) m
    t: np.ndarray                       # (N,) ns
    p: np.ndarray                       # (N,) GeV, momentum along the path
    stations: list = field(default_factory=list)   # (name, point (3,), t ns, p GeV) in crossing order
    stopped_in: str | None = None       # region where the muon stopped, or None if it left the detector

    @property
    def station_names(self):
        return [s[0] for s in self.stations]

    @property
    def station_points(self):
        return np.array([s[1] for s in self.stations]).reshape(-1, 3)

    @property
    def station_t(self):
        return np.array([s[2] for s in self.stations], float)

    def summary(self) -> str:
        reached = ", ".join(self.station_names) or "no muon station"
        end = f"stopped in the {self.stopped_in}" if self.stopped_in else f"left the detector with p = {self.p[-1]:.1f} GeV"
        return f"reached {reached}; {end}"


def propagate(x0, direction, p0: float, q: int, B: float, outer: OuterDetector | None = None, t0: float = 0.0,
              step: float = 0.02, max_steps: int = 4000) -> OuterTrack:
    """Follow a muon from ``x0`` (m) with momentum ``p0`` (GeV) along ``direction`` through ``outer``.

    Each step of ``step`` metres rotates the transverse momentum in the local axial field
    (positive charges turn clockwise seen from +z for B along +z, as in the tracker
    simulation), removes dE/dx x step of energy and advances the flight time by step / (beta c).
    """
    outer = outer or cms_outer()
    x = np.asarray(x0, float).copy()
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    p = float(p0)
    t = float(t0)
    pts, ts, ps, stations = [x.copy()], [t], [p], []
    stopped = None
    r_prev, az_prev = np.hypot(x[0], x[1]), abs(x[2])
    k = float(np.sign(q))
    for _ in range(max_steps):
        r, z = np.hypot(x[0], x[1]), x[2]
        if r >= WORLD_R or abs(z) >= WORLD_Z:
            break
        # bend: the transverse direction turns by dphi = -q 0.3 B ds / p
        bz = outer.bz(r, z, B)
        if bz != 0.0:
            dphi = -k * C_TRACK * bz * step / p
            c, s = np.cos(dphi), np.sin(dphi)
            half_c, half_s = np.cos(dphi / 2), np.sin(dphi / 2)
            mid = np.array([u[0] * half_c - u[1] * half_s, u[0] * half_s + u[1] * half_c, u[2]])
            u = np.array([u[0] * c - u[1] * s, u[0] * s + u[1] * c, u[2]])
        else:
            mid = u
        e_old = np.hypot(p, M_MU)
        e_new = e_old - outer.dedx(r, z) * step
        beta = (p / e_old)
        x = x + step * mid
        if e_new <= np.hypot(0.1, M_MU):              # below 0.1 GeV/c: stopped
            t += step / (beta * C_LIGHT)
            stopped = outer.region(r, z)
            pts.append(x.copy()); ts.append(t); ps.append(0.0)
            break
        p_new = np.sqrt(e_new**2 - M_MU**2)
        t += step / (0.5 * (beta + p_new / e_new) * C_LIGHT)
        p = p_new
        r_new, az_new = np.hypot(x[0], x[1]), abs(x[2])
        for st in outer.stations:
            if st.kind == "barrel":
                crossed = r_prev < st.pos <= r_new and az_new < st.hi
                frac = (st.pos - r_prev) / (r_new - r_prev) if crossed else 0.0
            else:
                crossed = az_prev < st.pos <= az_new and st.lo <= r_new <= st.hi
                frac = (st.pos - az_prev) / (az_new - az_prev) if crossed else 0.0
            if crossed:
                pt_hit = pts[-1] + frac * (x - pts[-1])
                stations.append((st.name, pt_hit, ts[-1] + frac * (t - ts[-1]), p))
        pts.append(x.copy()); ts.append(t); ps.append(p)
        r_prev, az_prev = r_new, az_new
    return OuterTrack(np.array(pts), np.array(ts), np.array(ps), stations, stopped)


def continue_track(path, times, p_mag: float, q: int, B: float, outer: OuterDetector | None = None) -> OuterTrack:
    """Outer track starting where a tracker ``path`` (N,3) ends, in the direction of its last segment."""
    path = np.asarray(path, float)
    d = path[-1] - path[-2]
    return propagate(path[-1], d / np.linalg.norm(d), p_mag, q, B, outer, float(np.asarray(times)[-1]))


def rotate_arc_check(p, q, B, s):
    """Direction after transverse arc ``s`` on a uniform-field helix (used by the tests)."""
    p = np.asarray(p, float)
    pt = np.hypot(p[0], p[1])
    phi = np.arctan2(p[1], p[0]) - np.sign(q) * C_TRACK * B * s / pt
    return np.array([pt * np.cos(phi), pt * np.sin(phi), p[2]]) / np.linalg.norm(p)

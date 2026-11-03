"""Detector configurations.

`cms_inspired()` is an EDUCATIONAL model with the main features of the CMS
silicon tracker (3.8 T solenoid; pixel barrel + strip barrel layers; pixel,
inner and outer endcap disks). Layer positions and resolutions are rounded
values in the spirit of CMS Collaboration, JINST 3 (2008) S08004 and are NOT
the CMS geometry.

Material: the sensitive layers plus passive surfaces (beam pipe, cables, cooling,
support flanges) that scatter tracks but record no hits. Their thicknesses are
calibrated, subsystem by subsystem, to the published CMS material budget X0(η)
(JINST 9 (2014) P10009, Fig. 2), digitised in
`references/cms_tracker_material_x0.json`; see `calibration/fit_material.py`.
They are never adjusted to the dimuon data.

`uniform_barrel()` is the original reference model: 10 equally spaced
infinitely long layers, no material. It is kept because its momentum resolution
has a closed-form prediction (Gluckstern), which validates the fitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict

import numpy as np

from ..units import um, mm, cm

INF = float("inf")


@dataclass(frozen=True)
class BarrelLayer:
    name: str
    radius: float
    half_length: float
    sigma_rphi: float
    sigma_z: float          # inf for single-sided (mono) strip layers
    x_over_X0: float


@dataclass(frozen=True)
class EndcapDisk:
    name: str
    z: float                # |z| of the disk; mirrored at -z
    r_min: float
    r_max: float
    sigma_rphi: float
    sigma_r: float          # inf for mono strip disks
    x_over_X0: float


@dataclass(frozen=True)
class PassiveCylinder:
    """Scattering material on a cylinder of radius r for z_min <= |z| <= z_max (no hits)."""
    name: str
    radius: float
    z_min: float
    z_max: float
    x_over_X0: float


@dataclass(frozen=True)
class PassiveDisk:
    """Scattering material on a disk at ±z between r_min and r_max (no hits)."""
    name: str
    z: float
    r_min: float
    r_max: float
    x_over_X0: float


@dataclass(frozen=True)
class DetectorConfig:
    name: str
    B: float
    barrels: tuple
    disks: tuple = ()
    hit_efficiency: float = 1.0
    multiple_scattering: bool = True
    smear: bool = True
    vertex_sigma_z: float = 0.0       # luminous-region length (Gaussian sigma)
    beamspot_constraint: float = 0.0  # sigma of the (0,0) pseudo-hit in the fit; 0 = off
    min_hits: int = 6
    eta_max: float = 2.4              # analysis cuts applied to reconstructed muons
    pt_min: float = 3.0
    passive: tuple = ()               # PassiveCylinder / PassiveDisk: scattering only

    # convenience views ------------------------------------------------------
    @property
    def radii(self) -> np.ndarray:
        return np.array([b.radius for b in self.barrels])

    @property
    def lever_arm(self) -> float:
        return float(self.radii.max() - self.radii.min())

    @property
    def sigma_rphi(self) -> float:
        return float(np.mean([b.sigma_rphi for b in self.barrels]))

    def ideal(self) -> "DetectorConfig":
        """Same geometry with no smearing, no multiple scattering and full efficiency."""
        return replace(self, smear=False, multiple_scattering=False, hit_efficiency=1.0)

    def with_(self, **kw) -> "DetectorConfig":
        return replace(self, **kw)

    def scaled_resolution(self, factor: float) -> "DetectorConfig":
        return replace(
            self,
            barrels=tuple(replace(b, sigma_rphi=b.sigma_rphi * factor, sigma_z=b.sigma_z * factor) for b in self.barrels),
            disks=tuple(replace(d, sigma_rphi=d.sigma_rphi * factor, sigma_r=d.sigma_r * factor) for d in self.disks),
        )

    def scaled_material(self, factor: float) -> "DetectorConfig":
        return replace(
            self,
            barrels=tuple(replace(b, x_over_X0=b.x_over_X0 * factor) for b in self.barrels),
            disks=tuple(replace(d, x_over_X0=d.x_over_X0 * factor) for d in self.disks),
            passive=tuple(replace(q, x_over_X0=q.x_over_X0 * factor) for q in self.passive),
        )

    def material_at_eta0(self) -> float:
        return float(sum(material_along(self, 0.0).values()))

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v if v != INF else "inf") for k, v in d.items()}


SUBSYSTEMS = ("beam pipe", "pixel", "TIB/TID", "TOB", "TEC")


def subsystem(name: str) -> str:
    """Subsystem of a surface from its name prefix (the grouping of the CMS figure)."""
    for prefix, sub_ in (("BP", "beam pipe"), ("PX", "pixel"), ("FPIX", "pixel"), ("PIX", "pixel"),
                         ("TIB", "TIB/TID"), ("TID", "TIB/TID"), ("TOB", "TOB"), ("TEC", "TEC")):
        if name.upper().startswith(prefix):
            return sub_
    return "other"


def material_along(det: DetectorConfig, eta: float, z0: float = 0.0) -> dict:
    """x/X0 met by a straight track from (0, 0, z0) at pseudorapidity η, by subsystem.

    Cylinders contribute x·cosh η and disks x·|coth η| (the path length through a
    thin surface at the crossing angle)."""
    out = {k: 0.0 for k in SUBSYSTEMS}
    sh, ch = np.sinh(eta), np.cosh(eta)

    def add(name, x):
        k = subsystem(name)
        out[k] = out.get(k, 0.0) + x

    for b in det.barrels:
        if abs(z0 + b.radius * sh) <= b.half_length:
            add(b.name, b.x_over_X0 * ch)
    for q in det.passive:
        if isinstance(q, PassiveCylinder) and q.z_min <= abs(z0 + q.radius * sh) <= q.z_max:
            add(q.name, q.x_over_X0 * ch)
    if abs(sh) > 1e-12:
        disks = [(d.z, d.r_min, d.r_max, d.x_over_X0, d.name) for d in det.disks]
        disks += [(q.z, q.r_min, q.r_max, q.x_over_X0, q.name) for q in det.passive if isinstance(q, PassiveDisk)]
        for zd, rmin, rmax, x, name in disks:
            r = (np.copysign(zd, sh) - z0) / sh
            if r > 0 and rmin <= r <= rmax:
                add(name, x * ch / abs(sh))
    return out


# Thickness per surface (x/X0 at normal incidence) of each material group in
# cms_inspired(). Fitted by calibration/fit_material.py to the CMS X0(η) curve
# (references/cms_tracker_material_x0.json), one subsystem at a time.
CMS_MATERIAL = {
    "PXB": 0.0227, "FPIX": 0.01, "PIX_flange_inner": 0.1621, "PIX_flange_outer": 0.049, "PIX_service_cyl": 0.0398,
    "TIB": 0.0399, "TID": 0.0426, "TIB_end": 0.2188, "TIB_service_inner": 0.0345,
    "TIB_service_outer": 0.1915, "TID_end": 0.0458,
    "TOB": 0.0244, "TOB_service_inner": 0.1062, "TOB_service_outer": 0.171, "TOB_end": 0.0305,
    "TEC": 0.0519,
}
BEAM_PIPE_X0 = 0.08 / 35.28   # 0.8 mm of beryllium (X0 = 35.28 cm); fixed, not fitted


def cms_inspired(material: dict | None = None) -> DetectorConfig:
    """``material`` overrides entries of CMS_MATERIAL (and "BP" for the beam pipe)."""
    t = dict(CMS_MATERIAL, BP=BEAM_PIPE_X0)
    t.update(material or {})
    pix = [BarrelLayer(f"PXB{i+1}", r, 26.5 * cm, 10 * um, 20 * um, t["PXB"])
           for i, r in enumerate([4.4 * cm, 7.3 * cm, 10.2 * cm])]
    tib = [BarrelLayer(f"TIB{i+1}", r, 70 * cm, 30 * um, (0.5 * mm if i < 2 else INF), t["TIB"])
           for i, r in enumerate([25.5 * cm, 33.9 * cm, 41.8 * cm, 49.8 * cm])]
    tob = [BarrelLayer(f"TOB{i+1}", r, 118 * cm, 45 * um, (0.5 * mm if i < 2 else INF), t["TOB"])
           for i, r in enumerate([55.5 * cm, 63.5 * cm, 71.5 * cm, 79.5 * cm, 89.5 * cm, 108 * cm])]
    fpix = [EndcapDisk(f"FPIX{i+1}", z, 6 * cm, 15 * cm, 15 * um, 20 * um, t["FPIX"])
            for i, z in enumerate([34.5 * cm, 46.5 * cm])]
    tid = [EndcapDisk(f"TID{i+1}", z, 20 * cm, 50 * cm, 35 * um, 1 * mm, t["TID"])
           for i, z in enumerate([80 * cm, 90 * cm, 100 * cm])]
    tec = [EndcapDisk(f"TEC{i+1}", z, 22 * cm, 113 * cm, 40 * um, (1 * mm if i % 2 == 0 else INF), t["TEC"])
           for i, z in enumerate(np.linspace(124 * cm, 272 * cm, 9))]
    passive = [
        PassiveCylinder("BP beam pipe", 2.9 * cm, 0.0, 3.0, t["BP"]),
        # pixel: barrel end flange and cabling, and the service cylinder towards the endcap
        PassiveDisk("PIX flange inner", 28 * cm, 4 * cm, 12 * cm, t["PIX_flange_inner"]),
        PassiveDisk("PIX flange outer", 28 * cm, 12 * cm, 22 * cm, t["PIX_flange_outer"]),
        PassiveCylinder("PIX service cylinder", 18 * cm, 50 * cm, 2.8, t["PIX_service_cyl"]),
        # inner strips: TIB end services, TIB/TID service cylinder, TID back services
        PassiveDisk("TIB end services", 72 * cm, 22 * cm, 52 * cm, t["TIB_end"]),
        # cables run along z and accumulate towards the barrel ends, so the service
        # cylinders start away from z = 0 and come in two segments
        PassiveCylinder("TIB services, inner segment", 53 * cm, 12 * cm, 60 * cm, t["TIB_service_inner"]),
        PassiveCylinder("TIB services, outer segment", 53 * cm, 60 * cm, 110 * cm, t["TIB_service_outer"]),
        PassiveDisk("TID back services", 110 * cm, 20 * cm, 52 * cm, t["TID_end"]),
        # outer barrel: rod services (between TOB4 and TOB5), end flange and cabling
        PassiveCylinder("TOB services, inner segment", 85 * cm, 20 * cm, 70 * cm, t["TOB_service_inner"]),
        PassiveCylinder("TOB services, outer segment", 85 * cm, 70 * cm, 118 * cm, t["TOB_service_outer"]),
        PassiveDisk("TOB end flange", 120 * cm, 55 * cm, 112 * cm, t["TOB_end"]),
    ]
    return DetectorConfig(
        name="CMS-inspired tracker (educational model)",
        B=3.8, barrels=tuple(pix + tib + tob), disks=tuple(fpix + tid + tec),
        hit_efficiency=0.99, multiple_scattering=True, smear=True,
        vertex_sigma_z=6 * cm, beamspot_constraint=20 * um, min_hits=8,
        passive=tuple(q for q in passive if q.x_over_X0 > 0),
    )


def uniform_barrel(B: float = 3.8, n_layers: int = 10, r_min: float = 0.2, r_max: float = 1.1,
                   sigma: float = 20 * um, sigma_z: float = 50 * um) -> DetectorConfig:
    layers = tuple(BarrelLayer(f"L{i+1}", r, INF, sigma, sigma_z, 0.0)
                   for i, r in enumerate(np.linspace(r_min, r_max, n_layers)))
    return DetectorConfig(name="uniform barrel (reference)", B=B, barrels=layers,
                          multiple_scattering=False, min_hits=3)

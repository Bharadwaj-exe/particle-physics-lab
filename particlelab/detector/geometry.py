"""Detector configurations.

`cms_inspired()` is an EDUCATIONAL model with the main features of the CMS
silicon tracker (3.8 T solenoid; pixel barrel + strip barrel layers; pixel,
inner and outer endcap disks; roughly 0.4 X0 of material at η = 0). Layer
positions and resolutions are rounded values in the spirit of
CMS Collaboration, JINST 3 (2008) S08004 and are NOT the CMS geometry.

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
        )

    def material_at_eta0(self) -> float:
        return float(sum(b.x_over_X0 for b in self.barrels))

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v if v != INF else "inf") for k, v in d.items()}


def cms_inspired() -> DetectorConfig:
    pix = [BarrelLayer(f"PXB{i+1}", r, 26.5 * cm, 10 * um, 20 * um, 0.02)
           for i, r in enumerate([4.4 * cm, 7.3 * cm, 10.2 * cm])]
    tib = [BarrelLayer(f"TIB{i+1}", r, 70 * cm, 30 * um, (0.5 * mm if i < 2 else INF), 0.03)
           for i, r in enumerate([25.5 * cm, 33.9 * cm, 41.8 * cm, 49.8 * cm])]
    tob = [BarrelLayer(f"TOB{i+1}", r, 118 * cm, 45 * um, (0.5 * mm if i < 2 else INF), 0.03)
           for i, r in enumerate([55.5 * cm, 63.5 * cm, 71.5 * cm, 79.5 * cm, 89.5 * cm, 108 * cm])]
    fpix = [EndcapDisk(f"FPIX{i+1}", z, 6 * cm, 15 * cm, 15 * um, 20 * um, 0.03)
            for i, z in enumerate([34.5 * cm, 46.5 * cm])]
    tid = [EndcapDisk(f"TID{i+1}", z, 20 * cm, 50 * cm, 35 * um, 1 * mm, 0.04)
           for i, z in enumerate([80 * cm, 90 * cm, 100 * cm])]
    tec = [EndcapDisk(f"TEC{i+1}", z, 22 * cm, 113 * cm, 40 * um, (1 * mm if i % 2 == 0 else INF), 0.05)
           for i, z in enumerate(np.linspace(124 * cm, 272 * cm, 9))]
    return DetectorConfig(
        name="CMS-inspired tracker (educational model)",
        B=3.8, barrels=tuple(pix + tib + tob), disks=tuple(fpix + tid + tec),
        hit_efficiency=0.99, multiple_scattering=True, smear=True,
        vertex_sigma_z=6 * cm, beamspot_constraint=20 * um, min_hits=8,
    )


def uniform_barrel(B: float = 3.8, n_layers: int = 10, r_min: float = 0.2, r_max: float = 1.1,
                   sigma: float = 20 * um, sigma_z: float = 50 * um) -> DetectorConfig:
    layers = tuple(BarrelLayer(f"L{i+1}", r, INF, sigma, sigma_z, 0.0)
                   for i, r in enumerate(np.linspace(r_min, r_max, n_layers)))
    return DetectorConfig(name="uniform barrel (reference)", B=B, barrels=layers,
                          multiple_scattering=False, min_hits=3)

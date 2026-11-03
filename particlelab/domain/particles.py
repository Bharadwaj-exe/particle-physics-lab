"""Particle-property database loaded from ``particles_pdg.json``.

The JSON file is the single source of truth for masses, widths, branching
fractions and quantum numbers. Antiparticles are derived (quantum numbers
negated, same mass).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping

DB_FILE = Path(__file__).with_name("particles_pdg.json")


@dataclass(frozen=True)
class Particle:
    id: str
    label: str
    mass: float
    charge: int
    baryon: int
    Le: int
    Lmu: int
    Ltau: int
    spin: float = 0.0
    mass_err: float = 0.0
    width: float = 0.0
    width_err: float = 0.0
    br_mumu: float | None = None
    br_mumu_err: float | None = None
    pdg_code: int = 0
    verified: bool = False
    aliases: tuple = field(default_factory=tuple)

    @property
    def lepton_numbers(self) -> tuple[int, int, int]:
        return self.Le, self.Lmu, self.Ltau


@dataclass(frozen=True)
class ParticleDB:
    particles: Mapping[str, Particle]
    source: str
    retrieved: str

    def get(self, name: str) -> Particle:
        key = self._aliases().get(name.lower(), name)
        if key not in self.particles:
            raise KeyError(f"Unknown particle '{name}'. Known: {', '.join(sorted(self.particles))}")
        return self.particles[key]

    def _aliases(self) -> dict[str, str]:
        out = {}
        for p in self.particles.values():
            out[p.id.lower()] = p.id
            for a in p.aliases:
                out[a.lower()] = p.id
        return out

    def __contains__(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except KeyError:
            return False


_FIELDS = {f for f in Particle.__dataclass_fields__}


@lru_cache(maxsize=None)
def load(path: str | None = None) -> ParticleDB:
    raw = json.loads(Path(path or DB_FILE).read_text(encoding="utf-8"))
    parts: dict[str, Particle] = {}
    for d in raw["particles"]:
        kw = {k: v for k, v in d.items() if k in _FIELDS}
        kw["aliases"] = tuple(d.get("aliases", ()))
        p = Particle(**kw)
        parts[p.id] = p
        anti = d.get("antiparticle")
        if anti:
            parts[anti] = Particle(
                id=anti, label=anti, mass=p.mass, mass_err=p.mass_err, charge=-p.charge,
                baryon=-p.baryon, Le=-p.Le, Lmu=-p.Lmu, Ltau=-p.Ltau, spin=p.spin,
                pdg_code=-p.pdg_code, verified=p.verified,
                aliases=(f"anti-{p.id}",) if not anti.startswith("anti-") else (),
            )
    # common spellings for antibaryons
    ap = parts.get("anti-p")
    if ap:
        parts["anti-p"] = Particle(**{**ap.__dict__, "aliases": ("pbar", "p~")})
    an = parts.get("anti-n")
    if an:
        parts["anti-n"] = Particle(**{**an.__dict__, "aliases": ("nbar",)})
    return ParticleDB(parts, raw["source"], raw["retrieved"])


def muon_mass() -> float:
    return load().get("mu-").mass


RESONANCES = ("J/psi", "Upsilon(1S)", "Z")

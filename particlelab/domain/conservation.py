"""Conservation-law checks for decays written as 'A -> b c ...'.

Each law is classified:

* exact       — no violation ever observed and built into the Standard Model
                (electric charge; B - L).
* accidental  — conserved by all Standard Model perturbative processes and never
                observed to be violated (baryon number B, total lepton number L).
* approximate — individual lepton family numbers Le, Lmu, Ltau. Neutrino
                oscillations violate them, but charged-lepton flavour violation
                (e.g. mu -> e gamma) has never been observed.
* kinematic   — energy-momentum conservation: a parent at rest must be heavier
                than the sum of its decay products, and a one-body decay is
                impossible.

Passing every check does NOT prove a decay occurs: angular momentum, parity,
C, strangeness and dynamics are not checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .particles import load

LAWS = [
    # (key, description, attribute getter, kind)
    ("Q", "charge Q", lambda p: p.charge, "exact"),
    ("B-L", "B − L", lambda p: p.baryon - sum(p.lepton_numbers), "exact"),
    ("B", "baryon number B", lambda p: p.baryon, "accidental"),
    ("L", "total lepton number L", lambda p: sum(p.lepton_numbers), "accidental"),
    ("Le", "electron number Le", lambda p: p.Le, "approximate"),
    ("Lmu", "muon number Lμ", lambda p: p.Lmu, "approximate"),
    ("Ltau", "tau number Lτ", lambda p: p.Ltau, "approximate"),
]


@dataclass(frozen=True)
class Check:
    law: str
    kind: str
    before: float
    after: float
    passed: bool
    note: str = ""


@dataclass
class Verdict:
    decay: str
    parent: str
    daughters: list
    checks: list = field(default_factory=list)
    parent_mass: float = 0.0
    daughter_mass: float = 0.0

    @property
    def failed(self) -> list:
        return [c for c in self.checks if not c.passed]

    @property
    def allowed(self) -> bool:
        """True only if every checked law holds (necessary, not sufficient)."""
        return not self.failed

    @property
    def reasons(self) -> list:
        return [c.note for c in self.failed]

    @property
    def category(self) -> str:
        if self.allowed:
            return "passes all checks (necessary conditions only)"
        kinds = {c.kind for c in self.failed}
        if "exact" in kinds:
            return "forbidden by an exact conservation law"
        if "kinematic" in kinds:
            return "kinematically forbidden"
        if "accidental" in kinds:
            return "violates an accidental symmetry (B or L): never observed"
        return "violates lepton flavour only: not observed for charged leptons"

    def explain(self) -> str:
        lines = [f"{self.decay}: {self.category}"]
        for c in self.checks:
            mark = "ok " if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.law:<24s} ({c.kind}) {c.note if not c.passed else ''}".rstrip())
        return "\n".join(lines)


def parse(decay: str) -> tuple[str, list[str]]:
    if "->" not in decay:
        raise ValueError("Write a decay as 'parent -> daughter1 daughter2 ...'")
    lhs, rhs = decay.split("->", 1)
    parent = lhs.split()
    if len(parent) != 1:
        raise ValueError("Exactly one parent particle is required")
    db = load()
    daughters = [db.get(t).id for t in rhs.split()]
    if not daughters:
        raise ValueError("At least one daughter is required")
    return db.get(parent[0]).id, daughters


def check(decay: str) -> Verdict:
    db = load()
    parent, daughters = parse(decay)
    P, Ds = db.get(parent), [db.get(d) for d in daughters]
    v = Verdict(decay.strip(), parent, daughters, parent_mass=P.mass,
                daughter_mass=sum(d.mass for d in Ds))
    for key, desc, get, kind in LAWS:
        b, a = get(P), sum(get(d) for d in Ds)
        v.checks.append(Check(desc, kind, b, a, b == a, f"{desc} not conserved ({b:+d} → {a:+d})"))
    if len(Ds) == 1:
        v.checks.append(Check("energy-momentum", "kinematic", P.mass, Ds[0].mass, False,
                              "a single-particle decay cannot conserve both energy and momentum"))
    else:
        ok = P.mass > v.daughter_mass
        v.checks.append(Check("mass threshold", "kinematic", P.mass, v.daughter_mass, ok,
                              f"not enough mass: {P.mass:.4g} GeV ≤ {v.daughter_mass:.4g} GeV"))
    return v


def check_four_momentum(parent_p4, daughter_p4s, rtol=1e-9) -> Check:
    """Numerical four-momentum conservation for explicit four-vectors (E, px, py, pz)."""
    parent = np.asarray(parent_p4, float)
    total = np.sum([np.asarray(d, float) for d in daughter_p4s], axis=0)
    scale = max(np.max(np.abs(parent)), 1e-12)
    dev = float(np.max(np.abs(total - parent)) / scale)
    return Check("four-momentum", "kinematic", 0.0, dev, dev <= rtol,
                 f"four-momentum mismatch, relative deviation {dev:.2e}")


EXAMPLES = [
    "Z -> mu+ mu-", "J/psi -> mu+ mu-", "Upsilon(1S) -> mu+ mu-", "J/psi -> p anti-p",
    "Z -> mu+ e-", "J/psi -> mu+ mu+", "J/psi -> tau+ tau-", "n -> p e- anti-nu_e",
    "p -> e+ pi0", "mu- -> e- gamma", "mu- -> e- anti-nu_e nu_mu",
]

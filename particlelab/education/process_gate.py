"""Process gate: only a process that passes the conservation-law checks, and that
the simulation implements, can be sent to the generator and detector.

This connects the conservation checker (domain) to the simulation: the learner
types or picks a decay; a forbidden one is refused with the law it breaks, an
allowed but unimplemented one is explained, and an allowed implemented one is
simulated.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.conservation import Verdict, check
from ..domain.particles import RESONANCES

FINAL_STATE = sorted(["mu+", "mu-"])


@dataclass(frozen=True)
class GateResult:
    decay: str
    allowed: bool            # passes every conservation check
    simulable: bool          # allowed and implemented
    process: str | None      # key for the simulation (e.g. "Z") when simulable
    message: str
    verdict: Verdict | None = None


def gate(decay: str) -> GateResult:
    try:
        v = check(decay)
    except (KeyError, ValueError) as e:
        return GateResult(decay, False, False, None, f"Cannot read this decay: {e}")
    if not v.allowed:
        laws = "; ".join(v.reasons)
        return GateResult(decay, False, False, None,
                          f"Not simulated: {v.category}. {laws}.", v)
    if v.parent in RESONANCES and sorted(v.daughters) == FINAL_STATE:
        return GateResult(decay, True, True, v.parent,
                          "Passes charge, B − L, B, L, Le, Lμ, Lτ and the mass threshold; simulated with the "
                          "generator, the tracker and the fit.", v)
    return GateResult(decay, True, False, None,
                      "Passes the conservation checks, but only J/ψ, Υ(1S) and Z → μ⁺μ⁻ are implemented "
                      "in the simulation.", v)

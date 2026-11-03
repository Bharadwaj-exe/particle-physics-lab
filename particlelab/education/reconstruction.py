"""Guided invariant-mass reconstruction done by the learner, step by step.

The tool gives only what a tracker measures for each muon:

* the chord L from the beam line to the last recorded hit (transverse plane) and
  the sagitta s of the fitted track over that chord,
* the charge (from the bending direction), the azimuth φ at the vertex and the
  pseudorapidity η,
* the field B and the muon mass.

The learner then computes, for each muon,

    R  = L²/(8s) + s/2                  (circle geometry)
    pT = 0.299792458 · B[T] · R[m]      (GeV)
    E  = √(pT² cosh²η + m_μ²)

and finally the pair mass

    m² = (E₁ + E₂)² − |p⃗₁ + p⃗₂|²,   p⃗ = pT (cos φ, sin φ, sinh η).

Every answer is compared with the tool's own value. Wrong answers are matched
against the usual slips (forgetting 0.3, R in cm, using pT instead of p, adding
energies only, ignoring directions, degrees instead of radians …) so the feedback
names the mistake instead of only saying "wrong". The exact chain reproduces the
reconstructed mass that the simulation itself reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.kinematics import M_MU
from ..units import C_TRACK

REL_TOL = 0.005          # 0.5 %: generous enough for hand rounding to 3–4 digits
STEPS = ("R", "pT", "E", "mass")


@dataclass(frozen=True)
class MeasuredTrack:
    label: str        # "μ⁻" / "μ⁺"
    charge: int
    chord: float      # m, beam line → last recorded hit, transverse plane
    sagitta: float    # m, of the fitted circle over that chord
    phi: float        # rad, azimuth of the momentum at the vertex
    eta: float

    @property
    def radius(self) -> float:
        return self.chord**2 / (8 * self.sagitta) + self.sagitta / 2


@dataclass(frozen=True)
class Feedback:
    ok: bool
    message: str
    expected: float | None = None   # revealed only when the answer is right


def _close(a, b, tol=REL_TOL):
    return np.isfinite(a) and abs(a - b) <= tol * max(abs(b), 1e-12)


def _sagitta(R, chord):
    return R - np.sqrt(max(R**2 - chord**2 / 4, 0.0))


@dataclass
class Exercise:
    B: float
    tracks: tuple
    m_mu: float = M_MU

    # ---- the tool's own answers -------------------------------------------
    def radius(self, i): return self.tracks[i].radius
    def pt(self, i): return C_TRACK * self.B * self.radius(i)

    def p3(self, i):
        t = self.tracks[i]
        pt = self.pt(i)
        return np.array([pt * np.cos(t.phi), pt * np.sin(t.phi), pt * np.sinh(t.eta)])

    def energy(self, i):
        return float(np.sqrt(self.p3(i) @ self.p3(i) + self.m_mu**2))

    def mass(self):
        E = self.energy(0) + self.energy(1)
        P = self.p3(0) + self.p3(1)
        return float(np.sqrt(max(E**2 - P @ P, 0.0)))

    def expected(self) -> dict:
        return {"R": [self.radius(i) for i in (0, 1)], "pT": [self.pt(i) for i in (0, 1)],
                "E": [self.energy(i) for i in (0, 1)], "mass": self.mass()}

    def worked_solution(self) -> list:
        """Human-readable worked solution, one line per step."""
        out = []
        for i, t in enumerate(self.tracks):
            out.append(f"{t.label}: R = L²/(8s) + s/2 = {t.chord:.4f}²/(8·{t.sagitta:.5f}) + {t.sagitta:.5f}/2 "
                       f"= {self.radius(i):.4f} m")
            out.append(f"{t.label}: pT = 0.2998 · {self.B:g} T · {self.radius(i):.4f} m = {self.pt(i):.4f} GeV")
            p = self.p3(i)
            out.append(f"{t.label}: p = pT (cos φ, sin φ, sinh η) = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) GeV, "
                       f"E = √(|p|² + m_μ²) = {self.energy(i):.4f} GeV")
        E = self.energy(0) + self.energy(1)
        P = self.p3(0) + self.p3(1)
        out.append(f"pair: E₁+E₂ = {E:.4f} GeV, |p⃗₁+p⃗₂| = {np.linalg.norm(P):.4f} GeV, "
                   f"m = √(E² − |p⃗|²) = {self.mass():.4f} GeV")
        return out

    # ---- checking the learner ---------------------------------------------
    def check(self, step: str, value: float, track: int | None = None) -> Feedback:
        if step not in STEPS:
            raise ValueError(f"unknown step {step!r}; choose from {STEPS}")
        if step != "mass" and track not in (0, 1):
            raise ValueError("this step needs track = 0 or 1")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return Feedback(False, "Enter a number.")
        return getattr(self, f"_check_{step if step != 'mass' else 'mass'}")(value, track)

    def _check_R(self, v, i):
        t, R = self.tracks[i], self.radius(i)
        if _close(v, R):
            return Feedback(True, f"Correct: R = {R:.4f} m.", R)
        if _close(v, t.chord**2 / (8 * t.sagitta)):
            return Feedback(False, "Almost: you used R ≈ L²/(8s). Add the small correction s/2.")
        if _close(v, 100 * R):
            return Feedback(False, "Check units: R should be in metres, not centimetres.")
        if _close(v, t.chord / 2):
            return Feedback(False, "L/2 is half the chord, not the radius. Use R = L²/(8s) + s/2.")
        return Feedback(False, "Not yet. Use the circle relation R = L²/(8s) + s/2 with L and s in metres.")

    def _check_pT(self, v, i):
        pt, R = self.pt(i), self.radius(i)
        if _close(v, pt, 0.001):
            return Feedback(True, f"Correct: pT = {pt:.4f} GeV.", pt)
        if _close(v, 0.3 * self.B * R, 0.0005):
            return Feedback(True, f"Correct to 0.07 % (0.3 instead of 0.2998): pT = {pt:.4f} GeV.", pt)
        if _close(v, self.B * R):
            return Feedback(False, "You left out the factor 0.2998 in pT [GeV] = 0.2998 · B[T] · R[m].")
        if _close(v, 0.2998 * self.B * R * 100) or _close(v, 0.3 * self.B * R * 100):
            return Feedback(False, "R must be in metres: your value is 100 times too large.")
        if _close(v, pt / self.B) or _close(v, 0.2998 * R):
            return Feedback(False, "Multiply by the field B as well.")
        return Feedback(False, "Not yet. pT [GeV] = 0.2998 · B [T] · R [m].")

    def _check_E(self, v, i):
        t, pt, E = self.tracks[i], self.pt(i), self.energy(i)
        if _close(v, E, 0.002):
            return Feedback(True, f"Correct: E = {E:.4f} GeV.", E)
        if _close(v, pt, 0.002) or _close(v, np.hypot(pt, self.m_mu), 0.002):
            return Feedback(False, "That is the transverse part only. Use the full momentum p = pT·cosh η.")
        if _close(v, abs(pt * np.sinh(t.eta)), 0.002):
            return Feedback(False, "That is |pz| = pT·sinh η. The total momentum is p = pT·cosh η.")
        if _close(v, pt * np.cosh(t.eta) + self.m_mu, 0.002):
            return Feedback(False, "Energies do not add like that: E = √(p² + m²), not p + m.")
        return Feedback(False, "Not yet. E = √(p² + m_μ²) with p = pT·cosh η.")

    def _check_mass(self, v, _):
        m = self.mass()
        if _close(v, m):
            return Feedback(True, f"Correct: m(μ⁺μ⁻) = {m:.4f} GeV.", m)
        E1, E2 = self.energy(0), self.energy(1)
        p1, p2 = self.p3(0), self.p3(1)
        if _close(v, E1 + E2):
            return Feedback(False, "You added the energies only. Subtract the total momentum: m² = (E₁+E₂)² − |p⃗₁+p⃗₂|².")
        if _close(v, m**2):
            return Feedback(False, "That is m². Take the square root.")
        scalar = (E1 + E2) ** 2 - (np.linalg.norm(p1) + np.linalg.norm(p2)) ** 2
        if scalar > 0 and _close(v, np.sqrt(scalar)):
            return Feedback(False, "Momenta are vectors: add the components before squaring, not the magnitudes.")
        pt1, pt2 = p1.copy(), p2.copy()
        pt1[2] = pt2[2] = 0
        ET = np.hypot(np.linalg.norm(pt1), self.m_mu) + np.hypot(np.linalg.norm(pt2), self.m_mu)
        mt2 = ET**2 - (pt1 + pt2) @ (pt1 + pt2)
        if mt2 > 0 and _close(v, np.sqrt(mt2)):
            return Feedback(False, "You dropped the z components. Include pz = pT·sinh η in both E and p⃗.")
        deg = self._mass_with(lambda t: np.degrees(t.phi))
        if deg is not None and _close(v, deg):
            return Feedback(False, "φ was used in degrees where radians were needed (or the reverse).")
        return Feedback(False, "Not yet. m² = (E₁+E₂)² − (px₁+px₂)² − (py₁+py₂)² − (pz₁+pz₂)².")

    def _mass_with(self, phi_fn):
        try:
            ps, Es = [], []
            for i, t in enumerate(self.tracks):
                ph, pt = phi_fn(t), self.pt(i)
                p = np.array([pt * np.cos(ph), pt * np.sin(ph), pt * np.sinh(t.eta)])
                ps.append(p); Es.append(np.sqrt(p @ p + self.m_mu**2))
            P = ps[0] + ps[1]
            m2 = (Es[0] + Es[1]) ** 2 - P @ P
            return float(np.sqrt(m2)) if m2 > 0 else None
        except FloatingPointError:
            return None


def from_momenta(p_minus, p_plus, B, last_hit_radius=(1.0, 1.0)) -> Exercise:
    """Exercise for two reconstructed momenta (GeV) in a field B (T). ``last_hit_radius``
    is the transverse distance of each muon's last recorded hit (the chord)."""
    tracks = []
    for label, q, p, L in (("μ⁻", -1, p_minus, last_hit_radius[0]), ("μ⁺", +1, p_plus, last_hit_radius[1])):
        p = np.asarray(p, float)
        pt = float(np.hypot(p[0], p[1]))
        R = pt / (C_TRACK * B)
        L = float(min(L, 1.999 * R))
        tracks.append(MeasuredTrack(label, q, L, float(_sagitta(R, L)), float(np.arctan2(p[1], p[0])),
                                    float(np.arcsinh(p[2] / pt))))
    return Exercise(B, tuple(tracks))


def from_display_event(ev) -> Exercise:
    """Exercise from a visualization.DisplayEvent: the learner sees only measured quantities."""
    by_label = {t.label: t for t in ev.tracks}
    tm, tp = by_label["μ⁻"], by_label["μ⁺"]
    chord = []
    for t in (tm, tp):
        # simulated events: last recorded hit; CMS data events: last computed layer crossing
        h = np.asarray(t.hits if len(t.hits) else getattr(t, "crossings", t.hits))
        chord.append(float(np.hypot(h[:, 0], h[:, 1]).max()) if len(h) else 1.0)
    ex = from_momenta(tm.p_reco, tp.p_reco, ev.B, tuple(chord))
    # charges as reconstructed (normally −1, +1)
    return Exercise(ex.B, (MeasuredTrack(tm.label, tm.charge_reco, *_rest(ex.tracks[0])),
                           MeasuredTrack(tp.label, tp.charge_reco, *_rest(ex.tracks[1]))))


def _rest(t: MeasuredTrack):
    return t.chord, t.sagitta, t.phi, t.eta

"""Hit simulation: exact helix crossings, multiple scattering, smearing, efficiency.

Coordinates in metres, momenta in GeV, field in tesla. Positive charges curve
clockwise seen from +z. Tracks start at (0, 0, z_vertex).

Multiple scattering uses the Highland formula in the Lynch–Dahl form (PDG,
"Passage of particles through matter", Eq. 34.16) and the standard linearised
random-walk approximation: a kick of angle θ_i at path length l_i displaces a
later hit at l_j by θ_i (l_j − l_i) perpendicular to the track.

The PDG warns that adding per-layer θ0 values in quadrature is "systematically
too small" and recommends applying the formula once to the combined scatterer.
So the logarithmic factor uses the total thickness X = Σ x_i/X0 met by the track,
and each surface gets the share θ_i² = (13.6 MeV/βcp)² (x_i/X0) [1 + 0.038 ln(X/β²)]²,
whose sum is exactly θ0² of the combined scatterer.

Passive surfaces (beam pipe, services) scatter the track but record no hits. Changes of the
track direction caused by the kicks are neglected in the geometry (second order).
Mean energy loss is not simulated (see docs/physics/DETECTOR_MODEL.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import C_TRACK, HIGHLAND_E0
from ..physics.kinematics import M_MU
from .geometry import INF, DetectorConfig, PassiveCylinder


@dataclass
class Hits:
    """Per-track hit arrays of shape (n, S), sorted by transverse arc length s."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    s: np.ndarray
    crossed: np.ndarray      # geometric crossing
    detected: np.ndarray     # crossed and recorded (efficiency)
    sigma_rphi: np.ndarray
    sigma_long: np.ndarray   # sigma_z (barrel) or sigma_r (disk)
    is_disk: np.ndarray
    surface: np.ndarray      # index into the detector's surface list

    @property
    def n_detected(self) -> np.ndarray:
        return self.detected.sum(1)


def helix_at(s, R, k, phi0, cot_theta, z0=0.0):
    """Position on a helix starting at (0, 0, z0) after transverse arc length s."""
    xc, yc = k * R * np.sin(phi0), -k * R * np.cos(phi0)
    ph = phi0 - k * s / R
    return xc - k * R * np.sin(ph), yc + k * R * np.cos(ph), z0 + s * cot_theta


def surfaces(det: DetectorConfig):
    """Flat surface table: sensitive barrels, sensitive disks, then passive material.
    A track can only reach the disks on the side it moves towards, so each disk pair
    is one entry (its z sign follows the track). Row:
    (kind, r, |z|_max, z_disk, (r_min, r_max), σ1, σ2, x/X0, name, |z|_min, sensitive)."""
    rows = [("b", b.radius, b.half_length, 0.0, 0.0, b.sigma_rphi, b.sigma_z, b.x_over_X0, b.name, 0.0, True)
            for b in det.barrels]
    rows += [("d", 0.0, 0.0, d.z, (d.r_min, d.r_max), d.sigma_rphi, d.sigma_r, d.x_over_X0, d.name, 0.0, True)
             for d in det.disks]
    for q in det.passive:
        if isinstance(q, PassiveCylinder):
            rows.append(("b", q.radius, q.z_max, 0.0, 0.0, INF, INF, q.x_over_X0, q.name, q.z_min, False))
        else:
            rows.append(("d", 0.0, 0.0, q.z, (q.r_min, q.r_max), INF, INF, q.x_over_X0, q.name, 0.0, False))
    return rows


def simulate_hits(p: np.ndarray, q: np.ndarray, det: DetectorConfig, rng: np.random.Generator,
                  z_vertex: np.ndarray | None = None) -> Hits:
    p = np.atleast_2d(p)[:, -3:]
    n = len(p)
    pt = np.hypot(p[:, 0], p[:, 1])
    ptot = np.linalg.norm(p, axis=1)
    R = pt / (C_TRACK * det.B)
    k = np.sign(q).astype(float)
    phi0 = np.arctan2(p[:, 1], p[:, 0])
    cot = p[:, 2] / pt
    sin_t = 1 / np.sqrt(1 + cot**2)
    cos_t = cot * sin_t
    z0 = np.zeros(n) if z_vertex is None else z_vertex

    rows = surfaces(det)
    S = len(rows)
    s = np.full((n, S), np.inf)
    crossed = np.zeros((n, S), bool)
    sig1 = np.empty((n, S)); sig2 = np.empty((n, S)); x0 = np.empty((n, S))
    disk = np.zeros((n, S), bool)
    sens = np.zeros((n, S), bool)
    for j, (kind, r, hl, zd, rr, s1, s2, xx, _, zmin, sensitive) in enumerate(rows):
        sig1[:, j], sig2[:, j], x0[:, j], sens[:, j] = s1, s2, xx, sensitive
        if kind == "b":
            ok = r < 2 * R
            sj = 2 * R * np.arcsin(np.clip(r / (2 * R), 0, 1))
            zj = z0 + sj * cot
            ok &= (np.abs(zj) <= hl) & (np.abs(zj) >= zmin)
        else:
            disk[:, j] = True
            with np.errstate(divide="ignore", invalid="ignore"):
                sj = (np.where(cot >= 0, zd, -zd) - z0) / cot
            ok = np.isfinite(sj) & (sj > 0) & (sj < np.pi * R)
            rj = 2 * R * np.abs(np.sin(np.where(ok, sj, 0) / (2 * R)))
            ok &= (rj >= rr[0]) & (rj <= rr[1])
        s[:, j] = np.where(ok, sj, np.inf)
        crossed[:, j] = ok

    order = np.argsort(s, axis=1)
    take = lambda a: np.take_along_axis(a, order, axis=1)
    s, crossed, sig1, sig2, x0, disk, sens = map(take, (s, crossed, sig1, sig2, x0, disk, sens))
    surf = order
    # crossed surfaces are now first; drop columns that no track uses
    smax = max(int(crossed.sum(1).max()), 1)
    s, crossed, sig1, sig2, x0, disk, surf, sens = (a[:, :smax] for a in (s, crossed, sig1, sig2, x0, disk, surf, sens))
    S = smax
    sf = np.where(crossed, s, 0.0)
    x, y, z = helix_at(sf, R[:, None], k[:, None], phi0[:, None], cot[:, None], z0[:, None])
    phit = phi0[:, None] - k[:, None] * sf / R[:, None]   # transverse track direction at each hit

    if det.multiple_scattering:
        # path length through the layer grows as 1/cos(incidence)
        rhat_dot_t = np.cos(phit - np.arctan2(y, x)) * sin_t[:, None]
        inc = np.where(disk, np.abs(cos_t)[:, None], np.abs(rhat_dot_t))
        xeff = np.where(crossed, x0 / np.maximum(inc, 1e-3), 0.0)
        beta = ptot / np.sqrt(ptot**2 + M_MU**2)
        # logarithmic factor of the combined scatterer (PDG): total thickness along the track
        xtot = xeff.sum(1)
        logf = np.where(xtot > 0, 1 + 0.038 * np.log(np.where(xtot > 0, xtot, 1.0) / beta**2), 0.0)
        theta0 = HIGHLAND_E0 / (beta * ptot)[:, None] * np.sqrt(xeff) * np.maximum(logf, 0.0)[:, None]
        l = sf / sin_t[:, None]
        th1, th2 = rng.standard_normal(size=(2, n, S), dtype=np.float32) * theta0
        def displacement(th):
            c1 = np.cumsum(th, 1) - th            # sum_{i<j} θ_i
            c2 = np.cumsum(th * l, 1) - th * l    # sum_{i<j} θ_i l_i
            return l * c1 - c2
        u, v = displacement(th1), displacement(th2)
        # u: transverse, perpendicular to the track
        x = x - u * np.sin(phit)
        y = y + u * np.cos(phit)
        # v: out of the bending plane, projected back onto the surface
        z = np.where(disk, z, z + v / sin_t[:, None])
        dr = np.where(disk, -v / np.where(np.abs(cos_t) > 1e-6, cos_t, 1e-6)[:, None], 0.0)
        x = x + dr * np.cos(phit)
        y = y + dr * np.sin(phit)
        # barrel hits stay on their cylinder
        r_nom = np.where(disk, 1.0, 2 * R[:, None] * np.sin(np.clip(sf / (2 * R[:, None]), 0, np.pi / 2)))
        rr = np.hypot(x, y)
        scale = np.where(disk | (rr == 0), 1.0, r_nom / np.where(rr == 0, 1, rr))
        x, y = x * scale, y * scale

    if det.smear:
        r = np.hypot(x, y)
        a = np.arctan2(y, x) + rng.standard_normal(size=(n, S), dtype=np.float32) * np.where(np.isfinite(sig1), sig1, 0.0) / np.maximum(r, 1e-9)
        long = rng.standard_normal(size=(n, S), dtype=np.float32) * np.where(np.isfinite(sig2), sig2, 0.0)
        r_new = np.where(disk, r + long, r)
        x, y = r_new * np.cos(a), r_new * np.sin(a)
        z = np.where(disk, z, z + long)

    # passive surfaces were needed for the scattering only: they are never "crossed
    # sensitive surfaces" and never record hits
    crossed = crossed & sens
    detected = crossed & (rng.random(size=(n, S), dtype=np.float32) < det.hit_efficiency) if det.hit_efficiency < 1 else crossed.copy()
    return Hits(x, y, z, s, crossed, detected, sig1, sig2, disk, surf)

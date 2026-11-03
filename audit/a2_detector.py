"""Audit A2: helix vs Lorentz-force integration, curvature constant, Highland scattering,
hit bookkeeping, truth independence of the fit, curvature offset (D1), acceptance, material."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlelab.detector import cms_inspired, simulate_hits, fit_tracks       # noqa: E402
from particlelab.detector.geometry import BarrelLayer, DetectorConfig        # noqa: E402
from particlelab.detector.simulation import helix_at                         # noqa: E402

OUT = {}
MMU = 0.1056583755

# ---- 1. curvature constant from SI constants ---------------------------------------
c = 299792458.0                 # m/s (exact)
e = 1.602176634e-19             # C (exact)
# p [kg m/s] = e B R ; p [GeV/c] = p[kg m/s] * c / (e * 1e9)
OUT["C_TRACK_from_SI"] = e * 1.0 * 1.0 * c / (e * 1e9)


# ---- 2. RK4 integration of dp/ds = kappa q (p_hat x B) -------------------------------
def rk4_track(p0, q, B, z0=0.0, ds=2e-4, r_stop=1.2):
    """Integrate position and momentum along path length s (m). B along +z (tesla)."""
    kappa = c * 1e-9            # GeV/(T m) per unit charge
    Bv = np.array([0, 0, B])
    def f(state):
        x, p = state[:3], state[3:]
        u = p / np.linalg.norm(p)
        return np.r_[u, kappa * q * np.cross(u, Bv)]
    st = np.r_[0.0, 0.0, z0, p0]
    pts = [st[:3].copy()]
    while np.hypot(st[0], st[1]) < r_stop and abs(st[2]) < 3.0 and len(pts) < 200000:
        k1 = f(st); k2 = f(st + ds / 2 * k1); k3 = f(st + ds / 2 * k2); k4 = f(st + ds * k3)
        st = st + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts.append(st[:3].copy())
    return np.array(pts), st[3:]


def crossing(pts, r):
    rr = np.hypot(pts[:, 0], pts[:, 1])
    i = np.flatnonzero(rr >= r)[0]
    t = (r - rr[i - 1]) / (rr[i] - rr[i - 1])
    return pts[i - 1] + t * (pts[i] - pts[i - 1])


det = cms_inspired().ideal().with_(vertex_sigma_z=0.0)
helix = []
for pt, eta, q in ((5.0, 0.0, 1), (5.0, 0.0, -1), (1.5, 0.8, 1), (20.0, -1.2, -1), (45.0, 0.3, 1)):
    phi = 0.7
    p0 = np.array([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)])
    pts, pend = rk4_track(p0, q, 3.8, z0=0.01)
    h = simulate_hits(p0[None], np.array([float(q)]), det, np.random.default_rng(0), np.array([0.01]))
    worst = 0.0
    for j in np.flatnonzero(h.crossed[0] & ~h.is_disk[0]):
        r = np.hypot(h.x[0, j], h.y[0, j])
        ref = crossing(pts, r)
        worst = max(worst, float(np.linalg.norm(ref - np.array([h.x[0, j], h.y[0, j], h.z[0, j]]))))
    # handedness: sign of the z-component of (direction at start) x (direction later)
    d0 = pts[1] - pts[0]; d1 = pts[-1] - pts[-2]
    turn = np.sign(d0[0] * d1[1] - d0[1] * d1[0])
    helix.append(dict(pt=pt, eta=eta, q=q, max_hit_vs_rk4_um=worst * 1e6, turn_sign=int(turn),
                      clockwise_for_positive=bool((q > 0) == (turn < 0)),
                      momentum_magnitude_drift=float(abs(np.linalg.norm(pend) / np.linalg.norm(p0) - 1))))
OUT["helix_vs_rk4"] = helix

# ---- 3. Highland: independent evaluation and empirical check -------------------------
def highland(p, x, beta=1.0):
    return 0.0136 / (beta * p) * np.sqrt(x) * (1 + 0.038 * np.log(x))


p = 5.0
beta = p / np.hypot(p, MMU)
X = 0.05
two = DetectorConfig("audit two-layer", 3.8,
                     (BarrelLayer("A", 0.10, 5.0, 1e-6, 1e-6, X), BarrelLayer("B", 1.00, 5.0, 1e-6, 1e-6, 0.0)),
                     multiple_scattering=True, smear=False, hit_efficiency=1.0, min_hits=2)
n = 200000
phi = np.random.default_rng(1).uniform(-np.pi, np.pi, n)
pp = np.column_stack([p * np.cos(phi), p * np.sin(phi), np.zeros(n)])
q = np.ones(n)
hms = simulate_hits(pp, q, two, np.random.default_rng(2))
hid = simulate_hits(pp, q, two.with_(multiple_scattering=False), np.random.default_rng(2))
j = 1
dphi = np.angle(np.exp(1j * (np.arctan2(hms.y[:, j], hms.x[:, j]) - np.arctan2(hid.y[:, j], hid.x[:, j]))))
drphi = dphi * 1.0
dz = hms.z[:, j] - hid.z[:, j]
R = p / (0.299792458 * 3.8)
s1 = 2 * R * np.arcsin(0.10 / (2 * R)); s2 = 2 * R * np.arcsin(1.0 / (2 * R))
# incidence at layer A: angle between track and radial direction = s1/(2R)
inc = np.cos(s1 / (2 * R))
th0 = highland(p, X / inc, beta)
lever = s2 - s1
# rphi displacement on the cylinder ~ theta0*lever / cos(incidence at B)
incB = np.cos(s2 / (2 * R))
OUT["highland_two_layer"] = dict(theta0_pred_mrad=1e3 * th0,
                                 rphi_rms_um=1e6 * float(np.std(drphi)), rphi_pred_um=1e6 * th0 * lever / incB,
                                 z_rms_um=1e6 * float(np.std(dz)), z_pred_um=1e6 * th0 * lever,
                                 ratio_rphi=float(np.std(drphi) / (th0 * lever / incB)),
                                 ratio_z=float(np.std(dz) / (th0 * lever)))

# per-layer vs total log term (PDG: apply Highland to the total thickness)
full = cms_inspired()
xs = np.array([b.x_over_X0 for b in full.barrels])
per_layer = np.sqrt(np.sum(highland(10.0, xs)**2))
total = highland(10.0, xs.sum())
OUT["highland_per_layer_vs_total_eta0_10GeV"] = dict(per_layer_mrad=1e3 * per_layer, total_mrad=1e3 * total,
                                                     ratio=float(per_layer / total), x_total=float(xs.sum()))

# ---- 4. hit bookkeeping and fit independence from truth ------------------------------
rng = np.random.default_rng(7)
N = 20000
pt = rng.uniform(3, 50, N); eta = rng.uniform(-2.4, 2.4, N); ph = rng.uniform(-np.pi, np.pi, N)
P = np.column_stack([pt * np.cos(ph), pt * np.sin(ph), pt * np.sinh(eta)])
Q = rng.choice([-1.0, 1.0], N)
zv = rng.normal(0, 0.06, N)
h = simulate_hits(P, Q, full, np.random.default_rng(8), zv)
f0 = fit_tracks(h, full)
OUT["detected_subset_of_crossed"] = bool(np.all(~h.detected | h.crossed))
OUT["mean_missed_fraction"] = float(1 - h.detected.sum() / h.crossed.sum())
# move every non-detected hit far away and scramble the true arc lengths
h2 = replace(h, x=np.where(h.detected, h.x, 1e3), y=np.where(h.detected, h.y, -1e3),
             z=np.where(h.detected, h.z, 7e2), s=np.random.default_rng(9).permutation(h.s.ravel()).reshape(h.s.shape),
             crossed=np.ones_like(h.crossed))
f2 = fit_tracks(h2, full)
same = np.allclose(np.nan_to_num(f0.p), np.nan_to_num(f2.p), rtol=0, atol=0) and np.array_equal(f0.q, f2.q)
OUT["fit_unchanged_when_undetected_hits_moved_and_truth_arclength_scrambled"] = bool(same)
OUT["charge_misid_rate"] = float(np.mean(f0.q[f0.ok] != Q[f0.ok]))

# z-fit fallback: fewer than 2 detected hits with finite longitudinal error
fin = h.detected & np.isfinite(h.sigma_long)
fallback = f0.ok & (fin.sum(1) < 2)
OUT["zfit_fallback_tracks"] = int(fallback.sum())
OUT["zfit_fallback_fraction_of_ok"] = float(fallback.sum() / max(f0.ok.sum(), 1))

# reproducibility of the hit simulation
ha = simulate_hits(P[:500], Q[:500], full, np.random.default_rng(3), zv[:500])
hb = simulate_hits(P[:500], Q[:500], full, np.random.default_rng(3), zv[:500])
OUT["hits_bitwise_reproducible"] = bool(all(np.array_equal(getattr(ha, k), getattr(hb, k)) for k in ("x", "y", "z", "detected")))


# ---- 5. curvature offset study (D1) ------------------------------------------------
def curvature_offset(det, pt, eta, n, seed):
    r = np.random.default_rng(seed)
    ph = r.uniform(-np.pi, np.pi, n)
    p = np.column_stack([pt * np.cos(ph), pt * np.sin(ph), np.full(n, pt * np.sinh(eta))])
    q = r.choice([-1.0, 1.0], n)
    z = r.normal(0, det.vertex_sigma_z, n) if det.vertex_sigma_z > 0 else None
    f = fit_tracks(simulate_hits(p, q, det, r, z), det)
    k = pt / np.hypot(f.p[f.ok, 0], f.p[f.ok, 1]) - 1          # (1/pT_reco)/(1/pT_true) - 1
    kq = (pt / np.hypot(f.p[f.ok, 0], f.p[f.ok, 1])) * f.q[f.ok] * q[f.ok] - 1
    return dict(mean=float(k.mean()), se=float(k.std() / np.sqrt(k.size)), pull=float(k.mean() / (k.std() / np.sqrt(k.size))),
                n=int(k.size), rms=float(k.std()))


cur = {}
variants = {"full": full, "no_ms": full.with_(multiple_scattering=False), "no_smear": full.with_(smear=False),
            "ideal": full.ideal(), "full_no_beamspot": full.with_(beamspot_constraint=0.0),
            "full_z_vertex_0": full.with_(vertex_sigma_z=0.0)}
for eta in (0.0, 2.3):
    for k_, d in variants.items():
        cur[f"eta{eta}_{k_}"] = curvature_offset(d, 5.0, eta, 200000, 100 + int(eta * 10))
OUT["curvature_offset_5GeV"] = cur

# ---- 6. acceptance and material vs eta ---------------------------------------------
acc = {}
for eta in (0.0, 1.0, 1.5, 2.0, 2.2, 2.4, 2.5, 2.7):
    r = np.random.default_rng(5)
    n = 4000
    ph = r.uniform(-np.pi, np.pi, n)
    p = np.column_stack([20 * np.cos(ph), 20 * np.sin(ph), np.full(n, 20 * np.sinh(eta))])
    hh = simulate_hits(p, np.ones(n), full, r, r.normal(0, 0.06, n))
    th = 2 * np.arctan(np.exp(-eta))
    xx = []
    for i in range(50):
        m = hh.crossed[i]
        x0 = np.array([full.barrels[s].x_over_X0 if s < len(full.barrels) else full.disks[s - len(full.barrels)].x_over_X0
                       for s in hh.surface[i][m]])
        inc = np.where(hh.is_disk[i][m], abs(np.cos(th)), np.sin(th))
        xx.append(np.sum(x0 / inc))
    acc[str(eta)] = dict(frac_ge8_hits=float(np.mean(hh.n_detected >= 8)), mean_hits=float(hh.n_detected.mean()),
                         material_X0=float(np.mean(xx)))
OUT["acceptance_material_vs_eta_20GeV"] = acc

Path(__file__).with_name("results").joinpath("a2_detector.json").write_text(json.dumps(OUT, indent=2))
print(json.dumps(OUT, indent=2))

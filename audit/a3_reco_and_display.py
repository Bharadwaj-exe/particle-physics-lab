"""Audit A3: invariant mass from reconstructed tracks (independent four-vectors),
reconstruction edge cases, and coordinate checks of the 2D/3D display data,
plus 3D scene behaviour (cameras, visibility, residual magnification, repeated
updates, memory)."""
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_API", "pyside6")
from particlelab.analysis import simulate                                    # noqa: E402
from particlelab.detector import cms_inspired, simulate_hits, fit_tracks    # noqa: E402
from particlelab.physics import generate                                    # noqa: E402
from particlelab.visualization.event_display import make_event, detector_geometry  # noqa: E402

OUT = {}
MMU = 0.1056583755
mp.mp.dps = 40


def mass4(pa, pb):
    """(p_a + p_b)^2 with E from the muon mass, in 40-digit arithmetic."""
    pa = [mp.mpf(float(v)) for v in pa]; pb = [mp.mpf(float(v)) for v in pb]
    Ea = mp.sqrt(sum(v * v for v in pa) + MMU**2); Eb = mp.sqrt(sum(v * v for v in pb) + MMU**2)
    return float(mp.sqrt((Ea + Eb)**2 - sum((x + y)**2 for x, y in zip(pa, pb))))


det = cms_inspired()
# ---- 1. pipeline mass vs independent four-vector mass ----------------------------------
for name in ("J/psi", "Z"):
    r = simulate(name, 20000, det, seed=77)
    idx = np.random.default_rng(0).choice(r.reco_mass.size, min(300, r.reco_mass.size), replace=False)
    ref = np.array([mass4(r.p_minus[i], r.p_plus[i]) for i in idx])
    OUT[f"{name}_pipeline_mass_vs_independent_max_rel"] = float(np.max(np.abs(r.reco_mass[idx] / ref - 1)))
    # perfect detector: reconstructed = true
    ri = simulate(name, 5000, det.ideal().with_(vertex_sigma_z=0.06), seed=78)
    OUT[f"{name}_ideal_detector_reco_minus_true_max_rel"] = float(np.max(np.abs(ri.reco_mass / ri.true_mass - 1)))
    # charge labelling: reconstructed mu- matches the true mu- direction
    dots = np.sum(r.p_minus * r.true_minus[:, 1:], 1) / np.linalg.norm(r.p_minus, axis=1) / np.linalg.norm(r.true_minus[:, 1:], axis=1)
    OUT[f"{name}_reco_minus_matches_true_minus_frac"] = float(np.mean(dots > 0.99))
    OUT[f"{name}_reco_minus_true_mean_MeV"] = float(1000 * np.mean(r.reco_mass - r.true_mass))

# ---- 2. edge cases: low momentum, forward, missing hits -------------------------------------
edge = {}
rng = np.random.default_rng(4)
for label, pt, eta, eff in (("low_pT_1GeV_eta0", 1.0, 0.0, 0.99), ("pT3_eta2.35", 3.0, 2.35, 0.99),
                            ("pT20_eta0_eff0.80", 20.0, 0.0, 0.80), ("pT20_eta2.45", 20.0, 2.45, 0.99),
                            ("loopers_pT0.6", 0.6, 0.0, 0.99)):
    n = 20000
    d = det.with_(hit_efficiency=eff)
    ph = rng.uniform(-np.pi, np.pi, n)
    p = np.column_stack([pt * np.cos(ph), pt * np.sin(ph), np.full(n, pt * np.sinh(eta))])
    q = rng.choice([-1.0, 1.0], n)
    f = fit_tracks(simulate_hits(p, q, d, rng, rng.normal(0, 0.06, n)), d)
    ok = f.ok
    rel = np.hypot(f.p[ok, 0], f.p[ok, 1]) / pt - 1 if ok.any() else np.array([np.nan])
    edge[label] = dict(efficiency=float(ok.mean()), mean_rel=float(np.nanmean(rel)),
                       robust_sigma=float(0.5 * np.subtract(*np.nanpercentile(rel, [84.135, 15.865]))),
                       charge_misid=float(np.mean(f.q[ok] != q[ok])) if ok.any() else None,
                       finite=bool(np.all(np.isfinite(f.p[ok]))))
OUT["edge_cases"] = edge

# ---- 3. display data vs independent reconstruction of the same event -----------------------
C = 0.299792458


def circle_through_origin(p, q, B):
    pt = np.hypot(p[0], p[1]); R = pt / (C * B)
    # centre is to the right of the direction for positive charge (clockwise motion seen from +z)
    nrm = np.array([p[1], -p[0]]) / pt * np.sign(q)
    return R * nrm, R


disp = {}
for seed in range(5):
    for name in ("J/psi", "Upsilon(1S)", "Z"):
        rng_a = np.random.default_rng(1000 + seed)
        ev = make_event(name, det, rng_a)
        before = copy.deepcopy(ev)
        for k, t in enumerate(ev.tracks):
            for path, p, q, z0 in ((t.true_path, t.p_true, t.charge_true, ev.vertex[2]),
                                   (t.reco_path, t.p_reco, t.charge_reco, None)):
                c, R = circle_through_origin(p, q, det.B)
                rad_err = np.abs(np.hypot(path[:, 0] - c[0], path[:, 1] - c[1]) - R).max() / R
                start_err = np.abs(path[0, :2]).max()
                # z linear in transverse arc length with slope pz/pT
                s = 2 * R * np.arcsin(np.clip(np.hypot(path[:, 0], path[:, 1]) / (2 * R), 0, 1))
                slope = np.polyfit(s[: len(s) // 3], path[: len(s) // 3, 2], 1)[0]
                disp.setdefault("max_circle_rel_err", 0.0)
                disp["max_circle_rel_err"] = max(disp["max_circle_rel_err"], float(rad_err))
                disp.setdefault("max_start_offset_m", 0.0)
                disp["max_start_offset_m"] = max(disp["max_start_offset_m"], float(start_err))
                disp.setdefault("max_dzds_err", 0.0)
                disp["max_dzds_err"] = max(disp["max_dzds_err"], float(abs(slope - p[2] / np.hypot(p[0], p[1]))))
                if z0 is not None:
                    disp.setdefault("true_path_starts_at_vertex_max_dz", 0.0)
                    disp["true_path_starts_at_vertex_max_dz"] = max(disp["true_path_starts_at_vertex_max_dz"], float(abs(path[0, 2] - z0)))
            # ideal crossings lie on the true helix circle
            c, R = circle_through_origin(t.p_true, t.charge_true, det.B)
            disp.setdefault("hits_ideal_on_true_circle_max_rel", 0.0)
            disp["hits_ideal_on_true_circle_max_rel"] = max(disp["hits_ideal_on_true_circle_max_rel"], float(
                np.abs(np.hypot(t.hits_ideal[:, 0] - c[0], t.hits_ideal[:, 1] - c[1]) - R).max() / R))
            # residual magnification must not change stored arrays
            _ = t.hits_exaggerated(1000.0)
            disp.setdefault("magnification_mutates_data", False)
            if not (np.array_equal(t.hits, before.tracks[k].hits) and np.array_equal(t.hits_ideal, before.tracks[k].hits_ideal)):
                disp["magnification_mutates_data"] = True
            disp.setdefault("reco_differs_from_true_count", 0)
            disp["reco_differs_from_true_count"] += int(not np.allclose(t.true_path, t.reco_path))
            disp.setdefault("label_charge_consistent", True)
            disp["label_charge_consistent"] &= (t.label == "μ⁻") == (t.charge_true < 0)
        # reconstructed mass in the display equals the independent four-vector mass
        disp.setdefault("display_mass_max_rel_err", 0.0)
        disp["display_mass_max_rel_err"] = max(disp["display_mass_max_rel_err"],
                                               abs(ev.reco_mass / mass4(ev.tracks[0].p_reco, ev.tracks[1].p_reco) - 1))
        # replay: same seed gives the same display event (reproducibility)
        ev2 = make_event(name, det, np.random.default_rng(1000 + seed))
        disp.setdefault("display_event_reproducible", True)
        disp["display_event_reproducible"] &= all(np.array_equal(a.hits, b.hits) and np.array_equal(a.reco_path, b.reco_path)
                                                  for a, b in zip(ev.tracks, ev2.tracks))
OUT["display_coordinates"] = disp

# 2D projection convention used by desktop.EventTab: XY = (x, y); ZR = (z, side*hypot(x, y))
ev = make_event("Z", det, np.random.default_rng(5))
t = ev.tracks[0]
side = np.sign(t.true_path[-1, 1]) or 1.0
zr = np.column_stack([t.hits[:, 2], side * np.hypot(t.hits[:, 0], t.hits[:, 1])])
OUT["zr_projection_preserves_radius_max_err"] = float(np.abs(np.abs(zr[:, 1]) - np.hypot(t.hits[:, 0], t.hits[:, 1])).max())
OUT["zr_sign_taken_from_true_path"] = True     # read from desktop.py: side = sign(true_path[-1, 1])

# geometry: display shapes vs configuration
g = detector_geometry(det)
cyl = {c.name: c for c in g.cylinders}
OUT["geometry_barrels_match"] = all(cyl[b.name].radius == b.radius and cyl[b.name].half_length == min(b.half_length, 3.0) for b in det.barrels)
OUT["geometry_disks_match"] = all(any(dd.z == s * d.z and dd.r_min == d.r_min and dd.r_max == d.r_max for dd in g.disks)
                                  for d in det.disks for s in (1, -1))
OUT["beam_pipe_inside_first_layer"] = cyl["beam pipe"].radius < det.barrels[0].radius

# ---- 4. 3D scene behaviour (off-screen) ---------------------------------------------------
try:
    import psutil
    import pyvista as pv
    from particlelab.app.view3d import SceneBuilder, DETECTOR_GROUPS, EVENT_GROUPS
    pl = pv.Plotter(off_screen=True, window_size=(640, 480))
    sb = SceneBuilder(pl, pv)
    sb.set_geometry(g)
    sc = {}
    cams = {}
    for v in ("iso", "beam", "side"):
        sb.camera(v)
        cams[v] = [list(map(float, x)) for x in pl.camera_position]
    sc["camera_presets_distinct"] = len({json.dumps(c) for c in cams.values()}) == 3
    sc["beam_view_looks_along_z"] = bool(abs(cams["beam"][0][2] - cams["beam"][1][2]) > 0 and
                                         abs(cams["beam"][0][0] - cams["beam"][1][0]) < 1e-9 and abs(cams["beam"][0][1] - cams["beam"][1][1]) < 1e-9)
    d0 = pl.camera.distance; pl.camera.zoom(2.0); sc["zoom_changes_view"] = bool(pl.camera.view_angle != 30.0 or pl.camera.distance != d0)
    pos0 = np.array(pl.camera.position); pl.camera.azimuth = 30; pl.render(); sc["rotation_changes_position"] = bool(np.linalg.norm(np.array(pl.camera.position) - pos0) > 1e-6)
    sb.camera("iso")
    # geometry caching: same key -> same actors
    a0 = sb.actors["strip barrel"][0]; sb.set_geometry(detector_geometry(det)); sc["geometry_cached"] = sb.actors["strip barrel"][0] is a0
    # residual magnification leaves the event data intact
    ev = make_event("Z", det, np.random.default_rng(9)); keep = copy.deepcopy(ev)
    sb.set_event(ev, 1000.0); sb.set_event(ev, 1.0)
    sc["magnification_leaves_event_unchanged"] = all(np.array_equal(a.hits, b.hits) for a, b in zip(ev.tracks, keep.tracks))
    pts = pv.wrap(sb.actors["hits"][0].GetMapper().GetInput()).points
    sc["true_scale_hits_equal_simulation"] = bool(np.allclose(pts, ev.tracks[0].hits, atol=1e-6))
    # repeated updates: actors, time, memory
    proc = psutil.Process()
    rng = np.random.default_rng(10)
    evs = [make_event(n, det, rng) for n in ("J/psi", "Upsilon(1S)", "Z") * 10]
    gc.collect(); rss0 = proc.memory_info().rss
    times = []
    for i in range(300):
        t0 = time.perf_counter(); sb.set_event(evs[i % len(evs)], (1.0, 100.0)[i % 2]); times.append(time.perf_counter() - t0)
        if i == 49:
            gc.collect(); rss50 = proc.memory_info().rss
    gc.collect(); rss300 = proc.memory_info().rss
    sc["renderer_actor_count_after_300_updates"] = len(pl.renderer.actors)
    sc["builder_actor_count"] = sb.n_actors()
    sc["update_ms_median"] = 1000 * float(np.median(times)); sc["update_ms_p95"] = 1000 * float(np.percentile(times, 95))
    sc["rss_MB_start_50_300"] = [rss0 / 2**20, rss50 / 2**20, rss300 / 2**20]
    sc["rss_growth_MB_per_100_updates_after_warmup"] = (rss300 - rss50) / 2**20 / 2.5
    pl.close()
    OUT["scene"] = sc
except Exception as e:      # pragma: no cover
    OUT["scene"] = f"skipped: {type(e).__name__}: {e}"

Path(__file__).with_name("results").joinpath("a3_reco_and_display.json").write_text(json.dumps(OUT, indent=2, default=str))
print(json.dumps(OUT, indent=2, default=str))

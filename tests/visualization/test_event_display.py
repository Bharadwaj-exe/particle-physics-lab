"""Display-data preparation: geometry generation and coordinate mapping (no renderer needed)."""
import numpy as np
import pytest

from particlelab.detector import cms_inspired, uniform_barrel, simulate_hits, fit_tracks
from particlelab.units import C_TRACK
from particlelab.visualization.event_display import (detector_geometry, make_event, helix_path, momentum_arrow,
                                                     BEAM_PIPE_RADIUS)


def test_geometry_matches_detector_configuration():
    det = cms_inspired()
    g = detector_geometry(det)
    barrels = [c for c in g.cylinders if c.group != "beam pipe"]
    assert [c.radius for c in barrels] == [b.radius for b in det.barrels]
    assert [c.half_length for c in barrels] == [b.half_length for b in det.barrels]
    assert {c.group for c in barrels} == {"pixel barrel", "strip barrel"}
    # every disk appears at +z and −z with the configured radii
    assert len(g.disks) == 2 * len(det.disks)
    for d in det.disks:
        pair = [s for s in g.disks if abs(abs(s.z) - d.z) < 1e-12 and s.r_min == d.r_min]
        assert sorted(s.z for s in pair) == [-d.z, d.z]
    assert {s.group for s in g.disks} == {"pixel disks", "inner disks", "endcap disks"}


def test_beam_pipe_and_axis():
    g = detector_geometry(cms_inspired())
    pipe = [c for c in g.cylinders if c.group == "beam pipe"]
    assert len(pipe) == 1 and pipe[0].radius == BEAM_PIPE_RADIUS
    assert BEAM_PIPE_RADIUS < min(c.radius for c in g.cylinders if c.group != "beam pipe")
    (a, b) = g.beam_axis
    assert a[:2] == b[:2] == (0.0, 0.0) and a[2] < -2.7 and b[2] > 2.7      # along z, beyond the last disk


def test_infinite_reference_layers_are_capped_and_key_changes_with_geometry():
    g = detector_geometry(uniform_barrel(), max_half_length=3.0)
    assert all(np.isfinite(c.half_length) and c.half_length <= 3.0 for c in g.cylinders)
    assert g.key != detector_geometry(cms_inspired()).key
    assert g.key == detector_geometry(uniform_barrel()).key


def test_helix_path_geometry():
    p, B, z0 = np.array([3.0, 4.0, 2.0]), 3.8, 0.05
    path = helix_path(p, +1, B, z0, s_max=1.0)
    assert np.allclose(path[0], [0, 0, z0])
    R = 5.0 / (C_TRACK * B)
    xc, yc = R * np.sin(np.arctan2(4, 3)), -R * np.cos(np.arctan2(4, 3))
    assert np.allclose(np.hypot(path[:, 0] - xc, path[:, 1] - yc), R)        # on its circle
    assert np.allclose(np.diff(path[:, 2]) / np.diff(np.linspace(0, 1, len(path))), 2.0 / 5.0)   # dz/ds = pz/pT


def test_momentum_arrow():
    o, d, L = momentum_arrow([3.0, 4.0, 0.0], [0, 0, 0.1])
    assert np.allclose(d, [0.6, 0.8, 0.0]) and np.allclose(o, [0, 0, 0.1])
    assert momentum_arrow([30.0, 40.0, 0.0], [0, 0, 0])[2] > L                # longer for larger |p|


def test_event_is_reproducible_and_passes_analysis_cuts():
    det = cms_inspired()
    a = make_event("Upsilon(1S)", det, np.random.default_rng(5))
    b = make_event("Upsilon(1S)", det, np.random.default_rng(5))
    assert np.array_equal(a.tracks[0].hits, b.tracks[0].hits)
    assert a.passes_cuts
    for t in a.tracks:
        assert t.pt_reco > det.pt_min and abs(t.eta_reco) < det.eta_max and t.n_hits >= det.min_hits
    assert {t.charge_reco for t in a.tracks} == {-1, 1}


def test_coordinate_mapping_equals_simulation_arrays():
    """Hits in the display are exactly the simulated hit coordinates (same random stream)."""
    det = cms_inspired()
    ev = make_event("Z", det, np.random.default_rng(11))
    # replay the same random stream through the simulation directly
    from particlelab.physics import generate
    rng = np.random.default_rng(11)
    while True:
        gen = generate("Z", 1, seed=rng)
        z0 = rng.normal(0, det.vertex_sigma_z)
        hs, fs = [], []
        for p4, q in ((gen.mu_minus[0], -1.0), (gen.mu_plus[0], 1.0)):
            h = simulate_hits(p4[None, 1:], np.array([q]), det, rng, np.array([z0]))
            hs.append(h); fs.append(fit_tracks(h, det))
        if np.isclose(gen.mass[0], ev.true_mass):
            break
    for t, h, f in zip(ev.tracks, hs, fs):
        m = h.detected[0]
        assert np.array_equal(t.hits, np.column_stack([h.x[0][m], h.y[0][m], h.z[0][m]]))
        assert np.array_equal(t.p_reco, f.p[0])
    assert np.allclose(ev.vertex, [0, 0, z0])


def test_ideal_detector_true_and_reconstructed_paths_coincide():
    det = cms_inspired().ideal()
    ev = make_event("J/psi", det, np.random.default_rng(3))
    for t in ev.tracks:
        assert np.allclose(t.reco_path, t.true_path, atol=1e-9)
        assert np.allclose(t.hits, t.hits_ideal, atol=1e-12)                  # no smearing, no scattering
        r = np.hypot(t.hits[:, 0], t.hits[:, 1])
        barrel_r = np.array([b.radius for b in det.barrels])
        on_barrel = ~t.hits_disk
        assert np.all(np.min(np.abs(r[on_barrel, None] - barrel_r[None, :]), axis=1) < 1e-12)
        disk_z = np.array([d.z for d in det.disks])
        assert np.all(np.min(np.abs(np.abs(t.hits[t.hits_disk, 2])[:, None] - disk_z[None, :]), axis=1) < 1e-12)
        assert np.allclose(t.true_path[0], ev.vertex)


def test_smeared_hits_and_residual_exaggeration():
    ev = make_event("Upsilon(1S)", cms_inspired(), np.random.default_rng(8))
    t = ev.tracks[0]
    res = np.linalg.norm(t.hits - t.hits_ideal, axis=1)
    assert 0 < res.max() < 5e-3                                              # µm–mm scale displacements
    assert np.array_equal(t.hits_exaggerated(1.0), t.hits)
    assert np.allclose(t.hits_exaggerated(100.0) - t.hits_ideal, 100 * (t.hits - t.hits_ideal))


def test_missed_hits_come_from_inefficiency():
    det = cms_inspired().with_(hit_efficiency=0.6, min_hits=3)
    ev = make_event("Z", det, np.random.default_rng(2), require_cuts=False)
    assert sum(len(t.missed) for t in ev.tracks) > 0
    full = make_event("Z", cms_inspired().with_(hit_efficiency=1.0), np.random.default_rng(2))
    assert all(len(t.missed) == 0 for t in full.tracks)

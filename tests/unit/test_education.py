import numpy as np
import pytest

from particlelab.detector import cms_inspired
from particlelab.domain.particles import RESONANCES
from particlelab.education import from_display_event, from_momenta, gate
from particlelab.visualization.event_display import make_event


@pytest.mark.parametrize("name", RESONANCES)
def test_learner_chain_reproduces_the_simulated_reconstructed_mass(name):
    """Hits → R → pT → E → m by hand must give exactly the mass the simulation reports."""
    det = cms_inspired()
    for seed in range(5):
        ev = make_event(name, det, np.random.default_rng(seed))
        ex = from_display_event(ev)
        assert ex.mass() == pytest.approx(ev.reco_mass, rel=1e-9)
        for i, t in enumerate(sorted(ev.tracks, key=lambda t: t.label != "μ⁻")):
            assert ex.pt(i) == pytest.approx(t.pt_reco, rel=1e-9)
            assert ex.tracks[i].charge == t.charge_reco


def test_radius_from_chord_and_sagitta_is_exact():
    for R, L in [(0.5, 0.3), (3.3, 1.08), (40.0, 1.08), (0.6, 1.19)]:
        s = R - np.sqrt(R**2 - L**2 / 4)
        ex = from_momenta([0.299792458 * 3.8 * R, 0, 1.0], [-1.0, 2.0, -0.5], 3.8, (L, 0.8))
        assert ex.tracks[0].sagitta == pytest.approx(s, rel=1e-12)
        assert ex.radius(0) == pytest.approx(R, rel=1e-12)


@pytest.fixture
def ex():
    # a J/ψ-like pair: pT 4 and 3 GeV, opposite azimuths, moderate η
    pm = 4.0 * np.array([np.cos(0.3), np.sin(0.3), np.sinh(0.4)])
    pp = 3.0 * np.array([np.cos(2.9), np.sin(2.9), np.sinh(-0.2)])
    return from_momenta(pm, pp, 3.8, (1.08, 1.08))


def test_correct_answers_are_accepted_with_rounding(ex):
    e = ex.expected()
    for i in (0, 1):
        assert ex.check("R", round(e["R"][i], 3), i).ok
        assert ex.check("pT", round(e["pT"][i], 3), i).ok
        assert ex.check("E", round(e["E"][i], 3), i).ok
    assert ex.check("mass", round(e["mass"], 3)).ok
    assert ex.check("pT", 0.3 * 3.8 * e["R"][0], 0).ok            # 0.3 instead of 0.2998 is fine


@pytest.mark.parametrize("step, wrong, track, keyword", [
    ("R", "chord_over_8s", 0, "s/2"),
    ("R", "R_cm", 0, "centimetres"),
    ("pT", "no_constant", 0, "0.2998"),
    ("pT", "R_cm", 0, "metres"),
    ("E", "pt_only", 1, "cosh"),
    ("E", "pz_only", 0, "sinh"),
    ("mass", "sum_E", None, "energies only"),
    ("mass", "m_squared", None, "square root"),
    ("mass", "scalar_p", None, "vectors"),
    ("mass", "no_z", None, "z components"),
    ("mass", "degrees", None, "degrees"),
])
def test_common_mistakes_are_named(ex, step, wrong, track, keyword):
    e = ex.expected()
    t = ex.tracks[track] if track is not None else None
    p1, p2 = ex.p3(0), ex.p3(1)
    E1, E2 = e["E"]
    value = {
        "chord_over_8s": lambda: t.chord**2 / (8 * t.sagitta),
        "R_cm": lambda: 100 * e["R"][track] if step == "R" else 0.2998 * 3.8 * e["R"][track] * 100,
        "no_constant": lambda: 3.8 * e["R"][track],
        "pt_only": lambda: e["pT"][track],
        "pz_only": lambda: abs(e["pT"][track] * np.sinh(t.eta)),
        "sum_E": lambda: E1 + E2,
        "m_squared": lambda: e["mass"] ** 2,
        "scalar_p": lambda: np.sqrt((E1 + E2) ** 2 - (np.linalg.norm(p1) + np.linalg.norm(p2)) ** 2),
        "no_z": lambda: np.sqrt((np.hypot(np.hypot(*p1[:2]), ex.m_mu) + np.hypot(np.hypot(*p2[:2]), ex.m_mu)) ** 2
                                - np.sum((p1[:2] + p2[:2]) ** 2)),
        "degrees": lambda: ex._mass_with(lambda tr: np.degrees(tr.phi)),
    }[wrong]()
    fb = ex.check(step, value, track)
    assert not fb.ok and keyword in fb.message and fb.expected is None


def test_wrong_answers_never_reveal_the_value(ex):
    fb = ex.check("mass", 1.0)
    assert not fb.ok and fb.expected is None
    assert not ex.check("pT", "abc", 0).ok
    with pytest.raises(ValueError):
        ex.check("pT", 1.0)            # track index required


def test_worked_solution_ends_with_the_mass(ex):
    lines = ex.worked_solution()
    assert len(lines) == 7 and f"{ex.mass():.4f}" in lines[-1]


@pytest.mark.parametrize("decay, simulable, allowed, word", [
    ("Z -> mu+ mu-", True, True, "simulated"),
    ("J/psi -> mu+ mu-", True, True, "simulated"),
    ("Upsilon(1S) -> mu- mu+", True, True, "simulated"),
    ("Z -> mu+ e-", False, False, "muon number"),
    ("J/psi -> mu+ mu+", False, False, "charge"),
    ("J/psi -> tau+ tau-", False, False, "mass"),
    ("J/psi -> p anti-p", False, True, "implemented"),
    ("Z mu+ mu-", False, False, "Cannot read"),
])
def test_process_gate(decay, simulable, allowed, word):
    g = gate(decay)
    assert g.simulable is simulable and g.allowed is allowed and word in g.message
    if simulable:
        assert g.process in RESONANCES

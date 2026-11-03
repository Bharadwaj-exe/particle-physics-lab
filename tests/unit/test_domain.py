import numpy as np
import pytest

from particlelab.domain.particles import load, RESONANCES
from particlelab.domain.conservation import check, parse, check_four_momentum


def test_database_has_verified_resonances_with_sources():
    db = load()
    assert "PDG" in db.source and db.retrieved
    for name in RESONANCES:
        p = db.get(name)
        assert p.verified and p.mass > 0 and p.width > 0 and 0 < p.br_mumu < 1 and p.mass_err > 0


def test_pdg_2026_values():
    # values read from PDG Live (2026 edition) on 2026-11-02
    db = load()
    assert db.get("Z").mass == 91.1879 and db.get("Z").width == 2.4955
    assert db.get("J/psi").mass == 3.0969 and db.get("J/psi").width == pytest.approx(92.6e-6)
    assert db.get("Upsilon(1S)").mass == 9.4604 and db.get("Upsilon(1S)").width == pytest.approx(54.02e-6)


def test_antiparticles_and_aliases():
    db = load()
    assert db.get("mu+").charge == +1 and db.get("mu+").Lmu == -1
    assert db.get("anti-p").baryon == -1 and db.get("pbar").id == "anti-p"
    assert db.get("jpsi").id == "J/psi"
    with pytest.raises(KeyError):
        db.get("unicorn")


@pytest.mark.parametrize("decay", ["Z -> mu+ mu-", "J/psi -> mu+ mu-", "Upsilon(1S) -> mu+ mu-",
                                   "J/psi -> p anti-p", "n -> p e- anti-nu_e", "mu- -> e- anti-nu_e nu_mu"])
def test_allowed(decay):                                       # original tests 17-22
    v = check(decay)
    assert v.allowed, v.reasons
    assert "necessary" in v.category


@pytest.mark.parametrize("decay,word,category", [
    ("Z -> mu+ e-", "muon number", "lepton flavour"),
    ("J/psi -> mu+ mu+", "charge", "exact"),
    ("J/psi -> tau+ tau-", "mass", "kinematically"),
    ("p -> e+ pi0", "baryon", "accidental"),
    ("mu- -> e- gamma", "muon number", "lepton flavour"),
])
def test_forbidden(decay, word, category):                     # original tests 23-27
    v = check(decay)
    assert not v.allowed and any(word in r for r in v.reasons), v.reasons
    assert category in v.category


def test_single_particle_decay_is_kinematically_forbidden():
    v = check("Z -> gamma")
    assert "kinematically" in v.category


def test_every_law_is_reported():
    v = check("Z -> mu+ mu-")
    kinds = {c.kind for c in v.checks}
    assert kinds == {"exact", "accidental", "approximate", "kinematic"}
    assert len(v.checks) == 8


def test_parse_errors():                                       # original test 28
    with pytest.raises(KeyError):
        parse("Z -> mu+ unicorn")
    with pytest.raises(ValueError):
        parse("Z mu+ mu-")


def test_four_momentum_check():
    from particlelab.physics import generate
    ev = generate("J/psi", 3, seed=1)
    assert check_four_momentum(ev.parent[0], [ev.mu_minus[0], ev.mu_plus[0]]).passed
    assert not check_four_momentum(ev.parent[0], [ev.mu_minus[0]]).passed

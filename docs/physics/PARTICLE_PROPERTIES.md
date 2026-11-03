# Particle Properties

**Source:** Particle Data Group, *Review of Particle Physics*, PDG Live, 2026 edition (Int. J. Mod. Phys. A 41, 2630011 (2026)). Retrieved 2 November 2026. **File:** `particlelab/domain/particles_pdg.json`.

| Particle | Mass (GeV) | Natural width Γ | BR(μ⁺μ⁻) | J | Checked against PDG Live |
|---|---|---|---|---|---|
| J/ψ(1S) | 3.096900 ± 0.000006 | 92.6 ± 1.7 keV | (5.961 ± 0.033) % | 1 | yes |
| Υ(1S) | 9.46040 ± 0.00010 | 54.02 ± 1.25 keV | (2.48 ± 0.04) % | 1 | yes |
| Z | 91.1879 ± 0.0020 | 2.4955 ± 0.0023 GeV | (3.3662 ± 0.0066) % | 1 | yes |
| μ | 0.1056583755 | — | — | ½ | no (standard value, re-check) |

The conservation checker also uses e, τ, neutrinos, p, n, π, K, γ, ψ(2S) and H. These are flagged `verified: false` until they are checked against PDG Live.

## Mass, width and resolution

- **Mass** is the line-shape parameter M. For the Z it is defined for a Breit–Wigner with s-dependent width, Γ(s) = Γ·s/M².
- **Natural width** Γ is a property of the particle (Γ = ħ/τ). For J/ψ and Υ it is less than 0.1 MeV, far below detector resolution (tens of MeV), so their observed peak widths are entirely detector effects. For the Z, the width (2.5 GeV) is larger than the resolution (about 1 GeV).
- **Detector resolution** is not a particle property. It is simulated in `detector/` and fitted as σ (Gaussian or Crystal Ball core).

## Line-shape models used in generation

| Particle | Model | Window |
|---|---|---|
| J/ψ, Υ | non-relativistic BW (Cauchy), sampled by inverse CDF | M ± 10Γ |
| Z | dσ/ds ∝ s/[(s − M²)² + s²Γ²/M²], × ds/dm = 2m, sampled by numerical inverse CDF | 60–120 GeV |

The Z model leaves out parton luminosity (which steepens the low-mass side) and γ*/Z interference. Both slightly distort the shape away from the pole. Generation and fit use the same function, so the fit recovers M_Z (tested at truth level: 91.1847 ± 0.0028 GeV).

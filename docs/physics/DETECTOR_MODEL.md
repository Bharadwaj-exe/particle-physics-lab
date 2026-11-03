# Detector Model

Code: `particlelab/detector/geometry.py` and `simulation.py`.

> **Scope.** `cms_inspired()` is an **educational model** with the main features of the CMS silicon tracker. Layer radii, lengths, resolutions and material are rounded values in the spirit of CMS, JINST 3 (2008) S08004. It is **not** the CMS geometry and makes no claim of CMS-level fidelity.

## Geometry (CMS-inspired)

| Subsystem | Positions | Coverage | σ(rφ) | σ(z or r) | x/X₀ per layer |
|---|---|---|---|---|---|
| Pixel barrel ×3 | r = 4.4, 7.3, 10.2 cm | \|z\| < 26.5 cm | 10 µm | 20 µm | 0.02 |
| Inner strip barrel ×4 | r = 25.5–49.8 cm | \|z\| < 70 cm | 30 µm | 0.5 mm (layers 1–2, stereo); none (mono) | 0.03 |
| Outer strip barrel ×6 | r = 55.5–108 cm | \|z\| < 118 cm | 45 µm | 0.5 mm (layers 1–2); none | 0.03 |
| Pixel disks ×2 per side | \|z\| = 34.5, 46.5 cm | 6 < r < 15 cm | 15 µm | 20 µm | 0.03 |
| Inner disks ×3 per side | \|z\| = 80–100 cm | 20 < r < 50 cm | 35 µm | 1 mm | 0.04 |
| Endcap disks ×9 per side | \|z\| = 1.24–2.72 m | 22 < r < 113 cm | 40 µm | 1 mm (odd); none (even) | 0.05 |

Other settings:
- **Field:** B = 3.8 T uniform along z.
- **Material at η = 0:** 0.36 X₀, rising with \|η\| through the 1/cos(incidence) path length and the disks.
- **Luminous region:** vertex z ~ N(0, 6 cm), with x = y = 0.
- **Hit efficiency:** 99 %.
- **Track requirement:** at least 8 recorded hits.

The original `uniform_barrel()` (10 infinitely long layers from 0.2 to 1.1 m, 20 µm, no material) is kept as the analytic reference.

## Trajectory and crossings

Helix from (0, 0, z₀):
- R = pT/(0.299792458 B), φ(s) = φ₀ − q s/R, centre q R (sin φ₀, −cos φ₀), z = z₀ + s cot θ.
- Positive charges curve clockwise seen from +z, consistent with F = q v × B.

Crossings:
- **Barrel:** s = 2R arcsin(r/2R) if r < 2R and \|z\| ≤ half-length.
- **Disk:** s = (±z_d − z₀)/cot θ on the side the track moves towards, with s < πR (no loopers), and r = 2R\|sin(s/2R)\| within the disk.

## Multiple Coulomb scattering

Highland formula (V. L. Highland, NIM 129 (1975) 497; PDG "Passage of particles through matter"):

θ₀ = (13.6 MeV / βcp) √(x/X₀) [1 + 0.038 ln(x/X₀)], with x/X₀ → (x/X₀)/cos α for incidence angle α.

At each crossed surface two independent kicks, θ₁ in the bending plane and θ₂ out of it, are drawn from N(0, θ₀). A hit at path length l_j is displaced by Σ_{i<j} θ_i (l_j − l_i), computed with prefix sums. The displacement is projected back onto the measuring surface (barrel: Δz = v/sin θ; disk: Δr = −v/cos θ).

**Approximation.** The kicks do not change the direction used for later crossings (second order in θ₀). Material is placed only at measurement surfaces; the beam pipe and support structures are not modelled.

**Validation.**
- The hit displacement scales as 1/p (ratio 3.5–4.5 for a 4× momentum change; `test_multiple_scattering_scales_as_one_over_p`).
- At pT = 3 GeV the resolution agrees with the uniform-scatterer estimate within a factor of two.

## Energy loss (not simulated)

A minimum-ionising muon loses about 1.66 MeV cm²/g. Over 0.4–1.8 X₀ of silicon-like material (X₀ = 9.37 cm, 2.33 g/cm³) that is roughly 15–65 MeV. CMS track fits correct for the mean loss, so simulating it without the correction would introduce an artificial bias. The remaining loss fluctuations are small compared with the multiple-scattering term above 3 GeV. This is listed in `research/LIMITATIONS.md`.

## Smearing and efficiency

- Each recorded hit is rotated by N(0, σ_rφ)/r in φ and shifted by N(0, σ_z) (barrel) or N(0, σ_r) (disk).
- Mono strips carry no longitudinal measurement.
- A crossing is recorded with probability equal to the hit efficiency. Its material still scatters the muon when the hit is lost.

## Resolution achieved (robust σ of pT_reco/pT_true − 1)

| pT | η = 0 | η = 2.0 | η = 2.35 |
|---|---|---|---|
| 5 GeV | 0.7 % | — | — |
| 45 GeV | 1.0 % | 1.8 % | 3.3 % |

These are of the same order as the published CMS tracker performance (about 1–2 % at 100 GeV centrally, worse forward; CMS, JINST 9 (2014) P10009). The agreement is qualitative only. The final numbers are in `outputs/detector_studies.json` and `figures/resolution_vs_pt.png`.

# Detector Model

Code: `particlelab/detector/geometry.py` and `simulation.py`.

> **Scope.** `cms_inspired()` is an **educational model** with the main features of the CMS silicon tracker. Layer radii, lengths, resolutions and material are rounded values in the spirit of CMS, JINST 3 (2008) S08004. It is **not** the CMS geometry and makes no claim of CMS-level fidelity.

## Geometry (CMS-inspired)

| Subsystem | Positions | Coverage | σ(rφ) | σ(z or r) | x/X₀ per layer |
|---|---|---|---|---|---|
| Pixel barrel ×3 | r = 4.4, 7.3, 10.2 cm | \|z\| < 26.5 cm | 10 µm | 20 µm | 0.0227 |
| Inner strip barrel ×4 | r = 25.5–49.8 cm | \|z\| < 70 cm | 30 µm | 0.5 mm (layers 1–2, stereo); none (mono) | 0.0399 |
| Outer strip barrel ×6 | r = 55.5–108 cm | \|z\| < 118 cm | 45 µm | 0.5 mm (layers 1–2); none | 0.0244 |
| Pixel disks ×2 per side | \|z\| = 34.5, 46.5 cm | 6 < r < 15 cm | 15 µm | 20 µm | 0.0100 |
| Inner disks ×3 per side | \|z\| = 80–100 cm | 20 < r < 50 cm | 35 µm | 1 mm | 0.0426 |
| Endcap disks ×9 per side | \|z\| = 1.24–2.72 m | 22 < r < 113 cm | 40 µm | 1 mm (odd); none (even) | 0.0519 |

Passive material (scatters the muon, records no hits):

| Surface | Position | x/X₀ |
|---|---|---|
| Beam pipe (0.8 mm Be, fixed) | r = 2.9 cm | 0.0023 |
| Pixel flange, inner | \|z\| = 28 cm, 4 < r < 12 cm | 0.162 |
| Pixel flange, outer | \|z\| = 28 cm, 12 < r < 22 cm | 0.049 |
| Pixel service cylinder | r = 18 cm, 0.5 < \|z\| < 2.8 m | 0.040 |
| Inner-barrel end services | \|z\| = 72 cm, 22 < r < 52 cm | 0.219 |
| Inner-barrel service cylinder | r = 53 cm, 12 < \|z\| < 60 cm / 60 < \|z\| < 110 cm | 0.035 / 0.192 |
| Inner-disk back services | \|z\| = 110 cm, 20 < r < 52 cm | 0.046 |
| Outer-barrel rod services | r = 85 cm, 20 < \|z\| < 70 cm / 70 < \|z\| < 118 cm | 0.106 / 0.171 |
| Outer-barrel end flange | \|z\| = 120 cm, 55 < r < 112 cm | 0.031 |

**Material calibration.** The thicknesses of the sensitive layers and passive surfaces are fitted by `calibration/fit_material.py`, one subsystem at a time, to the published CMS material budget X₀(η) (JINST 9 (2014) P10009, Fig. 2). The curve was digitised from the paper into `references/cms_tracker_material_x0.json`; the method is recorded in that file. The fit uses bounded linear least squares with relative residuals. No dimuon data or resolution information enters the fit. The support tube outside the tracker is not modelled, because it lies beyond the last measurement.

| \|η\| range | CMS (X₀) | model (X₀) | model / CMS |
|---|---|---|---|
| 0–0.5 | 0.46 | 0.48 | 1.05 |
| 0.5–1.0 | 0.78 | 0.73 | 0.93 |
| 1.0–1.5 | 1.64 | 1.44 | 0.88 |
| 1.5–2.0 | 1.44 | 1.30 | 0.91 |
| 2.0–2.5 | 1.05 | 1.06 | 1.00 |

CMS states that its simulated material agrees with data to better than 10 %. `test_material_budget_matches_the_cms_reference` keeps every 0.5-wide range within ±15 %.

Other settings:
- **Field:** B = 3.8 T uniform along z.
- **Material at η = 0:** 0.38 X₀ (CMS: about 0.4 X₀), rising to about 1.5–1.7 X₀ at \|η\| = 1.2–1.5.
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

Highland formula in the Lynch–Dahl form (V. L. Highland, NIM 129 (1975) 497; G. R. Lynch and O. I. Dahl, NIM B58 (1991) 6; PDG "Passage of particles through matter", Eq. 34.16):

θ₀ = (13.6 MeV / βcp) √(x/X₀) [1 + 0.038 ln(x/(X₀β²))], with x/X₀ → (x/X₀)/cos α for incidence angle α.

The PDG review states that adding the θ₀ of individual layers in quadrature "is systematically too small" and recommends applying the formula once to the combined scatterer. The model therefore uses the total thickness X = Σ (x_i/X₀)/cos α_i along each track in the logarithm, and gives each surface the share

θ_i² = (13.6 MeV / βcp)² (x_i/X₀)/cos α_i · [1 + 0.038 ln(X/β²)]²,

so that Σ θ_i² equals θ₀² of the combined scatterer. The earlier per-layer logarithm gave about 10 % less scattering at η = 0. `test_highland_is_applied_to_the_combined_thickness` checks that ten 1 % X₀ surfaces scatter like one 10 % X₀ surface, to 1 %.

At each crossed surface, including the passive ones, two independent kicks, θ₁ in the bending plane and θ₂ out of it, are drawn from N(0, θ_i). A hit at path length l_j is displaced by Σ_{i<j} θ_i (l_j − l_i), computed with prefix sums. The displacement is projected back onto the measuring surface (barrel: Δz = v/sin θ; disk: Δr = −v/cos θ).

**Approximation.** The kicks do not change the direction used for later crossings (second order in θ₀). Services are represented by a small number of thin passive surfaces; their positions are rounded and only their x/X₀ is fitted. Non-Gaussian single-scattering tails are not modelled (Highland describes the central 98 %).

**Validation.**
- The hit displacement scales as 1/p (ratio 3.5–4.5 for a 4× momentum change; `test_multiple_scattering_scales_as_one_over_p`).
- At pT = 3 GeV the resolution agrees with the uniform-scatterer estimate within a factor of two.

## Energy loss (not simulated)

A minimum-ionising muon loses about 1.66 MeV cm²/g. Over 0.4–1.8 X₀ of silicon-like material (X₀ = 9.37 cm, 2.33 g/cm³) that is roughly 15–65 MeV. CMS track fits correct for the mean loss, so simulating it without the correction would introduce an artificial bias. The remaining loss fluctuations are small compared with the multiple-scattering term above 3 GeV. This is listed in `research/LIMITATIONS.md`.

## Smearing and efficiency

- Each recorded hit is rotated by N(0, σ_rφ)/r in φ and shifted by N(0, σ_z) (barrel) or N(0, σ_r) (disk).
- Mono strips carry no longitudinal measurement.
- A crossing is recorded with probability equal to the hit efficiency. Its material still scatters the muon when the hit is lost.

## Resolution compared with CMS

σ(pT)/pT is the half-width of the central 68 % interval of pT_reco/pT_true − 1 for single muons. The CMS values are read from JINST 9 (2014) P10009, Fig. 14 (`references/cms_muon_pt_resolution.json`). The CMS tracks are not vertex-constrained, so the beam-spot point is switched off here; with it, the model resolution improves by 3–25 %.

| pT | \|η\| | CMS | model | model / CMS |
|---|---|---|---|---|
| 1 GeV | 0.05 | 0.70 % | 0.54 % | 0.78 |
| 1 GeV | 0.85 | 1.17 % | 0.83 % | 0.71 |
| 1 GeV | 1.65 | 1.76 % | 1.80 % | 1.02 |
| 1 GeV | 2.25 | 2.46 % | 2.55 % | 1.04 |
| 10 GeV | 0.05 | 0.76 % | 0.87 % | 1.14 |
| 10 GeV | 0.85 | 1.25 % | 1.18 % | 0.95 |
| 10 GeV | 1.65 | 1.58 % | 1.53 % | 0.97 |
| 10 GeV | 2.25 | 2.65 % | 2.53 % | 0.96 |
| 100 GeV | 0.05 | 1.96 % | 1.81 % | 0.92 |
| 100 GeV | 0.85 | 2.43 % | 2.01 % | 0.83 |
| 100 GeV | 1.65 | 3.54 % | 2.03 % | 0.57 |
| 100 GeV | 2.25 | 12.6 % | 6.5 % | 0.52 |

Before the material calibration and the Highland change, the model was 0.5–0.95 of CMS at 10 GeV.

**Remaining differences.**
- **10 GeV**, the momentum range of the J/ψ and Υ muons: within ±15 % at all η.
- **1 GeV, \|η\| < 1:** the model is about 25 % too precise. This is consistent with missing Molière tails and with material concentrated at a few surfaces.
- **100 GeV, \|η\| > 1.5:** the model is about 2× too precise. Material cannot explain this, since it now matches CMS there. The likely causes are the simplified forward geometry and hit resolution (rounded disk positions, no r-dependent strip pitch) and the absence of alignment uncertainty. This affects only the forward Z muons.

These values come from the CMS simulation, not from data. The full model curves are in `outputs/detector_studies.json` (`cms_reference_comparison`) and `outputs/figures/cms_reference_comparison.png`.

## Momentum-scale bias from the resolution

Because pT ∝ 1/κ and the curvature κ is measured with a symmetric error, the mean reconstructed pT is biased upwards by about (σ_κ/κ)². With σ(pT)/pT ≈ 1.5–2 %, this is a few × 10⁻⁴. The simulated Υ(1S) peak, for example, sits about 5 MeV above its generated mass. Real experiments remove this with a momentum-scale calibration; the model does not, and the effect is reported in `outputs/results.json` (`sim_mean_reco_minus_true_MeV`).

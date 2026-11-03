# CMS 2011 Data vs Simulation

Produced by `python run_analysis.py` (configs/default.json, seed 20261102). The full numbers are in `outputs/results.json` and `report.md`; figures are in `outputs/figures/`. **The simulation was not tuned to the data at any point.**

## Dataset

- **File:** CERN Open Data record 545, `Dimuon_DoubleMu.csv`. pp collisions at √s = 7 TeV, Run2011A, DoubleMu primary dataset. Licence CC0.
- **Integrity:** 13 935 840 bytes, adler32 `5b87a72a`, both matching the record.
- **Schema:** the 21 expected columns, exactly.
- **Producer selection:** two muons with |η| < 2.4, at least one global, opposite charge, 0.3 < M < 300 GeV.

### Validation (no rows silently dropped)

| Check | Rows |
|---|---|
| total | 100 000 |
| rejected (non-finite, charge ≠ ±1, unknown type, E < \|p\|, same sign) | 0 |
| warning: published `pt` ≠ √(px²+py²) (muon 1 / muon 2) | 2 886 / 3 218 |
| warning: published `eta` ≠ recomputed (muon 1 / muon 2) | 82 / 71 |
| published M vs M recomputed from four-vectors | max difference 0.6 MeV |

The published `pt` column evidently comes from a different momentum estimate than (px, py, pz). M is consistent with the four-vectors, so all cuts use recomputed pT and η (ADR-10).

### Common selection (data and simulation)

Both muons pT > 3 GeV and |η| < 2.4, opposite charge. Data: 93 469 events pass.

## Peak comparison

| | PDG M (GeV) | Data M (GeV) | Data − PDG | Sim M (GeV) | Sim − PDG | FWHM data / sim (MeV) | sim/data |
|---|---|---|---|---|---|---|---|
| J/ψ | 3.0969 | 3.0933 ± 0.0005 (stat) ± 0.0003 (syst) | −3.6 MeV | 3.0981 ± 0.0003 | +1.2 ± 0.3 MeV | 62 / 66 | 1.07 |
| Υ(1S) | 9.4604 | 9.4439 ± 0.0031 (stat) ± 0.0011 (syst) | −16.5 MeV | 9.4624 ± 0.0004 | +2.0 ± 0.4 MeV | 214 / 205 | 0.96 |
| Z | 91.1879 | 90.915 ± 0.076 (stat) ± 0.030 (syst) | −273 MeV | 91.193 ± 0.010 | +5 ± 10 MeV | 4381 / 4098 | 0.94 |

Before the audit fixes, the sim/data width ratios were 0.74, 0.66 and 0.79. The change comes only from calibrating the detector material to the published CMS material budget and applying the Highland formula to the combined thickness (ADR-16). **No parameter was adjusted to these data.**

**Data fits:**

| | Model | Fitted signal events | p-value |
|---|---|---|---|
| J/ψ | CB + Gaussian + exponential | 6360 ± 106 | 0.24 |
| Υ(1S) | Υ(1S, 2S, 3S) double Gaussians + exponential; 2S and 3S found at 10.019 and 10.346 GeV | 3111 ± 200 (1S) | 0.23 |
| Z | relativistic BW ⊗ CB + exponential | 5550 ± 165 | 0.63 |

**Simulation fits:** double Gaussian for J/ψ (p = 0.53) and Υ (p = 0.002), and relativistic BW ⊗ Gaussian for the Z (p = 0.04). The double Gaussian does not fully describe the simulated Υ shape (audit F-STA-5), so the Υ width comparison carries a model uncertainty. That is why the model-independent FWHM is quoted.

**Changes to the Z data fit.** The mass moved from 90.948 to 90.915 GeV and its statistical error from ±39 to ±76 MeV. The earlier fit stopped short of the minimum because the numerical line shape jittered with σ, and its error came from a noisy Hessian (audit F-STA-3, ADR-17). The toys below confirm that the new errors have the right size.

## Systematic uncertainty of the data masses

One fit choice is changed at a time. A variation counts if the minimiser converged and p > 0.01. The full list is in `outputs/report.md`.

| | window | binning | signal model | total syst | stat |
|---|---|---|---|---|---|
| J/ψ | 0.01 MeV | 0.18 MeV | 0.21 MeV (double Gaussian; CB alone rejected, p = 2×10⁻⁴) | 0.28 MeV | 0.48 MeV |
| Υ(1S) | 0.21 MeV | 0.71 MeV | 0.80 MeV (CB + Gaussian) | 1.1 MeV | 3.1 MeV |
| Z | 28 MeV | 11 MeV | — (BW ⊗ Gaussian rejected, p = 6×10⁻⁹; it would shift the mass by −218 MeV) | 30 MeV | 76 MeV |

## Data masses by detector region

Offset of the fitted mass from PDG, with statistical errors only. "Central" means |η| < 0.9.

| | both central | one central | both forward |
|---|---|---|---|
| J/ψ | −0.089 ± 0.016 % | −0.110 ± 0.036 % | −0.169 ± 0.029 % |
| Υ(1S) | −0.162 ± 0.035 % | (fit p < 10⁻³, not used) | −0.300 ± 0.075 % |
| Z | −0.23 ± 0.15 % | −0.25 ± 0.12 % | −0.39 ± 0.12 % |

Some J/ψ and Z region fits have a Hessian that is not positive definite. Their masses are valid, but their errors are approximate.

### Interpretation

**Measured facts:**
- The data peaks sit below PDG by 0.12 % (J/ψ), 0.17 % (Υ) and 0.30 % (Z). These offsets are many times the quoted uncertainties.
- For all three resonances, the offset is larger when both muons are forward than when both are central. For the J/ψ the difference is 0.08 ± 0.03 %.
- The simulation reproduces the PDG masses to within its own resolution-induced shift. The mean reconstructed − true mass is +1.4 ± 0.3, +3.1 ± 0.4 and +15 ± 5 MeV (see `physics/MASS_RECONSTRUCTION_AND_FITTING.md` and `physics/DETECTOR_MODEL.md`).

**Hypotheses (not tested here):**
- **Momentum scale.** These CSV files carry no muon momentum-scale corrections. An η-dependent scale error, for example from residual misalignment or the magnetic-field map, would explain the η dependence. This is the leading hypothesis.
- **Final-state radiation.** FSR pulls peak positions down, most for the Z. To first order it does not depend on η, so it cannot explain the regional pattern alone.
- **Z line shape.** The model has no parton luminosity or γ*/Z interference; these shift a fitted Z pole by tens of MeV.

No correction was applied to the data.

**Widths.** With calibrated material, the simulated widths are within 7 % of the data for all three resonances. The remaining differences are of the size expected from what the model still lacks:
- final-state radiation (the data tails);
- alignment;
- the forward high-pT resolution (TECHNICAL_DEBT D11);
- the different η mix of data and simulation (see below).

## Kinematic shapes (sideband-subtracted data vs simulation, unit area)

| | muon pT χ²/ndf | muon η χ²/ndf |
|---|---|---|
| J/ψ | 12 235 / 26 | 1 504 / 23 |
| Υ(1S) | 780 / 25 | 165 / 23 |
| Z | 675 / 26 | 401 / 23 |

**These shapes disagree strongly, and that is expected:**

- **J/ψ:** data muons peak at 6–8 GeV, while the simulated spectrum falls steeply from 3 GeV. The DoubleMu trigger thresholds favour harder muons, and the phenomenological pT spectrum is softer than real prompt plus non-prompt J/ψ production. Data muons are also more central (|η| < 1.2), consistent with trigger and muon-system acceptance, neither of which is simulated.
- **Z:** the simulation reproduces the Jacobian peak near M_Z/2 ≈ 45 GeV. It lacks the high-pT tail, which comes from Z transverse momentum due to QCD radiation (the simulation's Z pT spectrum is too soft) plus resolution tails.
- **Υ:** closest agreement, but the data are again harder.

Because the data and simulation have different pT and η mixes, the width comparison is not exactly like for like. This is part of why it is quoted only to about 10 %.

**Conclusion.** Mass reconstruction and detector resolution are now modelled to about 10 % in width. The data mass scale differs from PDG by 0.1–0.3 %, with an η dependence. Production kinematics and trigger effects are not modelled. The way to improve them is Pythia 8 production plus a trigger emulation (`research/LIMITATIONS.md`), not tuning the phenomenological spectra to these data.

## Simulation cut flow and acceptance

| | generated | both tracks with ≥ 8 hits | both fits ok, opposite charge | pT and η cuts (acceptance) |
|---|---|---|---|---|
| J/ψ | 1 000 000 | 556 098 | 556 098 | 12 168 (1.22 %) |
| Υ(1S) | 200 000 | 109 081 | 109 081 | 60 659 (30.3 %) |
| Z | 100 000 | 65 687 | 65 687 | 56 954 (57.0 %) |

The low J/ψ acceptance comes from the soft generated spectrum combined with the pT > 3 GeV cut on both muons. Acceptances depend on the production model (D4), so they are not used to predict yields.

## Fit validation (100 toys per model, mass parameter)

| | pull mean | pull width | failed fits |
|---|---|---|---|
| J/ψ (double Gaussian) | −0.15 ± 0.10 | 0.98 ± 0.07 | 0 |
| Υ(1S) (double Gaussian) | −0.11 ± 0.11 | 1.09 ± 0.08 | 0 |
| Z (rel. BW ⊗ Gaussian) | −0.01 ± 0.10 | 1.00 ± 0.07 | 0 |

The pull means are consistent with zero and the widths with one. The fits are therefore unbiased and their errors have the right size, within what 100 toys can resolve (about ±7 % on a width). These toys are generated from the fitted model itself. Toys with an independent truth are in the audit (`audit/a4_fits_data_toys.py`).

## Detector performance (σ(pT)/pT, central 68 %, CMS-inspired model, with the beam-spot point)

| pT (GeV) | 3 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| η = 0 | 0.80 % | 0.83 % | 0.85 % | 0.89 % | 1.12 % | 1.71 % | 3.08 % |
| η = 1 | 1.45 % | 1.47 % | 1.46 % | 1.48 % | 1.63 % | 2.08 % | 3.26 % |
| η = 2 | 1.95 % | 1.92 % | 1.91 % | 2.02 % | 2.34 % | 3.28 % | 5.81 % |
| η = 0, no scattering | 0.04 % | 0.07 % | 0.15 % | 0.29 % | 0.75 % | 1.49 % | 2.98 % |

Below about 30 GeV, multiple scattering sets a constant floor; above that, hit resolution makes σ(pT)/pT grow linearly with pT. The uniform reference model agrees with the Gluckstern formula to a ratio of 0.96–0.98. The comparison with the CMS single-muon resolution is in `physics/DETECTOR_MODEL.md` and `outputs/figures/cms_reference_comparison.png`.

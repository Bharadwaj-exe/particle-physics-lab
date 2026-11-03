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
| J/ψ | 3.0969 | 3.0933 ± 0.0005 | −3.6 ± 0.5 MeV | 3.0973 ± 0.0002 | +0.4 ± 0.2 MeV | 62 / 46 | 0.74 |
| Υ(1S) | 9.4604 | 9.4439 ± 0.0031 | −16.5 ± 3.1 MeV | 9.4615 ± 0.0003 | +1.1 ± 0.3 MeV | 214 / 142 | 0.66 |
| Z | 91.1879 | 90.948 ± 0.039 | −240 ± 39 MeV | 91.190 ± 0.009 | +1.7 ± 9.1 MeV | 4438 / 3523 | 0.79 |

**Data fits:**

| | Model | Fitted signal events | p-value |
|---|---|---|---|
| J/ψ | CB + Gaussian + exponential | 6360 ± 106 | 0.24 |
| Υ(1S) | Υ(1S, 2S, 3S) double Gaussians + exponential; 2S and 3S found at 10.019 and 10.346 GeV | 3111 ± 200 (1S) | 0.23 |
| Z | relativistic BW ⊗ CB + exponential | 5321 ± 82 | 0.36 |

**Simulation fits:** double Gaussian (J/ψ, Υ) and relativistic BW ⊗ Gaussian (Z).

### Interpretation

- **The simulation reproduces the PDG masses** to within its own second-order resolution shift: a reconstructed − true mean of +0.6, +1.4 and +13 MeV, explained in MASS_RECONSTRUCTION_AND_FITTING.
- **The data peaks sit below PDG** by 0.12 % (J/ψ), 0.17 % (Υ) and 0.26 % (Z). The likely causes are final-state radiation, which pulls peak-based estimators down (most visibly for Z, whose low-mass tail needs the CB term), and the uncalibrated momentum scale of this derived dataset. CMS applies dedicated muon momentum corrections that are not part of these CSV files. This is a measurement of the dataset, not a model failure; no correction was applied.
- **The simulated peaks are 21–34 % narrower than the data.** Missing in the model:
  - final-state radiation;
  - material beyond the measuring layers, and the real material distribution;
  - alignment and calibration uncertainties;
  - the larger forward fraction in the data (see below);
  - the scattering-aware fit that real tracking uses.

  Before multiple scattering was added, the simulated J/ψ peak was about 10× too narrow. It is now 0.74 × the data width.

## Kinematic shapes (sideband-subtracted data vs simulation, unit area)

| | muon pT χ²/ndf | muon η χ²/ndf |
|---|---|---|
| J/ψ | 12 396 / 26 | 1 539 / 23 |
| Υ(1S) | 782 / 25 | 160 / 23 |
| Z | 682 / 26 | 392 / 23 |

**These shapes disagree strongly, and that is expected:**

- **J/ψ:** data muons peak at 6–8 GeV, while the simulated spectrum falls steeply from 3 GeV. The DoubleMu trigger thresholds favour harder muons, and the phenomenological pT spectrum is softer than real prompt plus non-prompt J/ψ production. Data muons are also more central (|η| < 1.2), consistent with trigger and muon-system acceptance, neither of which is simulated.
- **Z:** the simulation reproduces the Jacobian peak near M_Z/2 ≈ 45 GeV. It lacks the high-pT tail, which comes from Z transverse momentum due to QCD radiation (the simulation's Z pT spectrum is too soft) plus resolution tails.
- **Υ:** closest agreement, but the data are again harder.

**Conclusion.** Mass reconstruction and detector resolution are modelled at the 20–35 % level in width and 0.1–0.3 % in scale. Production kinematics and trigger effects are not modelled. The way to improve them is Pythia 8 production plus a trigger emulation (`research/LIMITATIONS.md`), not tuning the phenomenological spectra to these data.

## Simulation cut flow and acceptance

| | generated | both tracks with ≥ 8 hits | both fits ok, opposite charge | pT and η cuts (acceptance) |
|---|---|---|---|---|
| J/ψ | 1 000 000 | 556 110 | 556 110 | 12 184 (1.22 %) |
| Υ(1S) | 200 000 | 109 026 | 109 026 | 60 637 (30.3 %) |
| Z | 100 000 | 65 692 | 65 692 | 56 948 (57.0 %) |

The low J/ψ acceptance comes from the soft generated spectrum combined with the pT > 3 GeV cut on both muons. Being model-dependent (D4), acceptances are not used to predict yields.

## Fit validation (100 toys per model, mass parameter)

| | pull mean | pull width | failed fits |
|---|---|---|---|
| J/ψ (double Gaussian) | −0.04 ± 0.09 | 0.95 | 0 |
| Υ(1S) (double Gaussian) | −0.04 ± 0.09 | 0.93 | 0 |
| Z (rel. BW ⊗ Gaussian) | +0.14 ± 0.09 | 0.93 | 2 |

The fits are unbiased, and the Hessian errors are correct to within about 7 % (slightly conservative). An earlier configuration fitted the simulated J/ψ with CB + Gaussian; 65 of its 100 toys had singular Hessians because simulation has no radiative tail to constrain the CB parameters. That was the reason for switching to the double Gaussian.

## Detector performance (σ(pT)/pT, central 68 %, CMS-inspired model)

| pT (GeV) | 3 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| η = 0 | 0.67 % | 0.70 % | 0.71 % | 0.77 % | 1.02 % | 1.63 % | 3.03 % |
| η = 1 | 0.91 % | 0.91 % | 0.92 % | 0.96 % | 1.19 % | 1.71 % | 3.10 % |
| η = 2 | 1.40 % | 1.38 % | 1.39 % | 1.46 % | 1.93 % | 2.99 % | 5.43 % |
| η = 0, no scattering | 0.04 % | 0.07 % | 0.15 % | 0.29 % | 0.73 % | 1.48 % | 2.97 % |

Below about 30 GeV, multiple scattering sets a constant floor; above that, hit resolution makes σ(pT)/pT grow linearly with pT. The uniform reference model agrees with the Gluckstern formula to a ratio of 0.95–0.99.

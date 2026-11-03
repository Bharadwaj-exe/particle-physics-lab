# Speed and accuracy against a fast-simulation baseline

Produced by `python benchmarks/compare_baselines.py`. Results are in `outputs/baseline_comparison.json`.

**Machine:** Intel N150 (4 cores, low power), 16 GB RAM, Windows 11, Python 3.11, one process.

## Baseline

The Delphes CMS card (`cards/delphes_card_CMS.tcl`) has two muon modules:

- `MuonMomentumSmearing`: relative pT resolution in three |η| bands. Delphes draws the smeared pT from a log-normal distribution.
- `MuonTrackingEfficiency`.

The benchmark re-implements both formulas and applies them to **the same generated muons** as this simulation. Both samples use the same cuts and the same fit, so the detector model is the only difference.

This is a Delphes-style baseline, not a Delphes run:

- No ROOT, no Delphes executable and no particle propagator were used.
- The card describes present-day CMS, not the 2011 detector.

## Accuracy: resolution width compared with CMS 2011 data

The ratio is the fitted resolution width divided by the data width. 1 means a perfect match, and the ± is the data's statistical uncertainty.

| Peak | Region | Data σ (MeV) | This simulation | Delphes-style |
|---|---|---|---|---|
| J/ψ | all | 26.3 ± 0.8 | **1.07 ± 0.03** | 1.24 ± 0.04 |
| J/ψ | both muons with \|η\| < 0.9 | 20.8 ± 0.7 | **1.00 ± 0.04** | 1.16 ± 0.04 |
| J/ψ | both muons with \|η\| > 1.2 | 41.3 ± 1.9 | **1.02 ± 0.05** | 1.20 ± 0.06 |
| Υ(1S) | all | 91.0 ± 3.5 | **0.96 ± 0.04** | 1.08 ± 0.04 |
| Υ(1S) | both muons with \|η\| < 0.9 | 67.1 ± 3.5 | **0.97 ± 0.05** | 1.18 ± 0.06 |
| Υ(1S) | both muons with \|η\| > 1.2 | 138 ± 16 | 0.95 ± 0.11 | 1.09 ± 0.13 |
| Z | all | 1116 ± 92 | 0.95 ± 0.08 | 1.11 ± 0.09 |
| Z | both muons with \|η\| < 0.9 | 917 ± 171 | 0.73 ± 0.14 | 0.95 ± 0.18 |
| Z | both muons with \|η\| > 1.2 | 1512 ± 135 | 0.93 ± 0.08 | 1.13 ± 0.10 |

**Reading:**

- **J/ψ and Υ (muons of a few GeV).** The hit-level model matches the data to within 0–7 %. The Delphes CMS resolution is 8–24 % too wide, which is 2–4.5 standard deviations of the data for J/ψ.
- **J/ψ and Υ muon identification.** The card's identification module also requires pT > 10 GeV, so it keeps only 0.4 % of selected J/ψ events and 0.3 % of Υ events. The default card does not describe these peaks.
- **Z (muons of 30–50 GeV).** Both models agree with the data within their uncertainties. For central Z decays this model is 27 % narrower than the data (2 standard deviations), which is consistent with the known high-pT gap (LIMITATIONS, D11).
- **Not tuned to these data.** The model's material comes from the published CMS material map (`calibration/fit_material.py`), not from these dimuon events.

## Speed

| | J/ψ | Υ(1S) | Z |
|---|---|---|---|
| This simulation, full chain (generated events/s) | 5,200 | 5,400 | 5,300 |
| Delphes-style smearing (generated events/s) | 254,000 | 235,000 | 128,000 |
| Stage shares (hits / track fits) | 53 % / 46 % | 47 % / 52 % | 46 % / 51 % |
| Time to update the mass plot, 100,000 events | 20.9 s | 18.6 s | 20.8 s |

Parametric smearing is 25–50× faster because it skips hits and fits entirely. Speed is the price this model pays for the accuracy above.

## Skipping events that cannot pass the cuts

The test skips events where either true muon has pT below 0.8 × pT_min or |η| above η_max + 0.1. It then simulates hits and fits for the remaining events only, and checks on the same events how many finally selected events were thrown away.

| | J/ψ | Υ(1S) | Z |
|---|---|---|---|
| Speed-up | **34×** | 2.2× | 1.5× |
| Selected events lost (out of 1,215 / 30,370 / 56,954) | 0 | 0 | 0 |

This is now built in as `simulate(..., preselect=True)` and the GUI uses it. The margins grow with the detector resolution, see ADR-21; a fixed 0.8 × pT_min margin was not safe at B = 0.5 T.

## Compiled kernels and pre-selection in the GUI

Seconds to simulate 100,000 generated events (the mass fit adds 0.1–0.3 s; same machine, 4 threads; NumPy timings vary by about ±15 % between runs on this laptop). The selected events and masses are identical between NumPy and Rust.

| | NumPy | Rust | NumPy + pre-selection | Rust + pre-selection |
|---|---|---|---|---|
| J/ψ | 24.3 | 3.07 (7.9×) | 0.94 | **0.31 (78×)** |
| Υ(1S) | 20.7 | 1.96 (10.6×) | 11.1 | **0.90 (23×)** |
| Z | 23.2 | 2.57 (9.0×) | 18.0 | **1.74 (13×)** |

| 100,000 tracks | NumPy | Rust, 1 thread | Rust, 4 threads |
|---|---|---|---|
| hit simulation | 4.3–6.1 s | 2.1 s | 0.81 s |
| track fits | 4.9–6.2 s | 0.58 s | 0.20 s |

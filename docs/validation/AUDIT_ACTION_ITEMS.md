# Audit Action Items

These come from `FINAL_INDEPENDENT_AUDIT.md` (baseline commit `440ef55`). None has been implemented. The list is ordered by priority.

- **Priority:** P1 = fix before presenting quantitative results; P2 = fix in the next development phase; P3 = optional or housekeeping.
- **Effort:** S = under 2 hours, M = half a day to a day, L = several days.

**Rule for every item:** do not tune any parameter to the CMS dimuon peaks or spectra. Calibrate only against independent references, such as the published material budget or the published single-track resolution. Then compare with the data again as a test.

## P1 — scientific correctness of reported results

| # | Item | Finding | Why it matters scientifically | Depends on | Effort |
|---|---|---|---|---|---|
| A1 | Rebuild the material model from the published CMS X₀(η) curve: per-layer x/X₀ plus passive material such as supports, cables and the beam pipe. Validate against the curve with a new test. | F-DET-3, F-DET-6 | The forward material is 2–2.5× too low, and the low-pT resolution is about 2× better than CMS. This is the main suspect for the data/simulation width ratios of 0.66–0.79. | — | M |
| A2 | Decide and document how the Highland logarithmic term is applied: to the total thickness along the track (the PDG recommendation, to be quoted verbatim) or per layer. Add a test of the θ₀ amplitude. | F-DET-4 | The per-layer treatment gives about 10 % less scattering at η = 0. | — | S |
| A3 | Re-measure the single-muon σ(pT)/pT against η and pT after A1 and A2, and compare with the CMS published values, defined the same way. Only then re-run the data comparison. | F-DET-6 | Separates the detector realism from the other causes of the width difference. | A1, A2 | M |
| A4 | Make the fit fallback bounded: L-BFGS-B restarts, a bounded Nelder–Mead or a transformed parameterisation. Return a failed `FitResult` instead of raising. Report the optimiser status separately from the Hessian check. | F-STA-3, F-ARC-4 | An unbounded fallback crashed on a real data subsample. It can also return clipped, non-minimum parameters with misleading errors. | — | S |
| A5 | Quote systematic errors on the data masses from fit-window and signal-model variations. Report the masses in η categories. | F-DAT-3 | The Z mass moves by 46 MeV across windows (compared with ±39 MeV statistical), and the offsets depend on η. The current numbers carry statistical errors only. | A4 (the category fits need a robust fitter) | M |

## P2 — interpretation and documentation of results

| # | Item | Finding | Why it matters scientifically | Depends on | Effort |
|---|---|---|---|---|---|
| A6 | Rewrite the mass-offset interpretation in CMS_DATA_COMPARISON.md. Separate the measured facts (η dependence, model dependence) from the hypotheses (momentum scale, FSR, missing parton luminosity of about −30 MeV at the Z). | F-DAT-3 | The current text attributes the offsets to causes without evidence, and the η dependence argues against an FSR-only explanation. | A5 | S |
| A7 | Give the Υ width ratio with a background-model systematic: compare the fitted ratio (0.66) with the sideband-subtracted one (0.87). Find a simulation shape that fits; the double Gaussian has p = 0.004. | F-STA-5 | The Υ width conclusion is not robust as stated. | A4 | M |
| A8 | Replace "the errors are about 7 % conservative" with "pull widths consistent with 1 (0.93–0.95 ± 0.07)". Add the independent-truth toy results. | F-TOY-1 | The claim is not supported by 100 toys. | — | S |
| A9 | Validate width uncertainties with toys or profile likelihoods for the CB-type shapes, or stop quoting them. | F-STA-4, D9 | Parameter correlations reach 0.96, so Hessian width errors are unreliable. | A4 | M |
| A10 | Update TECHNICAL_DEBT D1 with the measured offset ((−2.8 ± 0.6) × 10⁻⁵, smearing only, η = 2.3) and its cause: the disk radial error is not projected into the circle-fit weights. Optionally add that projection. | F-REC-2 | Corrects an overstated debt entry (−1 × 10⁻⁴) and records the mechanism. | — | S (docs) / M (fit change) |
| A11 | State in PRODUCTION_AND_DECAY.md that quarkonium isotropy is measured only for pT > 10–14 GeV and \|y\| < 1.2, and that the Z decay has no A₀/A₄ terms and no γ*/Z interference. | F-PHY-1, F-PHY-2 | These are approximations outside the measured region. | — | S |
| A12 | Regenerate `outputs/` from a clean commit. Make `run_record()` refuse a dirty tree, or store `git diff`. | F-REP-1 | The published run record does not identify the code that produced it. | Do it after any P1 changes. | S |
| A13 | Add the analysis settings (`DATA_FIT`, `SIM_FIT`, windows, bins, sideband definitions), a SHA-256 of `particles_pdg.json`, and the PyVista/VTK versions to the run record. | F-ARC-2, F-REP-2 | Full reproducibility without relying on the Git history. | — | S |
| A14 | In `compare()`, centre the simulated signal region on the simulation's own fitted mass, and account for signal leaking into the sidebands. | F-STA-6 | Small bias in the kinematic comparison, most visible for the Z. | — | S |
| A15 | Add to REFERENCES.md: the PDG author list and DOI, the CMS record's "not suitable for a full physics analysis" caveat, and the reference statuses from Appendix A of the audit. Verify Chernov (2010) and Gaiser (SLAC-R-255). | Appendix A | Citation completeness. | — | S |

## P3 — robustness and housekeeping

| # | Item | Finding | Depends on | Effort |
|---|---|---|---|---|
| A16 | Replace "→" in `run_analysis.py` log output with ASCII, or reconfigure stdout to UTF-8. Add a smoke test that runs `configs/quick.json` with redirected output. | F-ARC-3 | — | S |
| A17 | Close the PyVista plotter from `MainWindow.closeEvent`: a tab's own `closeEvent` is never called. | F-VIS-6 | — | S |
| A18 | Remove the latent truth path in the z-fit fallback: exclude unsmeared mono-layer z, or fail the track. | F-REC-6 | — | S |
| A19 | Remove the unused `s_true` in `fit_tracks`. | F-ARC-7 | — | S |
| A20 | Document the display conventions: the reconstructed helix is drawn from (0, 0, z₀) with d₀ ignored and over the true arc length, and the ZR side comes from the true path. Quote the 3D update time as a range (about 60–130 ms). | F-VIS-3, F-VIS-4, F-VIS-5 | — | S |
| A21 | Add the missing tests: Highland amplitude, material against a reference, bounded fallback, same selection on data and simulation, forward curvature bias, and 2D widget smoke tests (with pytest-qt). | §15 | A1, A2, A4 | M |
| A22 | Optional, physics extensions: Pythia 8 production with a trigger emulation (D4), FSR (e.g. PHOTOS-like or a parametrised model), and non-prompt J/ψ. Do these only after A1–A5. | §11 | A1–A5 | L |

## Suggested order

1. A4 and A16: robust fitting and the output crash.
2. A2, then A1, then A3: the detector material and resolution chain.
3. A5, A6, A7 and A9: statistics and interpretation, re-run on the improved detector.
4. A12 and A13: regenerate all outputs from a clean commit with complete provenance.
5. The remaining documentation and housekeeping items.

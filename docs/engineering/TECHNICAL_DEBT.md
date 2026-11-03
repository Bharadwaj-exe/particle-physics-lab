# Technical Debt and Open Discrepancies

Recorded rather than hidden. Priority as in the audit (P1 critical … P3 optional).

| # | Pri | Item | Evidence | Suggested resolution |
|---|---|---|---|---|
| D1 | P2 | Residual curvature offset of about −1×10⁻⁴ (≈ 2σ) for 5 GeV muons at \|η\| ≈ 2.3 | single-track study, 80k tracks | Check the linearised scattering projection on disks and the fit's treatment of disk r-errors; compare with a step-wise propagator |
| D2 | P2 | Fit weights ignore scattering correlations, so the hit-only σ(pT) underestimates the spread at low pT | `TrackFit.sigma_pt` | Add the scattering covariance matrix to the fit (generalised least squares) or a Kalman filter |
| D3 | P2 | Scattering kicks do not alter the direction used for later crossings | `simulation.py` docstring | Step-wise helix propagation between surfaces |
| D4 | P2 | Production spectra are phenomenological; acceptance is model-dependent | `production.py` | Pythia 8 backend (optional dependency) or published dσ/dpT dy tables |
| D5 | P2 | No momentum-scale calibration; the reconstructed peak sits about (3/8)Σσ² above the true mass | Υ: +1.5 ± 0.3 MeV | Report it (done). Optionally calibrate on the simulated Υ |
| D6 | P3 | The GUI is checked with screenshots only, with no automated widget tests | — | pytest-qt smoke tests |
| D7 | P3 | The Υ multi-peak fit takes about 25 s (Nelder–Mead fallback) | timing log | Analytic gradients, or iminuit as an optional fitter |
| D8 | P3 | Only J/ψ, Υ(1S), Z and their decay values are verified against PDG Live | `particles_pdg.json` `verified` flags | Verify the remaining entries |
| D9 | P3 | Width uncertainty for the CB-based shapes is propagated from the Hessian by finite differences of the FWHM; σ errors for heavily correlated CB parameters can be unreliable (e.g. 0.4 MeV for Z data) | `FitResult._width` | Validate with toys, or report profile-likelihood intervals |
| D10 | P3 | Multiprocessing gives no speed-up on the reference laptop | benchmark | Re-check on other hardware; keep the default at 1 process |

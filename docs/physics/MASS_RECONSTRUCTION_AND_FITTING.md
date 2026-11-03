# Mass Reconstruction and Statistical Analysis

Code: `particlelab/analysis/fitting.py`, `shapes.py`, `studies.py` and `compare.py`.

## Mass and its resolution

The reconstructed mass is m(μ⁺μ⁻) from the fitted momenta (cancellation-free formula, see PRODUCTION_AND_DECAY §4). To first order:

σ_m/m ≈ ½ √[(σ_p1/p1)² + (σ_p2/p2)²] ⊕ (opening-angle term)

**Second-order shift.** Curvature errors are Gaussian, so ⟨1/(1+ε)⟩ > 1 and ⟨m_reco/m_true⟩ − 1 ≈ (3/8)(σ₁² + σ₂²) > 0. For Υ this is about +1 MeV, as measured in simulation: +1.5 ± 0.3 MeV mean shift. Experiments remove it with a momentum-scale calibration; this model does not. It is reported, not corrected.

## Extended binned likelihood

−ln L = Σᵢ [μᵢ − nᵢ ln μᵢ], with μᵢ = N_s ∫_bin S + N_b ∫_bin B, where:
- each bin integral uses 5 sub-points;
- shapes are normalised numerically on the fit window, so N_s and N_b are event counts.

**Minimisation:** L-BFGS-B with bounds on scaled parameters, with Nelder–Mead only if it does not converge.
**Uncertainties:** inverse of the numerical (central-difference) Hessian of −ln L. A fit is flagged *not converged* when the Hessian is not positive definite, and parameters at a bound are reported.
**Goodness of fit:** Poisson deviance D = 2Σ[μ − n + n ln(n/μ)] (Baker & Cousins, NIM 221 (1984) 437) against χ²(N_bins − N_par).

## Line shapes (selected by physics and fit quality)

| Shape | Formula | Used for |
|---|---|---|
| Gaussian | — | tests |
| Double Gaussian | f·G(σ) + (1−f)·G(rσ), common mean | Υ: the resolution varies event by event, so a sum of Gaussians |
| Crystal Ball | Gaussian core, power-law tail below −α (Oreglia, SLAC-R-236 (1980)) | — |
| CB + Gaussian | f·CB + (1−f)·G | J/ψ: radiative tail plus resolution spread |
| Relativistic BW ⊗ Gaussian | s-dependent-width BW convolved on an aligned grid | Z, simulation |
| Relativistic BW ⊗ CB | as above, with a CB kernel | Z, data (absorbs the final-state-radiation tail) |
| Fixed-width Voigt | scipy `voigt_profile` | kept only to demonstrate the ~34 MeV convention bias |

**Background:** exponential, normalised on the window (slope of either sign).
**Υ data:** the 2S and 3S peaks share the 1S shape, with σ scaled by μᵢ/μ₁, and have free masses and yields.

**Convolution correctness.** The kernel is sampled at offsets k·step with k = −K…K, so offset 0 sits on a grid point and introduces no shift. A first version used `mode="same"` with an even-length kernel, which could shift the peak by half a step (up to 50 MeV for the Z). It was found in review and fixed, and `test_rbw_convolution_has_no_shift` guards it.

## Validation of the fitter

| Check | Result |
|---|---|
| Gaussian + exponential toys (200), mass pull | mean 0.02, width 1.06 |
| Same toys, signal-yield pull | mean −0.08, width 0.995 |
| Toy studies of each simulation fit model (100 toys) | `outputs/fit_validation.json` |
| Truth-level Z, s-dependent BW fit | 91.1847 ± 0.0028 GeV (PDG 91.1879) |
| Truth-level Z, fixed-width Voigt fit | +35 MeV high |
| Sparse histogram (300 events, 200 bins) | mass recovered |
| Three-peak Υ toy | 1S mass, 2S mass and 2S/1S ratio recovered |

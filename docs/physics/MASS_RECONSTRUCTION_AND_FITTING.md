# Mass Reconstruction and Statistical Analysis

Code: `particlelab/analysis/fitting.py`, `shapes.py`, `studies.py`, `compare.py` and `systematics.py`.

## Mass and its resolution

The reconstructed mass is m(μ⁺μ⁻) from the fitted momenta (cancellation-free formula, see PRODUCTION_AND_DECAY §4). To first order:

σ_m/m ≈ ½ √[(σ_p1/p1)² + (σ_p2/p2)²] ⊕ (opening-angle term)

**Second-order shift.** Curvature errors are Gaussian, so ⟨1/(1+ε)⟩ > 1 and ⟨m_reco/m_true⟩ − 1 ≈ (3/8)(σ₁² + σ₂²) > 0. In the full simulation the mean reconstructed − true mass is +1.4 ± 0.3 MeV (J/ψ), +3.1 ± 0.4 MeV (Υ) and +15 ± 5 MeV (Z). That is (2–5) × 10⁻⁴, larger than the Gaussian estimate because scattering tails and the forward tracks contribute. Experiments remove it with a momentum-scale calibration; this model does not. It is reported, not corrected.

## Extended binned likelihood

−ln L = Σᵢ [μᵢ − nᵢ ln μᵢ], with μᵢ = N_s ∫_bin S + N_b ∫_bin B, where:
- each bin integral uses 5 sub-points;
- shapes are normalised numerically on the fit window, so N_s and N_b are event counts.

**Minimisation:** L-BFGS-B with bounds on scaled parameters. If it does not report success, two bounded restarts follow, then a bounded Nelder–Mead and a final L-BFGS-B step. No step can leave the physical region (σ > 0, α > 0, n > 1); a point where the model cannot be evaluated returns a large finite −ln L instead of raising (ADR-17).
**Uncertainties:** inverse of the numerical (central-difference) Hessian of −ln L. Each fit reports `optimizer_ok` (the minimiser converged) and `hessian_ok` (the Hessian is positive definite) separately; `converged` requires both. Parameters at a bound are listed.
**Goodness of fit:** Poisson deviance D = 2Σ[μ − n + n ln(n/μ)] (Baker & Cousins, NIM 221 (1984) 437) against χ²(N_bins − N_par).

## Line shapes (selected by physics and fit quality)

| Shape | Formula | Used for |
|---|---|---|
| Gaussian | — | tests |
| Double Gaussian | f·G(σ) + (1−f)·G(rσ), common mean | Υ: the resolution varies event by event, so a sum of Gaussians |
| Crystal Ball | Gaussian core, power-law tail below −α (Oreglia, SLAC-R-236 (1980)) | — |
| CB + Gaussian | f·CB + (1−f)·G | J/ψ: radiative tail plus resolution spread |
| Relativistic BW ⊗ Gaussian | s-dependent-width BW convolved on a fixed grid (FFT) | Z, simulation |
| Relativistic BW ⊗ CB | as above, with a CB kernel | Z, data (absorbs the final-state-radiation tail) |
| Fixed-width Voigt | scipy `voigt_profile` | kept only to demonstrate the ~34 MeV convention bias |

**Background:** exponential, normalised on the window (slope of either sign).
**Υ data:** the 2S and 3S peaks share the 1S shape, with σ scaled by μᵢ/μ₁, and have free masses and yields.

**Convolution correctness.** The kernel is sampled at offsets k·step with k = −K…K, so offset 0 sits on a grid point and introduces no shift. A first version used `mode="same"` with an even-length kernel, which could shift the peak by half a step (up to 50 MeV for the Z). It was found in review and fixed, and `test_rbw_convolution_has_no_shift` guards it.

**Smoothness in the parameters.** The grid nodes sit at integer multiples of a power-of-two step (at least 20 points per min(σ, Γ)), and the extents do not depend on the parameters: ±8σ for the Gaussian kernel, and a fixed 30 GeV for the Crystal Ball tail, which is not negligible at any fixed number of σ. An earlier grid moved with σ, which made −ln L jitter by about 0.1 and broke the convergence test and the Hessian (audit F-STA-3). `test_rbw_line_shapes_are_smooth_in_sigma` guards this.

## Systematic uncertainties of the data masses

`analysis/systematics.py` refits the data with one choice changed at a time:
- **window:** narrower and wider windows, with the same bin width;
- **binning:** half and double the number of bins;
- **model:** alternative signal shapes (J/ψ: double Gaussian, Crystal Ball; Υ: CB + Gaussian; Z: BW ⊗ Gaussian).

A variation counts only if the minimiser converged and p > 0.01. A shape that fails the goodness-of-fit test, such as BW ⊗ Gaussian on the Z data (p ≈ 10⁻⁸), is reported but does not define an uncertainty. A Hessian that is not positive definite is allowed, because the CB-type tail parameters have nearly flat directions (correlations up to 0.96) that do not affect the mass.

The largest accepted shift in each category is taken, and the categories are added in quadrature. Window variations change the sample, so they include a statistical part and the estimate is conservative.

The same nominal fit is also run in three detector regions (both muons |η| < 0.9, one, or neither), with statistical errors only. Results: `outputs/results.json` (`data_mass_systematics`, `data_mass_by_region`) and `outputs/report.md`.

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

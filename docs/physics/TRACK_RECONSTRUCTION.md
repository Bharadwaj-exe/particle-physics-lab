# Track Reconstruction

Code: `particlelab/detector/tracking.py`.

## Parameters

Each fitted track gives:
- pT (curvature R = pT/0.2998B);
- charge q;
- azimuth φ₀;
- cot θ (dip angle);
- transverse impact parameter d₀ = √(a² + b²) − R;
- longitudinal impact parameter z₀;
- the hit-error σ(pT);
- transverse χ²/ndf.

## Transverse fit

1. **Algebraic start (Kåsa).** x² + y² = 2ax + 2by + c is linear in (a, b, c). It is solved by weighted least squares with w = 1/σ_rφ². This fit is known to be biased towards small radii (Chernov, *Circular and Linear Regression*, 2010).
2. **Geometric refinement.** Gauss–Newton on Σ wᵢ(|hᵢ − c| − R)², up to 6 iterations, stopping when |ΔR|/R < 1e-11. The baseline showed the algebraic-only fit biased pT by −2e-4, which is −20 MeV on the Z mass; the refinement removes it.
3. **Beam-spot constraint.** A pseudo-hit at (0, 0) with σ = 20 µm (CMS-inspired model only).
4. **Covariance.** (JᵀWJ)⁻¹ at the minimum gives σ_R, hence σ(pT).
5. **Charge.** The sign of the turning direction between the first and last recorded hits (ordered by arc length).

## Longitudinal fit

z = z₀ + s cot θ, with s = 2R arcsin(ρ/2R), by weighted least squares:
- barrel hits use w = 1/σ_z² (mono strips w = 0);
- disk hits have exact z and an uncertain radius, so σ_z,equiv = σ_r |cot θ| (two passes);
- with fewer than two usable hits the fit falls back to unweighted.

## Known limitations

- Multiple-scattering correlations are not in the weights. The fit is therefore unbiased but not optimal at low pT, and σ(pT) from the covariance covers hit errors only (pull width 1.0 without material, larger with it). A Kalman filter (Frühwirth, NIM A262 (1987) 444) would solve both.
- No pattern recognition: hits are assigned to the correct track by construction. There are no fakes or outlier rejection.

## Validation

| Test | Result |
|---|---|
| Ideal detector (no smearing, no scattering): p, q and z₀ recovered for 3000 random tracks | exact to 1e-7 |
| Uniform barrel vs Gluckstern σ(pT)/pT = σ pT/(0.3BL²) √(720/(N+4)) at 5–100 GeV | ratio within 0.9–1.1 (0.94–0.99 measured) |
| Hit-error pulls without material | width 0.9–1.1 |
| Curvature 1/pT unbiased with full material | within 4 standard errors |
| ⟨pT⟩ bias equals var(1/pT) (Gaussian curvature errors) | within 4 standard errors |
| Forward region (\|η\| ≈ 2.3): residual curvature offset | about −1e-4 (2σ), an open item in TECHNICAL_DEBT |

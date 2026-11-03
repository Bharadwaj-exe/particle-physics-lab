# Production, Decay Kinematics and Lorentz Transformations

Units: GeV, c = 1. Four-vectors are (E, px, py, pz). Code: `particlelab/physics/`.

## 1. Production (phenomenological)

**Mathematics.**
- dN/dpT ∝ pT (1 + pT²/p₀²)^(−n), sampled exactly by pT = p₀ √((1−u)^(1/(1−n)) − 1).
- dN/dy is a trapezoid: flat for |y| < y_flat, falling linearly to 0 at y_max.
- φ is uniform.
- Lab four-momentum: E = m_T cosh y, p_z = m_T sinh y, with m_T = √(m² + pT²).

| | p₀ (GeV) | n | y_flat | y_max | mean pT (GeV) |
|---|---|---|---|---|---|
| J/ψ | 4.5 | 3.5 | 3.0 | 4.5 | ≈ 3.0 |
| Υ(1S) | 8.0 | 3.5 | 3.0 | 4.5 | ≈ 5.3 |
| Z | 10 | 3.0 | 2.0 | 3.6 | ≈ 7.9 |

**Status.** These are shape assumptions chosen to reproduce the qualitative features of LHC spectra (soft peak, power-law tail, rapidity plateau). They are **not** derived from parton distributions, **not** fitted to the CMS data, and no cross sections are used. Acceptances and kinematic distributions therefore carry a model dependence. The data comparison measures it rather than hiding it (see `validation/CMS_DATA_COMPARISON.md`). A Pythia 8 backend is the documented path to realistic production (`research/LIMITATIONS.md`).

**Validation.** The fraction of events with pT < p₀ matches the analytic CDF 1 − 2^(−2.5), and the |y| plateau fraction matches the trapezoid areas (`test_production_spectra_match_their_definitions`).

## 2. Two-body decay in the rest frame

- E* = m/2 and |p*| = √(m²/4 − m_μ²) for each muon, back to back.
- The μ⁻ direction has polar angle θ about the frame axis, with W(cos θ) ∝ 1 + λ cos²θ, sampled by rejection, and φ uniform.
- If m < 2m_μ, generation raises an error.

| Process | Frame | λ | Justification |
|---|---|---|---|
| J/ψ, Υ | any (isotropic) | 0 | prompt quarkonium polarisation at 7 TeV is compatible with zero (CMS, PLB 727 (2013) 381; PRL 110 (2013) 081802) |
| Z (Drell–Yan) | Collins–Soper | +1 | qq̄ → Z/γ* → ℓℓ via a vector coupling gives 1 + cos²θ at leading order. The forward–backward asymmetry term is neglected. |

**Collins–Soper axes** (Collins & Soper, PRD 16 (1977) 2219):
- Boost both beams (√s = 7 TeV) into the parent rest frame.
- z = unit(p̂₁ − p̂₂), with its sign set to follow the parent's p_z.
- x = −unit(p̂₁ + p̂₂), orthogonalised; y = z × x.

**Independent validation.** The angle is recomputed from the lab muons with the closed form
cos θ_CS = sign(Q_z) · 2(p₁⁺p₂⁻ − p₁⁻p₂⁺) / (Q √(Q² + Q_T²)), with p± = (E ± p_z)/√2.
Generated Z events give ⟨cos²θ⟩ = 0.400 ± 0.003 (expected 2/5) and J/ψ events 1/3 (isotropic).

## 3. Lorentz boost

For velocity β (|β| < 1, enforced):

E' = γ(E + β·p),   p' = p + [γ²/(γ+1) (β·p) + γE] β

γ²/(γ+1) equals (γ − 1)/β² but stays finite at β = 0.

**Validation.**
- Boost-then-inverse returns the input (1e-10).
- The pure-z boost matches the textbook form.
- Property-based tests (Hypothesis, 200 random cases each) check that the invariant mass is preserved and that the inverse boost composes.
- The pair four-momentum equals the parent's (1e-9).

## 4. Invariant mass

m² = m₁² + m₂² + 2(E₁E₂ − |p₁||p₂|) + 4|p₁||p₂| sin²(θ₁₂/2), with

E₁E₂ − |p₁||p₂| = (m₁²|p₂|² + m₂²|p₁|² + m₁²m₂²)/(E₁E₂ + |p₁||p₂|) and 2 sin(θ/2) = |û₁ − û₂|.

This avoids the cancellation in (E₁ + E₂)² − |p₁ + p₂|² for collinear high-energy pairs. It is checked against a 50-digit mpmath reference for two 1 TeV muons 0.1 mrad apart (relative error < 1e-12).

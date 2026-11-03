# Final Independent Scientific and Engineering Audit — Particle Physics Lab

**Audit baseline:** commit `440ef55` ("Add PyVista 3D event display alongside the 2D views"), clean working tree.
**Audit date:** 3 November 2026.
**Scope:** all of the code (physics, detector, reconstruction, statistics, CMS data, GUI, 2D and 3D visualization, configuration, tests) and `docs/`.
**Rules followed:** no production code, parameters, test expectations or Git history were changed. All evidence comes from separate scripts in `audit/`, with their outputs in `audit/results/`. Nothing was tuned or corrected.

---

## 1. Executive summary

The physics chain is **correct where it can be checked against first principles**:
- two-body decay kinematics, Lorentz boosts and the invariant mass;
- the helix model, the charge convention and the 0.299792458 constant;
- the Collins–Soper frame;
- the Highland formula as coded;
- hit bookkeeping;
- the reconstructed mass.

I checked each of these with independent methods: symbolic algebra, explicit Lorentz matrices, 40–50-digit arithmetic, a Runge–Kutta integration of the Lorentz force, and a separately built rest frame. They agree to rounding precision.

The PDG 2026 particle values and the CMS dataset identity (size and adler32 checksum) are confirmed against the primary sources.

A rerun of `run_analysis.py` at the baseline commit reproduces every published number **bit for bit**. All 93 tests pass.

The weak points are in the **detector realism and the statistical interpretation, not in the core algorithms**:

| # | Finding | Severity |
|---|---|---|
| F-DET-6 | The simulated momentum resolution for 1–10 GeV central tracks (0.67–0.71 %) is about **half** the CMS figure (≈ 1.5 %). This is the most likely main reason the simulated peaks are 13–34 % narrower than the data. | HIGH |
| F-DET-3 | The material budget is close to CMS at η = 0 (0.36 vs ≈ 0.4 X₀) but **2–2.5× too low** at \|η\| ≈ 1.4–2 (0.77 vs ≈ 1.9 X₀). | MEDIUM |
| F-DET-4 | Applying the Highland formula layer by layer gives about **10 % less** scattering than applying it to the total thickness. | MEDIUM |
| F-STA-3 | The fitter's fallback minimiser (Nelder–Mead) ignores parameter bounds. On a CMS subsample it stepped to a negative width and **crashed**. The published fits are not affected. | MEDIUM |
| F-DAT-3 | The data mass offsets are **more negative in the forward region** for all three resonances (J/ψ: −0.09 % barrel vs −0.17 % forward, 2.3σ). This points to a momentum-scale effect that depends on η, not only to final-state radiation. The Z data mass moves by up to 46 MeV with the fit window and by 250 MeV with the signal model. Its quoted ±39 MeV is statistical only. | MEDIUM |
| F-STA-5 | The Υ(1S) width ratio (0.66) is **not robust**. A simple sideband-subtracted estimate gives 0.87, and the simulated Υ fit itself has p = 0.004. | MEDIUM |
| F-TOY-1 | The claim that the uncertainties are "about 7 % conservative" is **not supported**. The published pull widths are each within 1.1σ of 1, and my independent toys give 0.98 ± 0.04 and 0.98 ± 0.06. The uncertainties are consistent with correct coverage. | LOW (wording) |
| F-REC-2 | The documented forward curvature offset (D1) **is real but smaller** than recorded: −2.8 × 10⁻⁵ (4.7σ) with smearing only. Its cause is identified: the circle fit ignores the 1 mm radial error of disk hits. It is negligible for the masses. | LOW |
| F-DAT-2 | The inconsistent `pt` column affects 5 794 rows (5.8 %), only global muons, with the published value mostly 1–1.5 % lower. Recomputing pT from the momentum components is justified: the published M agrees with the components in 100 % of those rows. Switching between the two changes the selection by about 12 events and the fitted masses by less than 0.02 MeV. | VERIFIED |

**Maturity:** a scientifically sound, reproducible **educational** simulation with professional software engineering. Its kinematics, tracking geometry and statistics are verified. Its detector response is only qualitatively realistic, and its production model is phenomenological. It is **not** a quantitative model of CMS. The data–simulation comparison supports statements at the level of "the peaks are found at the right masses, the widths are within a factor of about 1.3–1.5, and the shapes differ for documented reasons". It does not support precise resolution or scale claims.

---

## 2. Audit methodology

1. **Read all the code** in `particlelab/`, `run_analysis.py`, the configs, the tests and the docs, and the Git history.
2. **Derived independently, then compared numerically.** No result was accepted from reading the implementation alone.
   - `a1_kinematics.py`: sympy derivation; 4×4 Lorentz matrices; mpmath at 50 digits; my own rest-frame and Collins–Soper construction; Kolmogorov–Smirnov tests of the samplers; numerical integration of the Z line shape.
   - `a2_detector.py` and `a2b_curvature_cause.py`: RK4 Lorentz-force integrator; the curvature constant from the SI definitions of c and e; an empirical check of Highland scattering; truth-scrambling tests of the fit; curvature-offset studies with 200 000–400 000 tracks per configuration.
   - `a3_reco_and_display.py`: mass from 40-digit four-vectors; edge cases; display coordinates against my own helix geometry; off-screen 3D scene tests and a memory test over 300 updates.
   - `a4_fits_data_toys.py`: every data-model / simulation-model combination of the fits; model-independent widths; CMS fits per detector region and muon type; the `pt`-column forensics; toys generated from known truths that are not the fitted model; disjoint simulated subsamples.
   - `a5_architecture.py`: static import graph.
3. **Checked primary sources online:**
   - PDG Live, 2026 edition (retrieved 2026-11-03);
   - CERN Open Data record 545 and its API file listing;
   - the full text of the CMS tracking paper (arXiv:1405.6569 = JINST 9 (2014) P10009);
   - bibliographic records for the other references.
4. **Reproduced the published run:** `run_analysis.py` with `configs/default.json`, output redirected to `audit/results/rerun_default/`, then a field-by-field diff against `outputs/`.
5. **Ran the full test suite** at the baseline.

**Classifications:**
- VERIFIED: independent evidence agrees.
- VERIFIED WITH LIMITATIONS: agrees within a stated scope.
- UNVERIFIED: not established either way.
- INCORRECT: contradicted by the evidence.

**Severities:** CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL.

---

## 3. Repository and architecture findings

**Dependency graph** (from the static analysis, `a5_architecture.json`; an arrow means "imports"):

```
app ──► visualization ──► detector ──► physics ──► domain
 │  ╲                        ▲   ▲        ▲
 │   ╲──► analysis ──────────┘   │        │
 │   ╲──► data ──────────────────┼────────┘   (data → physics.kinematics only)
 │   ╲──► provenance ──► detector, domain
 └──► units ◄── detector, visualization
```

- **No import cycles.** No Qt, pyqtgraph, PyVista or VTK import exists outside `app`. There are no `global` statements.
- **Physical constants are defined once:** `units.C_TRACK` and `units.HIGHLAND_E0`, with particle values in `particles_pdg.json`.
- **The seven concerns are cleanly separated.** Physics, detector simulation, reconstruction, statistics, data ingestion, visualization and orchestration each live in their own package.

| ID | Sev. | Where | Finding | Evidence | Recommendation |
|---|---|---|---|---|---|
| F-ARC-1 | INFO | whole package | Layering is clean, as above. | `a5_architecture.json` | Keep. |
| F-ARC-2 | LOW | `analysis/compare.py` `DATA_FIT`, `SIM_FIT`; `analysis/fitting.py` `WINDOWS` | Fit models, windows, bins and sideband definitions are hard-coded. They are not in `RunConfig` and not in `run_record.json`, so they can be recovered only from the Git commit. | read | Record the fit settings in the run record. |
| F-ARC-3 | LOW | `run_analysis.py:307` | The last log line prints "→". With stdout redirected under Windows cp1252, it raises `UnicodeEncodeError` after all outputs are written, so the process exits with an error. | `audit/results/rerun_default.log` | Use ASCII, or reconfigure stdout to UTF-8. |
| F-ARC-4 | LOW | `fitting.fit_counts` | `converged = bool(posdef and res.success or posdef)` reduces to `posdef`. A failed L-BFGS-B followed by Nelder–Mead is still reported as converged whenever the Hessian is positive definite. | read | Report the optimiser status separately. |
| F-ARC-5 | INFO | `app/desktop.py:53`, `app/view3d.py:165` | The only broad `except Exception` handlers are at GUI boundaries, and both report the error. | `a5_architecture.json` | Acceptable. |
| F-ARC-6 | INFO | `pipeline.simulate` | The main bottleneck is the J/ψ acceptance of 1.2 %: 1 M generated events take 163 s for 12 k selected. The Υ multi-peak fit takes about 20 s. | rerun log | Optional generator-level pre-selection. No rewrite is justified. |
| F-ARC-7 | INFO | `tracking.fit_tracks` | `s_true = h.s[idx]` is assigned but never used. I confirmed that the fit does not depend on it (F-REC-1). | read; `a2` | Remove the variable to avoid confusion. |

No unit inconsistencies were found. Internally everything uses GeV, metres and tesla, with conversion only at the boundaries.

---

## 4. Particle physics verification

### 4.1 Particle properties

**Source:** F. Takahashi et al. (Particle Data Group), *Review of Particle Physics*, Int. J. Mod. Phys. A **41**, 2630011 (2026), doi:10.1142/S0217751X26300115. PDG Live entries S044 (Z), M070 (J/ψ(1S)) and M049 (Υ(1S)), retrieved 2026-11-03.

| | Code (`particles_pdg.json`) | PDG Live 2026 | Status |
|---|---|---|---|
| J/ψ mass | 3.096900 ± 0.000006 GeV | 3096.900 ± 0.006 MeV | VERIFIED |
| J/ψ Γ | 92.6 ± 1.7 keV | 92.6 ± 1.7 keV | VERIFIED |
| J/ψ B(μμ) | 5.961 ± 0.033 % | 5.961 ± 0.033 % | VERIFIED |
| Υ(1S) mass | 9.46040 ± 0.00010 GeV | 9460.40 ± 0.10 MeV | VERIFIED |
| Υ(1S) Γ | 54.02 ± 1.25 keV | 54.02 ± 1.25 keV | VERIFIED |
| Υ(1S) B(μμ) | 2.48 ± 0.04 % | 2.48 ± 0.04 % | VERIFIED |
| Z mass | 91.1879 ± 0.0020 GeV | 91.1879 ± 0.0020 GeV | VERIFIED |
| Z Γ | 2.4955 ± 0.0023 GeV | 2.4955 ± 0.0023 GeV | VERIFIED |
| Z B(μμ) | 3.3662 ± 0.0066 % | 3.3662 ± 0.0066 % | VERIFIED |
| J^PC of J/ψ, Υ | spin 1 | 0⁻(1⁻⁻) for both | VERIFIED |
| Charges | 0 | 0 | VERIFIED |

**Lifetimes** (not stored in the code; derived from τ = ħ/Γ with ħ = 6.582 × 10⁻²⁵ GeV s):

| | τ | cτ |
|---|---|---|
| J/ψ | 7.1 × 10⁻²¹ s | 2 pm |
| Υ(1S) | 1.2 × 10⁻²⁰ s | — |
| Z | 2.6 × 10⁻²⁵ s | — |

Decaying at the production vertex is therefore correct.

- **Not modelled:** non-prompt J/ψ from b-hadron decays, which would be displaced.
- **Natural width vs resolution:** the code keeps them apart. Γ enters only the generated line shape and the width-fixed fit shapes, while the resolution is fitted separately.
- **Muon and other entries:** these are flagged `verified: false`, and the audit did not verify them either (UNVERIFIED, as already documented in D8).
- **Documentation gap:** `docs/research/REFERENCES.md` cites the PDG without the author list or DOI. Add them.

### 4.2 Two-body decay kinematics — VERIFIED

- **Symbolic derivation (sympy):** for M → m₁m₂, with E₁ = (M² + m₁² − m₂²)/2M, E₂ = (M² + m₂² − m₁²)/2M and p = λ^½(M², m₁², m₂²)/2M:
  - E₁ + E₂ = M exactly;
  - both particles are on shell;
  - λ factorises as (M² − (m₁+m₂)²)(M² − (m₁−m₂)²), so p → 0 at threshold;
  - for m₁ = m₂, p reduces to √(M²/4 − m²), which is the form used in `decay_rest`.
- **`decay_rest` against these results** (1 000 decays at each of 2m_μ(1 + 10⁻⁹), 3.0969, 9.4604 and 91.1879 GeV):
  - Σp = 0 and ΣE = M, exactly;
  - |p| agrees to 3 × 10⁻¹⁶ relative;
  - the daughters' masses agree to 5 × 10⁻¹² GeV.
- **Threshold:** a mass below threshold raises an error.
- **Boost:** `boost()` agrees with the explicit Λ(β) matrix to 4.7 × 10⁻¹⁵ (2 000 random β up to 0.999999).
- **Invariant mass against a 50-digit reference:**

  | Case | `inv_mass` max relative error | Naive formula |
  |---|---|---|
  | random pairs | 2 × 10⁻¹⁶ | 5 × 10⁻¹⁴ |
  | collinear pairs at 1 TeV | 4 × 10⁻¹⁶ | 1.7 × 10⁻⁸ |
  | back to back | 2 × 10⁻¹⁶ | 2 × 10⁻¹⁶ |
  | p ≈ 10 MeV | 2 × 10⁻¹⁶ | 2 × 10⁻¹⁶ |

- **After the lab boost:** the pair mass equals the generated mass to 3 × 10⁻¹⁴, and p(μ⁻) + p(μ⁺) = p(parent) to 1.7 × 10⁻¹⁰ GeV.
- **Finite widths:**
  - J/ψ and Υ are sampled from a Cauchy distribution truncated at ±10Γ. That drops 3.2 % of the tails, irrelevant because the resolution is about 300 times Γ.
  - The Z uses the s-dependent relativistic Breit–Wigner (RBW). My numerical integration gives a mean of 91.628 GeV on 60–120 GeV; the sampler gives 91.634 (pull 0.88).
  - The mode of the s-dependent RBW lies 8.5 MeV below M. The fixed-width conversion is Γ²/2M = 34 MeV, consistent with the documented 34 MeV Voigt bias.

### 4.3 Angular distributions — VERIFIED WITH LIMITATIONS

λ_θ is measured by the method of moments with bootstrap errors, on 20 000 decays per sample. The frames are built independently of `frame_axes`:

| Sample | Frame | λ_θ | ⟨cos θ⟩ |
|---|---|---|---|
| Z | Collins–Soper | 0.999 ± 0.046 | −0.0002 ± 0.0045 |
| Z | helicity | 0.906 ± 0.043 | 0.0001 ± 0.0044 |
| J/ψ | CS / HX | 0.004 ± 0.025 / 0.016 ± 0.024 | ≈ 0 |
| Υ(1S) | CS / HX | 0.003 ± 0.025 / 0.013 ± 0.025 | ≈ 0 |

- **The Z frame is correct.** 1 + cos²θ holds in the Collins–Soper frame, with the axis bisecting the beam and minus-target directions and its sign set by the dimuon p_z. The same events give λ = 0.91 in the helicity frame, so the frame dependence is real and handled correctly.
- **The production formula `cos_theta_cs` agrees with my construction** to 6 × 10⁻⁶.
- **F-PHY-1 (LOW).** At the LHC the Z decay distribution is 1 + cos²θ + ½A₀(1 − 3cos²θ) + A₄ cos θ + …, with A₀ growing with q_T and a small forward–backward term A₄. The code sets A₀ = A₄ = 0 and has no γ*/Z interference. This is a documented approximation that affects lepton acceptance only slightly.
- **F-PHY-2 (LOW).** Isotropic quarkonium decays are supported by the CMS polarization measurements, which found no large polarization:
  - J/ψ: PLB 727 (2013) 381, for 14 < pT < 70 GeV and |y| < 1.2;
  - Υ: PRL 110 (2013) 081802, for 10 < pT < 50 GeV and |y| < 1.2.

  The simulation, however, uses isotropy at all pT and y, including the 3–10 GeV region that dominates its J/ψ sample. That is an extrapolation and should be stated as such.

### 4.4 Production spectra — VERIFIED (as defined), phenomenological

- **Samplers:** the pT sampler matches its analytic CDF (KS p = 0.11) and the rapidity trapezoid matches its own (KS p = 0.82).
- **Not derived from QCD:** the spectra come from no QCD or PDF calculation (see §11).

---

## 5. Detector physics verification

### 5.1 Geometry — VERIFIED WITH LIMITATIONS

The comparison is with the CMS tracking paper, JINST 9 (2014) P10009, §2, read in full text:

| Feature | CMS (JINST 9 P10009) | Model | Status |
|---|---|---|---|
| Pixel barrel radii | 4.4, 7.3, 10.2 cm | identical | VERIFIED |
| Pixel disks | z = ±34.5, ±46.5 cm | identical | VERIFIED |
| Pixel hit resolution | ≈ 10 µm (rφ), 20–40 µm (z) | 10 µm, 20 µm | VERIFIED (z at the optimistic end) |
| TIB + TID | r < 55 cm, \|z\| < 118 cm, 4 layers + 3 disks | 4 layers r = 25.5–49.8 cm, \|z\| < 70 cm; 3 disks at 80–100 cm | consistent; individual positions UNVERIFIED |
| TOB | r > 55 cm, \|z\| < 118 cm, 6 layers, outer radius ≈ 110 cm | 6 layers 55.5–108 cm, \|z\| < 118 cm | consistent; intermediate radii UNVERIFIED |
| TEC | 124 < \|z\| < 282 cm, 9 disks | 9 disks 124–272 cm | VERIFIED |
| Strip resolution | TIB 13–38 µm, TOB 18–47 µm | 30 µm, 45 µm | VERIFIED (conservative) |
| Stereo modules | first two TIB/TOB layers; TID rings 1–2; TEC rings 1, 2, 5 | first two TIB/TOB layers; alternate whole TEC disks | simplified |
| Acceptance | \|η\| < 2.5 | ≥ 8 hits: 99.8 % at η = 2.4, 95 % at 2.5, 36 % at 2.7 (20 GeV, σ_z = 6 cm vertex) | VERIFIED |
| Field | 3.8 T | 3.8 T | VERIFIED |
| Beam pipe | — | drawn at r = 3 cm, display only, no material | INFO |

- **Simulation and 3D view use the same geometry.** Every barrel radius and half-length and every disk (z, r_min, r_max), mirrored at ±z, in `detector_geometry()` equals the `DetectorConfig` used by `simulate_hits` (`a3`: `geometry_barrels_match`, `geometry_disks_match` both true).
- **The docstrings correctly say "educational model, NOT the CMS geometry".**

### 5.2 Magnetic field and helices — VERIFIED

- **Constant:** p[GeV/c] = (e B R)·c/(e·10⁹) = 0.299792458 · B[T] · R[m] from the exact SI values of c and e. This equals `units.C_TRACK`.
- **Trajectories:** RK4 integration of dp/ds = κq(p̂ × B), with κ = c·10⁻⁹ and step 0.2 mm, agrees with the barrel crossings of `simulate_hits` to **0.001 µm** or better. Tested at pT = 1.5–45 GeV, η from −1.2 to 0.8, both charges, vertex z = 1 cm.
- **Handedness:** positive charges turn clockwise when viewed from +z (F = qv × B with B along +z). The code and the RK4 integration agree.
- **Charge assignment:** no charge was misassigned in 20 000 tracks with pT 3–50 GeV and |η| < 2.4. None was misassigned at 1 GeV, at η = 2.45, or with 80 % hit efficiency either.
- **Longitudinal motion and forward tracks:** z = z₀ + s·cot θ holds to 3 × 10⁻¹⁴. Disk crossings are guarded against p_z = 0, and loopers stop at s < πR.

**F-REC-2 — the curvature offset (TECHNICAL_DEBT D1) — LOW, cause identified.** The quantity is ⟨pT,true/pT,reco − 1⟩ for 5 GeV muons, using 200 000–400 000 tracks per configuration:

| Configuration | η = 0 | η = 2.3 |
|---|---|---|
| full simulation | (−1.3 ± 1.6) × 10⁻⁵ | (−6.4 ± 3.7) × 10⁻⁵ (1.7σ) |
| smearing only (no scattering) | (+3.3 ± 1.2) × 10⁻⁶ | **(−2.8 ± 0.6) × 10⁻⁵ (4.7σ)** |
| smearing only, disk radial smearing off | — | (−0.4 ± 0.4) × 10⁻⁵ |
| scattering only | (−1.5 ± 1.6) × 10⁻⁵ | (−3.4 ± 3.6) × 10⁻⁵ |
| ideal detector | 10⁻¹⁷ (rounding) | 10⁻¹⁶ (rounding) |

- **What it is:** a genuine, small estimator bias, not a sign or convention error.
- **Cause:** the transverse fit weights disk hits isotropically with 1/σ_rφ². Their 1 mm radial error therefore leaks into the circle residual through the small angle between the track and the radial direction.
- **Earlier record was too large:** the "−1 × 10⁻⁴ (2σ)" in D1 is not reproduced at that size.
- **Impact:** at most 3 × 10⁻⁵ relative, so about 0.1 MeV for the J/ψ and about 3 MeV for a Z whose two muons are both forward. That is two orders of magnitude below the data offsets.
- **Recommendation:** document the cause. Optionally, project the disk radial error into the fit weights.

### 5.3 Multiple scattering — VERIFIED (as coded) with two MEDIUM realism findings

- **Formula:** θ₀ = 13.6 MeV/(βp)·|z|·√(x/X₀)·[1 + 0.038 ln(x/X₀)] (Lynch & Dahl's refit of Highland; this is the PDG form).
  - The units (GeV) and the β factor are correct.
  - The path length grows as 1/cos(incidence), with barrel incidence r̂·t̂ = sin θ cos(Δφ) and disk incidence |cos θ|.
- **Empirical check:** one 0.05 X₀ layer at r = 10 cm and a measuring layer at 1 m, 5 GeV, 200 000 tracks.

  | | Measured | Predicted (θ₀ × lever arm) | Ratio |
  |---|---|---|---|
  | rφ displacement | 483 µm | 490 µm | 0.987 |
  | z displacement | 486.2 µm | 486.4 µm | 1.000 |

  The linearised random walk and its projection onto the surfaces are correct.
- **Range of validity:** the per-layer values of x/X₀ (0.02–0.05, divided by the incidence factor) are within the formula's stated range of 10⁻³ to 100.
- **Not modelled:** non-Gaussian single-scattering tails (Highland describes the central 98 %), energy loss and its fluctuations, and the effect of kicks on later crossings (D3). These are acceptable for this level of model.

**F-DET-4 (MEDIUM) — logarithmic term applied per layer.** Each layer gets its own log term and the results are added in quadrature. For the 13 barrel layers at η = 0 (Σx/X₀ = 0.36) and 10 GeV:

| Method | θ₀ |
|---|---|
| per layer, summed in quadrature (as coded) | 0.705 mrad |
| Highland applied once to the total | 0.784 mrad |
| ratio | **0.90** |

The formula was fitted for a single scatterer of the stated total thickness, so the per-layer treatment underestimates the scattering by about 10 %. Its exact size relative to a full Molière calculation is UNVERIFIED. I did not fetch the PDG's own wording on combining layers.

**F-DET-3 (MEDIUM) — material distribution.** Model material along straight tracks (`a2`) against CMS. The CMS values are read from the material-budget figures cited in arXiv:2012.06888 and arXiv:1502.02701, so they are approximate:

| \|η\| | 0 | 1.0 | 1.4–1.5 | 2.0 | 2.4 |
|---|---|---|---|---|---|
| Model (X₀) | 0.36 | 0.57 | 0.77 | 0.76 | 0.58 |
| CMS (X₀) | ≈ 0.4 | — | ≈ 1.9–2.0 | ≈ 1.5 | ≈ 1.4 (at 2.5) |

The forward material is 2–2.5 times too low, which means the forward scattering is about 1.5 times too small.

**F-DET-6 (HIGH) — resolution too optimistic.**

| | σ(pT)/pT, central tracks, 1–10 GeV |
|---|---|
| CMS (JINST 9 P10009, abstract) | approximately 1.5 % |
| Model at η = 0, pT = 3–10 GeV | 0.67–0.71 % |

- **Caveat on the comparison:** the CMS figure is for all tracks over the central region, so the definitions are not identical.
- **What it means:** the factor of about 2 is still well outside any definitional ambiguity.
- **Contributing causes:**
  - F-DET-4 (−10 %);
  - F-DET-3 (forward only);
  - support structures and electronics not represented as separate scattering surfaces;
  - no alignment uncertainty;
  - no Molière tails.
- **Why it matters:** this is the most likely main cause of the narrow simulated peaks (§10). The current documentation lists causes without numbers.
- **Recommendation:** calibrate the material per layer against the published X₀(η) curve. Do not tune to the dimuon widths (see AUDIT_ACTION_ITEMS A1–A2).

### 5.4 Hit simulation — VERIFIED

- **Missed hits are never used as measurements:**
  - every detected hit is also a crossed hit;
  - the missed fraction is 1.006 % against the configured 1 %;
  - moving every undetected hit 1 km away and scrambling the true arc lengths leaves every fitted momentum and charge **bit-identical** (`a2`, 20 000 tracks).
- **Smearing:** rφ smearing is applied as an angle σ/r. z (barrel) and r (disk) are smeared only where the layer measures them (stereo or pixel). Mono layers are never weighted in the z fit.
- **Truth leakage path (LOW, F-REC-6):** the z-fit fallback for tracks with fewer than 2 longitudinal measurements would include the **unsmeared** z of mono-layer hits. It was triggered **0 times** in 20 000 tracks over |η| < 2.4, so it is a latent path only.
- **Reproducibility:** the same seed gives bit-identical hits.

---

## 6. Track reconstruction verification — VERIFIED

- **Fit method:** a weighted Kåsa start followed by Gauss–Newton geometric circle iterations, plus a weighted s–z line fit.
- **Independence from truth:** the reconstructed momentum depends only on detected hit positions, their resolutions and the detector layout. I tested this by scrambling truth fields and moving non-detected hits (F-REC-1).
- **Ideal detector:** reconstructed mass equals true mass to 7 × 10⁻¹⁴ (J/ψ) and 7 × 10⁻¹² (Z) relative.
- **Edge cases** (20 000 tracks each):

  | Case | Efficiency | Mean pT bias | Robust σ | Charge misID |
  |---|---|---|---|---|
  | 1 GeV, η = 0 | 100 % | −2 × 10⁻⁵ | 0.46 % | 0 |
  | 0.6 GeV looper | 100 % | 3 × 10⁻⁶ | 0.28 % | 0 |
  | 3 GeV, η = 2.35 | 99.9 % | 3 × 10⁻⁴ | 1.7 % | 0 |
  | 20 GeV, η = 2.45 | 99.4 % | 1.0 × 10⁻³ | 2.4 % | 0 |
  | 20 GeV, 80 % hit efficiency | 96.9 % | 1.2 × 10⁻⁴ | 0.80 % | 0 |

  The positive mean pT bias equals the expected second-order σ² effect of measuring 1/pT (documented as D5).
- **Uncertainty:** the reported σ(pT) contains hit errors only, with no scattering covariance (D2, already documented). It is not used in the mass fits.
- **Forward tracks:** see F-REC-2.

### 6.1 Invariant mass — VERIFIED

- **Against 40-digit four-vectors:** pipeline `reco_mass` matches independent 40-digit four-vectors built from the reconstructed momenta to 4 × 10⁻¹⁶ (300 events each for J/ψ and Z).
- **Charge labelling:** the reconstructed μ⁻ points along the true μ⁻ in 100 % of selected events.
- **Mean shift:** reconstructed minus true is +4.8 MeV for the Z (20 000-event sample), consistent with the documented (3/8)Σσ² shift.

---

## 7. Statistical analysis verification

- **Likelihood:** the extended binned Poisson likelihood −ln L = Σ(μᵢ − nᵢ ln μᵢ) is correctly constructed.
- **Normalisation:** shapes are normalised numerically over the window, and yields are counts within the window.
- **Bin integration:** 5 points per bin.
- **Goodness of fit:** the Baker–Cousins deviance against χ²(ndf) is correct.

**Effect of the fit model on the reported numbers** (`a4`; the published choice comes first in each cell):

| | Data mass (GeV) | Sim mass (GeV) | FWHM ratio sim/data, all model combinations | Model-independent FWHM ratio |
|---|---|---|---|---|
| J/ψ | 3.09331 (CB+G) / 3.09310 (2G) | 3.09734 (2G) / 3.09734 (CB+G) | 0.738–0.741 | 0.78 |
| Υ(1S) | 9.44390 (2G) / 9.44483 (CB+G, n at bound) | 9.46150 / 9.46150 | 0.662–0.665 | **0.87** |
| Z | 90.948 (BW⊗CB, p = 0.36) / 90.697 (BW⊗G, p = 6 × 10⁻⁹) | 91.1896 / 91.1903 | 0.754–0.794 | 0.78 |

How the model-independent ratio was computed:
- **Simulation:** FWHM of a lightly smoothed fine histogram.
- **Data:** the same, after subtracting an exponential fitted only to the sidebands.

| ID | Sev. | Finding | Evidence | Recommendation |
|---|---|---|---|---|
| F-STA-1 | VERIFIED WITH LIMITATIONS | Using different signal models for data and simulation does **not** distort the J/ψ and Z width ratios. All four model combinations agree to ±0.003 (J/ψ) and ±0.02 (Z). FWHM is a definition common to both. | table | Keep FWHM as the comparison quantity. Quote the model spread. |
| F-STA-2 | LOW | The `sigma` / "resolution param" column changes meaning between shapes. It is the Gaussian σ for `rbw_*`, the CB core σ for data Z, and FWHM/2.355 for J/ψ and Υ. The docs compare FWHM, which is correct, but the per-fit σ values must not be compared across models. | `FitResult._width` | Label each width with its definition, or drop σ from comparison tables. |
| F-STA-3 | MEDIUM | The Nelder–Mead fallback in `fit_counts` is **unbounded**. On the 1 305-event barrel–barrel Z subsample it stepped to σ < 0 and α < 0: `np.log` raised a RuntimeWarning, then `np.convolve` raised "a cannot be empty". A fallback that wanders outside the bounds can also return clipped parameters whose likelihood and Hessian do not belong to a minimum. The published fits are unaffected (all converged, none at a bound). | reproduced in the audit (spy on `optimize.minimize`) | Use a bounded fallback, or a transformed parameterisation. Return a failed fit instead of raising. |
| F-STA-4 | LOW | Data J/ψ CB+Gaussian correlations reach 0.96 (σ–frac) and −0.92 (α–n). The mass is not strongly correlated, but the Hessian errors on the CB shape parameters and the propagated width error (D9) are unreliable. | `a4` correlations | Report profile intervals or toy-based errors for widths. |
| F-STA-5 | MEDIUM | The Υ width comparison is **not robust**. The fitted data FWHM is 214 MeV, but a sideband-subtracted estimate gives 161 MeV, so the ratio is 0.66 vs 0.87. The difference comes from the background shape under the 1S peak and the 2S/3S overlap. The simulated Υ double-Gaussian fit has **p = 0.004** (60 k events), so it does not describe the simulated shape either. | `a4` | Report the Υ ratio with a background-model systematic. Use a better shape, or none, for the simulation. |
| F-STA-6 | LOW | In `compare()`, the simulated signal region is centred on the **data** mass, which is 240 MeV low for the Z. Sideband subtraction ignores signal leaking into the sidebands, and the Z FSR tail lies on the low side. | read | Centre on the simulation's own mass. Account for signal in the sidebands. |
| F-STA-7 | LOW | Subsample fits of the CMS data often report `converged = False` (J/ψ categories). Υ "mixed barrel/forward" gives an implausibly small error of 0.33 MeV. | `a4` | Treat them as unreliable (they are used here only as indications). |

---

## 8. Toy Monte Carlo verification

**Published toys** (`outputs/fit_validation.json`): these are generated from the **fitted model itself** and refitted, so they test self-consistency, not shape mismatch.

| | Pull mean | Pull width | Standard error on width (n = 100) | Width deviation from 1 |
|---|---|---|---|---|
| J/ψ | −0.04 ± 0.09 | 0.946 | 0.067 | 0.8σ |
| Υ(1S) | −0.04 ± 0.09 | 0.934 | 0.066 | 1.0σ |
| Z | +0.14 ± 0.09 | 0.928 | 0.067 | 1.1σ |

**F-TOY-1 (LOW, documentation).** "The Hessian errors are correct to within about 7 % (slightly conservative)" in CMS_DATA_COMPARISON.md overstates what 100 toys can show. Each width is consistent with 1, and the three together sit 1.7σ below 1 (0.936 ± 0.038).

**Independent toys** (unbinned samples from known truths that differ from the fitted model; pulls taken against the true mass):

| Truth | Fit | Toys | Pull mean | Pull width | Failures |
|---|---|---|---|---|---|
| 70/30 double Gaussian, σ = 18/35 MeV, 12 184 events | `double_gauss` | 300 | −0.035 ± 0.057 | **0.981 ± 0.040** | 0 |
| s-dependent RBW ⊗ Gaussian (σ = 1.2 GeV), 56 948 events in 70–110 GeV | `rbw_gauss` | 149 | +0.019 ± 0.080 | **0.980 ± 0.057** | 1 |

**Disjoint subsamples of the real simulated shape** (spread of fitted masses divided by the mean quoted error):

| | Subsamples × events | Ratio |
|---|---|---|
| Υ | 30 × 2 022 | 1.03 ± 0.13 |
| Z | 19 × 2 848 | 0.92 ± 0.15 |
| J/ψ | 6 × 2 031 | 1.77 ± 0.56 (inconclusive) |

Mass bias against the full-sample fit: −0.02 ± 0.27 MeV (Υ) and −1.0 ± 8.6 MeV (Z).

**Conclusion — VERIFIED WITH LIMITATIONS.**
- The mass estimators are unbiased.
- The Hessian mass errors have correct coverage to about 5 %, both for model-consistent toys and for the real simulated Υ and Z shapes.
- Coverage for the J/ψ simulated shape is not established, because 12 k selected events allow too few subsamples.
- Width coverage was not tested.
- The failure rate is 0–2 %.

---

## 9. CMS dataset verification

| Check | Result | Status |
|---|---|---|
| Identity | CERN Open Data record 545, "Datasets derived from the Run2011A SingleElectron, SingleMu, DoubleElectron, and DoubleMu primary datasets", T. McCauley, 2017, CMS, CC0-1.0 | VERIFIED (record page) |
| File and checksum | `Dimuon_DoubleMu.csv`, 13 935 840 bytes, adler32 `5b87a72a` | VERIFIED (record API file listing and local recomputation) |
| Producer selection | two muons with \|η\| < 2.4, at least one global, 0.3 < M < 300 GeV, opposite sign | VERIFIED (record "methodology") |
| Producer caveat | the record states the files "are not suitable for a full physics analysis" | not quoted in the project docs; add it |
| Schema and types | 21 columns as expected; Run, Event and Q are int64, kinematics float64, type is str | VERIFIED |
| Counts | 100 000 rows, runs 160957–173692, no duplicate (Run, Event) pairs, 0 rejected | VERIFIED |
| Selection | both muons pT > 3 GeV and \|η\| < 2.4, recomputed: 93 469 events; identical cuts in simulation | VERIFIED |
| Trigger | the DoubleMu trigger paths and thresholds are not documented in the CSV and are not simulated | limitation (§11) |

### 9.1 The pT inconsistency (F-DAT-2) — investigated

**Scale and pattern:**
- **Affected rows:** muon 1 in 2 886 rows (2.9 %), muon 2 in 3 218 (3.2 %); **5 794 rows (5.8 %)** have at least one affected muon. The tolerance is 0.1 % + 2 MeV.
- **Size:** the published pt is **lower** than √(px² + py²) in 86–88 % of affected muons. The median relative difference is −1.5 % (muon 1) and −0.9 % (muon 2). The 1st–99th percentiles span −32 % to +49 %.
- **Which muons:** only **global** muons are affected. None of the tracker-only muons is (muon 2: 0 of the T type; muon 1 had no T-type rows). The rate is 3.5 % at |η| < 1.6 and 0.7–1 % at |η| > 1.6.
- **Other columns:** the published φ still agrees in 90–94 % of affected rows. Only 3 and 11 rows match the other muon's pT, so the columns are not swapped.

**Which columns are trustworthy:**
- **Components agree with M:** in **100 %** of affected rows, the published M agrees with M recomputed from (px, py, pz).
- **pt/eta/phi do not:** only 8 % agree with M recomputed from (pt, eta, phi).
- **Conclusion:** the components and M are mutually consistent, while the published pt (and, rarely, eta) comes from a different estimate. A tracker-only versus global-fit momentum is a plausible hypothesis but is UNVERIFIED; the record does not say which momentum `pt` uses.
- **E column:** I did not use it to check the components. At the 10⁻⁴ GeV rounding, E² − p² cannot resolve the muon mass for momenta above a few GeV.

**Downstream effect** (selecting with the published pt instead of the recomputed one):
- 12 events are added and 11 lost.
- The fitted masses change by 0.000 MeV (J/ψ), −0.013 MeV (Υ) and 0.000 MeV (Z), against statistical errors of 0.48, 3.1 and 39 MeV.

Recomputing pT is **justified, and no conclusion depends on it**. No affected event is discarded: the warning is counted and the row is kept.

---

## 10. Experimental comparison

**Reproduction:** the bit-for-bit rerun reproduces every value in the comparison table:
- the masses (J/ψ 3.0933 ± 0.0005, Υ 9.4439 ± 0.0031, Z 90.948 ± 0.039; simulation 3.0973 ± 0.0002, 9.4615 ± 0.0003, 91.190 ± 0.009);
- the FWHM ratios 0.74, 0.66 and 0.79.

**Data mass offsets from PDG, by category** (each fitted with the published model; † = Hessian not positive definite):

| | Both \|η\| < 0.9 | Mixed | Both \|η\| > 0.9 | Both global |
|---|---|---|---|---|
| J/ψ | −0.089 % ± 0.017 † | −0.110 % ± 0.035 † | −0.169 % ± 0.030 † | −0.122 % ± 0.018 |
| Υ(1S) | −0.162 % ± 0.035 | −0.031 % (unreliable) † | −0.300 % ± 0.075 | −0.178 % ± 0.033 |
| Z | fit crashed (F-STA-3) | −0.087 % ± 0.099 | −0.390 % ± 0.106 | −0.230 % ± 0.042 |

**Z data mass across fit windows and models:**

| | 70–110 GeV | 75–105 GeV | 80–100 GeV |
|---|---|---|---|
| BW⊗CB | 90.948 | 90.927 | 90.902 |
| BW⊗G | 90.697 (p = 6 × 10⁻⁹) | 90.722 | 90.768 |

Measured facts:
1. The offsets are negative for all three resonances.
2. The offsets are **larger in the forward region** for all three. The J/ψ barrel–forward difference is 2.5 ± 1.1 MeV (2.3σ); Υ and Z show the same sign at 1.7–2.1σ.
3. The relative offsets grow with mass: −0.12 %, −0.18 % and −0.23 % for the both-global samples.
4. The Z mass depends strongly on how the low-mass tail is modelled. It moves 46 MeV across windows with BW⊗CB, and BW⊗G (a poor fit) gives 250 MeV lower.

Explanations, kept separate from the facts above:
- **Momentum scale (supported):** an η-dependent momentum-scale bias in this uncalibrated derived dataset fits fact 2. Final-state radiation in the dimuon rest frame does not depend on the muons' lab η, so it cannot explain the η dependence. CMS applies muon momentum corrections, and whether they are present in this CSV is UNVERIFIED.
- **Final-state radiation (part of the Z offset only):** facts 3 and 4 show that the peak position depends on the tail model, which is where FSR enters. The relative effect of FSR is expected to grow with mass, but this audit did not simulate FSR, so its size is UNVERIFIED.
- **Line-shape physics (estimate only):** the model has no parton luminosity and no γ*/Z interference. A falling luminosity ∝ m⁻ᵏ shifts the Z peak by about −kΓ²/8M ≈ −26 to −43 MeV for k = 3–5. This is my analytic estimate, not a measurement.
- **Selection and production (not quantified):** these change the mix of η regions and therefore the average scale, but were not measured here.

**F-DAT-3 (MEDIUM).** The docs' attribution ("likely final-state radiation … and the uncalibrated momentum scale") is plausible but incomplete. The data show an η dependence, and the quoted data errors leave out fit-model and window systematics (Z: at least 46 MeV, compared with the ±39 MeV statistical error). Recommendation: quote a systematic error from the window and model variations and report the masses in η categories. Do not "correct" the data.

**Width comparison.**
- **J/ψ and Z:** the simulated peaks are 21–26 % narrower than the data, and this holds for both fitted and model-independent widths.
- **Υ:** 13–34 % narrower, depending on the background treatment (F-STA-5).
- **Likely cause:** quantitatively, F-DET-6 (simulated resolution about 2 times better than CMS at low pT) is the strongest candidate.

---

## 11. Production-model discrepancies

Muons in a mass window around each peak. The data here are not background-subtracted; the sideband-subtracted χ² values in `outputs/results.json` are reproduced exactly.

| | Mean pT data / sim (GeV) | Median pT data / sim | Fraction \|η\| > 1.2 data / sim | KS D (pT) | KS D (η) | Published χ²/ndf pT; η |
|---|---|---|---|---|---|---|
| J/ψ (3.0–3.2) | 9.75 / 4.87 | 8.24 / 4.15 | 0.29 / 0.45 | 0.62 | 0.09 | 12 396/26; 1 539/23 |
| Υ(1S) (9.3–9.6) | 6.01 / 5.21 | 4.79 / 4.77 | 0.34 / 0.41 | 0.23 | 0.05 | 782/25; 160/23 |
| Z (81–101) | 39.9 / 38.0 | 40.0 / 40.0 | 0.38 / 0.44 | 0.09 | 0.04 | 682/26; 392/23 |

- **J/ψ (HIGH within its scope, documented as D4):** the simulated muon pT is half that in data. The two causes are:
  - the trigger, whose DoubleMu thresholds favour harder muons (not simulated);
  - the phenomenological pT spectrum (p₀ = 4.5 GeV, n = 3.5), which is softer than prompt plus non-prompt J/ψ production at 7 TeV. That spectrum is not derived from QCD or PDFs and not fitted to data.

  The simulation's 1.2 % J/ψ acceptance therefore has no physical meaning (the docs say so).
- **Muon η:** all three samples have 6–16 points fewer forward muons in data than in simulation. That fits muon-system acceptance and trigger effects, which are not simulated, and the simulated rapidity plateau (flat to |y| = 3 for quarkonia, 2 for the Z).
- **Z:** the core of the Jacobian peak matches (median 40.0 GeV in both). The data have a harder tail, consistent with the missing Z q_T from QCD radiation.
- **What is known:** the trigger, the PDFs and real reconstruction efficiency are absent by design. These are known model limitations.
- **What is not known:** the sizes of the individual effects cannot be separated without a trigger emulation and a generator such as Pythia. That is correctly not attempted by tuning.

---

## 12. 2D visualization verification — VERIFIED

**Projections** (`app/desktop.py` `EventTab.show_event`):
- **XY view:** (x, y).
- **ZR view:** (z, s·√(x² + y²)), with one sign s per track taken from the y of the true path's end point. That |r| is preserved exactly (maximum error 0).
- **Detector outlines:** barrel circles in XY, and segments at ±half-length and disks at ±z in ZR, all from `detector_geometry`.
- **Beam axis:** shown implicitly as the ZR horizontal axis. The beam pipe is not drawn in 2D.
- **Event data:** the 2D tab draws the same `DisplayEvent` object as the 3D tab, so the two views cannot disagree.

**F-VIS-4 (INFO):** the sign of the ZR side is chosen from the **true** trajectory. This uses truth for display layout only and changes no coordinate.

**Gap:** the 2D widget has no automated test (D6; pytest-qt is not installed).

---

## 13. 3D visualization verification — VERIFIED

**Coordinates** (`a3`, 15 events over all three processes, 30 tracks):
- True and reconstructed paths lie on circles of radius pT/(0.2998B) through the origin, with the centre on the correct side for the charge. Maximum relative error is 4 × 10⁻¹⁶.
- dz/ds equals pz/pT to 3 × 10⁻¹⁴.
- The true path starts exactly at the generated vertex.
- `hits_ideal` lie on the true helix to 4 × 10⁻¹⁶.
- The displayed reconstructed mass equals the independent four-vector mass to 2 × 10⁻¹⁶.
- The same seed gives the same display event.

**True vs reconstructed tracks:**
- The true path is built from the generator's momentum, charge and vertex. The reconstructed path is built from `TrackFit.p`, `.q` and `.z0`.
- They differ in 30 of 30 tracks. With an ideal detector they coincide (existing test).
- Charge colours are consistent with the labels.

**F-VIS-3 (LOW):** the reconstructed helix is drawn from (0, 0, z₀,fit), ignoring the fitted d₀, and over the **true** track's arc length. Both are display conventions that reuse no truth coordinates, but they should be stated in the user guide.

**Behaviour** (off-screen PyVista plus existing tests):
- The three camera presets are distinct, and the beam view looks along z.
- Zoom changes the view and azimuth rotation moves the camera.
- Geometry is cached: the same key reuses the same actors.
- Visibility settings persist across events.
- **Residual magnification is display-only.** The stored `hits` and `hits_ideal` are unchanged after magnification ×1000 and after set_event at ×1000 and ×1. At ×1, the drawn hits equal the simulated arrays.

**Performance and stability:**
- **300 consecutive updates:** the renderer holds 21 actors (19 event and detector actors plus text and axes). Nothing accumulates.
- **Memory:** after warm-up, resident memory grows by 0.14 MB per 100 updates, so there is no measurable leak.
- **Update time:** median 102 ms and 95th percentile 132 ms off-screen. The docs' "60–80 ms" was measured interactively and is hardware- and load-dependent (LOW, F-VIS-5).
- **Threading:** the PyVista import runs off the GUI thread (existing subprocess test). VTK objects are created on the GUI thread.

| ID | Sev. | Finding | Recommendation |
|---|---|---|---|
| F-VIS-6 | LOW | `Event3DTab.closeEvent` is never called for a widget inside a tab, because Qt only delivers close events to top-level windows. The plotter is never closed explicitly when the application exits, so VTK cleanup is left to garbage collection. | Close the plotter from `MainWindow.closeEvent`. |
| F-VIS-7 | LOW | Importing VTK in a worker thread works on the reference machine and in tests, but VTK does not document it as supported. | Keep the fallback message. Mention it in DESIGN_DECISIONS. |
| F-VIS-5 | LOW | The documented update time is optimistic. | Quote a range. |

---

## 14. Reproducibility assessment — VERIFIED WITH LIMITATIONS

- **Bit-for-bit rerun:** `run_analysis.py` at `440ef55` with `configs/default.json` reproduces `results.json`, `fit_validation.json`, `data_validation.json`, `data_cutflow.json` and `detector_studies.json` exactly.
- **Fixed seeds:**
  - hit simulation, display events and the pipeline are bit-reproducible;
  - chunk independence and serial/parallel equality are tested;
  - each chunk uses child i of `SeedSequence(seed)`.
- **What the run record captures:** seed, configuration, detector settings, package versions (numpy, scipy, pandas, matplotlib, PySide6, pyqtgraph), PDG edition and retrieval date, dataset record, file check, selection and cutflow.

| ID | Sev. | Finding | Recommendation |
|---|---|---|---|
| F-REP-1 | MEDIUM | The published `outputs/run_record.json` names an earlier development commit with **`dirty: true`**, so on its own it does not identify the code that produced the results. Mitigated: the audit showed the clean baseline reproduces them exactly. | Regenerate the outputs from a clean commit. Refuse to write a run record from a dirty tree, or store the diff. |
| F-REP-2 | LOW | The run record leaves out the analysis settings (F-ARC-2), a hash of `particles_pdg.json`, and the versions of PyVista, VTK, sympy and mpmath. None of these affects the numbers except the analysis settings. | Add them. |

---

## 15. Test-suite assessment

**Result:** **93 passed** in 308 s (`audit/results/pytest_full.log`). `RuntimeWarning` is configured as an error.

**Tests against independent scientific expectations** (strong):
- the invariant mass against mpmath, and Lorentz invariance (Hypothesis);
- the boost against a textbook z-boost, and boost∘inverse;
- two-body rest-frame back-to-back and threshold behaviour;
- the 1 + λcos²θ samplers for several λ;
- Collins–Soper recovered with a different formula;
- the Gluckstern resolution for the uniform barrel (ratio within 10 %);
- exact reconstruction with an ideal detector;
- Crystal Ball normalisation and continuity, and the shift-free convolution;
- the Voigt bias for the Z;
- data integrity against the published checksum and the validation counts;
- hit efficiency;
- multiple scattering ∝ 1/p;
- ⟨pT⟩ bias = var(1/pT).

**Self-consistency or regression tests** (useful against regressions, but not scientific proof):
- `test_fixed_seed_results_unchanged`: compares against its own baseline.
- `test_toy_pulls_have_unit_width`: toys from the fitted model.
- `test_pdg_2026_values`: the expected numbers are a second copy of the JSON. The audit confirmed them against PDG Live independently.
- The visualization coordinate tests: they replay the same random stream through the same functions, so they prove the plumbing, not the physics. The audit's independent helix check (§13) fills that gap.

**Weak tolerances:**
- the scattering floor against the analytic estimate is accepted within a factor of 2;
- the fit-recovery test allows 4σ plus 2 × 10⁻⁴·M, which is 18 MeV for the Z.

**Missing tests** (none would weaken an existing one):
- the Highland **amplitude**, not only its 1/p scaling;
- material against a reference X₀(η);
- the bounded fit fallback (F-STA-3 would have been caught);
- a smoke test of `run_analysis.py` with redirected stdout (F-ARC-3);
- 2D GUI widget tests;
- the selection on data and simulation being identical;
- the direction of the forward curvature bias (F-REC-2).

No test was weakened or modified during the audit.

---

## 16. Scientific limitations (consolidated)

1. **Detector:** educational geometry, about 2 times better low-pT resolution than CMS, too little forward material, per-layer Highland, Gaussian scattering only, no energy loss, ideal alignment, no trigger, no muon system.
2. **Production:** phenomenological pT and y spectra. No non-prompt J/ψ, no PDFs, no Z q_T from QCD radiation.
3. **Decay physics:** no FSR. No A₀/A₄ terms or γ*/Z interference for the Z. Quarkonium isotropy extrapolated below the measured pT range.
4. **Statistics:** data masses carry statistical errors only. The Υ width depends on the background model. Width errors for CB-type shapes are not validated.
5. **Data:** an outreach-level derived dataset, "not suitable for a full physics analysis" in the producer's words. Unknown momentum calibration and unknown trigger.

## 17. Outstanding technical debt

**Re-assessment of TECHNICAL_DEBT.md:**

| Item | Audit result |
|---|---|
| D1 | Re-measured and its cause identified (F-REC-2). Update the entry. |
| D2, D3, D4, D5, D6, D10 | Confirmed as stated. |
| D7 | Measured at about 20 s. |
| D8 | Confirmed: only the three resonances were checked. |
| D9 | Confirmed: correlations up to 0.96 (F-STA-4). |

**New items from this audit:**
- F-DET-3, F-DET-4, F-DET-6 (detector realism);
- F-STA-3 (unbounded fallback);
- F-STA-5 (Υ width robustness);
- F-DAT-3 (systematics and η dependence);
- F-REP-1 (dirty-tree run record);
- F-ARC-3 (Unicode crash);
- F-ARC-4 (converged flag);
- F-VIS-6 (plotter close);
- F-REC-6 (latent z-fit truth path);
- F-TOY-1 (wording).

## 18. Recommended corrections

Corrections are listed in priority order, with dependencies and effort, in **`AUDIT_ACTION_ITEMS.md`**. None has been implemented.

## 19. Final maturity assessment

| Area | Classification | Maturity |
|---|---|---|
| Particle data (J/ψ, Υ, Z) | VERIFIED | Publication-ready source and values. |
| Decay kinematics, boosts, invariant mass | VERIFIED | Correct to machine precision, with numerically stable formulas. |
| Angular distributions | VERIFIED WITH LIMITATIONS | Correct frames; simplified Z angular coefficients. |
| Production model | VERIFIED (as defined), not physical | Phenomenological. Acceptances meaningless. |
| Helix propagation, field, charge | VERIFIED | Matches a Lorentz-force integration to nanometres. |
| Detector geometry | VERIFIED WITH LIMITATIONS | Pixel and outer boundaries match CMS; inner strip positions approximate. |
| Material and multiple scattering | VERIFIED as coded; realism INCORRECT in magnitude | Low-pT resolution about 2 times too good. Forward material too low. |
| Track reconstruction | VERIFIED | Independent of truth; small, understood forward bias. |
| Likelihood fitting | VERIFIED WITH LIMITATIONS | Correct likelihood and coverage for masses. Unbounded fallback. Width errors unvalidated. |
| Toy validation | VERIFIED WITH LIMITATIONS | Mass coverage correct to about 5 %. The "7 % conservative" claim is unsupported. |
| CMS data pipeline | VERIFIED | Identity, integrity and schema confirmed. pT issue understood and harmless. |
| Data–simulation comparison | VERIFIED WITH LIMITATIONS | Reproducible. Mass offsets η-dependent and their explanation incomplete. Υ width ratio not robust. Kinematic shapes not modelled. |
| 2D and 3D visualization | VERIFIED | Coordinates exact; no leaks; display-only magnification confirmed. |
| Reproducibility | VERIFIED WITH LIMITATIONS | Bit-exact rerun; published run record from a dirty tree. |
| Software engineering and tests | Good | Clean architecture, 93 passing tests, a few self-consistency tests and gaps. |

**Overall:** a rigorous, reproducible and honestly documented **educational** particle-physics laboratory. Its algorithms are correct and its main shortfall is detector realism. No CRITICAL defect was found. The one HIGH finding (resolution realism) and the MEDIUM findings should be addressed, or clearly stated, before any quantitative claim about resolution, scale or acceptance is presented as a result.

---

## Appendix A — References checked

| # | Reference | Status | How checked |
|---|---|---|---|
| 1 | F. Takahashi et al. (PDG), *Review of Particle Physics*, Int. J. Mod. Phys. A 41, 2630011 (2026), doi:10.1142/S0217751X26300115 | VERIFIED | PDG Live S044, M070, M049, 2026-11-03 |
| 2 | T. McCauley (CMS), CERN Open Data record 545 (2017), CC0 | VERIFIED | record page and API |
| 3 | CMS Collaboration, "The CMS experiment at the CERN LHC", JINST 3 (2008) S08004 | VERIFIED | reference [5] of JINST 9 P10009, doi:10.1088/1748-0221/3/08/S08004 |
| 4 | CMS Collaboration, "Description and performance of track and primary-vertex reconstruction with the CMS tracker", JINST 9 (2014) P10009 | VERIFIED | arXiv:1405.6569 full text and journal reference, doi:10.1088/1748-0221/9/10/P10009 |
| 5 | CMS, "Measurement of the prompt J/ψ and ψ(2S) polarizations in pp collisions at √s = 7 TeV", PLB 727 (2013) 381 | VERIFIED WITH LIMITATIONS | doi:10.1016/j.physletb.2013.10.055, arXiv:1307.6070. Measured for pT > 14 GeV only (F-PHY-2). |
| 6 | CMS, "Measurement of the Υ(1S), Υ(2S), and Υ(3S) polarizations in pp collisions at √s = 7 TeV", PRL 110 (2013) 081802 | VERIFIED WITH LIMITATIONS | doi:10.1103/PhysRevLett.110.081802. Measured for 10 < pT < 50 GeV. |
| 7 | J. C. Collins, D. E. Soper, PRD 16 (1977) 2219 | VERIFIED | doi:10.1103/PhysRevD.16.2219 |
| 8 | V. L. Highland, NIM 129 (1975) 497; G. R. Lynch, O. I. Dahl, NIM B 58 (1991) 6 | VERIFIED | bibliographic records. The 13.6 MeV / 0.038 constants are Lynch–Dahl's, as coded. |
| 9 | PDG review "Passage of particles through matter" | VERIFIED WITH LIMITATIONS | part of reference 1; its exact wording on combining layers not fetched (F-DET-4) |
| 10 | R. L. Gluckstern, NIM 24 (1963) 381 | VERIFIED | INSPIRE record |
| 11 | I. Kåsa, IEEE Trans. Instrum. Meas. IM-25 (1976) 8 | VERIFIED | IEEE record, doi:10.1109/TIM.1976.6312298. Some sources give the title as "A curve fitting procedure…". |
| 12 | N. Chernov, *Circular and Linear Regression*, CRC Press (2010) | UNVERIFIED | not checked |
| 13 | R. Frühwirth, NIM A 262 (1987) 444 | VERIFIED | reference [32] of JINST 9 P10009 |
| 14 | M. J. Oreglia, SLAC-R-236 (1980); J. E. Gaiser, SLAC-R-255 (1982) | Oreglia VERIFIED WITH LIMITATIONS; Gaiser UNVERIFIED | OSTI lists SLAC-236, "Study of the reactions ψ′ → γγψ"; author and year not confirmed in the record seen |
| 15 | S. Baker, R. D. Cousins, NIM 221 (1984) 437 | VERIFIED | doi:10.1016/0167-5087(84)90016-4 |
| 16 | T. Sjöstrand et al., CPC 191 (2015) 159 | VERIFIED | doi:10.1016/j.cpc.2015.01.024 |
| 17 | S. Agostinelli et al. (Geant4), NIM A 506 (2003) 250 | VERIFIED | reference [6] of JINST 9 P10009 |
| — | CMS material budget values used in §5.3 | VERIFIED WITH LIMITATIONS | read from figures quoted in arXiv:2012.06888 and arXiv:1502.02701 (approximate) |

## Appendix B — Evidence files

All are in `audit/results/`:
- `a1_kinematics.json`
- `a2_detector.json`
- `a2b_curvature_cause.json`
- `a3_reco_and_display.json`
- `a4_fits_data_toys.json` (with `a4.log`)
- `a5_architecture.json`
- `rerun_default/` and `rerun_default.log`
- `pytest_full.log`

The scripts are described in `audit/README.md`.

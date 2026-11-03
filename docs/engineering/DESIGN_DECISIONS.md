# Design Decisions (ADR log)

Each entry records the context, the decision and the consequence.

**ADR-1 · Restructure incrementally, do not rewrite.** The original baseline was correctly layered. Modules were moved into `domain/physics/detector/analysis/data/app` subpackages with `git mv`, so history is kept. All 28 original checks were ported to the new API (marked "original test N" in the tests).

**ADR-2 · Particle data in one JSON file.** Values (PDG 2026, retrieved 2026-11-02) carry uncertainties, the source and a `verified` flag. Only J/ψ, Υ(1S) and Z were checked against PDG Live; the other entries are flagged unverified. One change from the baseline: M_Z = 91.1879 GeV (the baseline had 91.1876).

**ADR-3 · Z line shape uses an s-dependent width.** PDG quotes M_Z for a Breit–Wigner with s-dependent width. A fixed-width (Voigt) fit shifts the mass by about Γ²/2M ≈ 34 MeV; this is demonstrated in `test_fixed_width_voigt_is_biased_for_the_z`. Generation and fitting both use the relativistic, s-dependent shape. Parton-luminosity effects are not modelled.

**ADR-4 · Angular distributions per process.** J/ψ and Υ decay isotropically (λθ = 0), following the CMS 7 TeV polarisation measurements. Z decays follow 1 + cos²θ in the Collins–Soper frame. The baseline's universal 1 + cos²θ along the beam axis was removed. The generated distribution is checked with the independent Collins–Soper formula in `kinematics.cos_theta_cs`.

**ADR-5 · Two detector models.**
- `cms_inspired()` adds finite barrels, endcap disks, per-layer material, mono/stereo strips, hit efficiency, a luminous region and a beam-spot constraint.
- `uniform_barrel()` keeps the original reference model, whose resolution has a closed form (Gluckstern). It validates the fitter.

The CMS-inspired model is labelled educational everywhere and is not the CMS geometry.

**ADR-6 · Linearised multiple scattering.** Highland angles at each crossed surface are summed as displacements θᵢ(lⱼ − lᵢ). This captures the dominant low-momentum resolution term at about 2× the cost of no material. A full step-wise helix propagation (or Geant4) would also change the direction used for later crossings; the effect is second order and is listed in TECHNICAL_DEBT.

**ADR-7 · No mean energy loss.** A minimum-ionising muon loses 1.66 MeV cm²/g × 2.33 g/cm³ × 0.094 cm ≈ 0.36 MeV per 1 % X₀ of silicon-like material: 0.7–1.8 MeV per layer, and about 15–65 MeV in total for 0.4–1.8 X₀. Real track fits correct the mean loss, so simulating loss without simulating the correction would create an artificial bias. The fluctuations are much smaller than the multiple-scattering term for pT > 3 GeV. This is documented as a limitation.

**ADR-8 · Extended binned Poisson likelihood replaces least squares.** Neyman χ² (σ = √n) is biased in low-count bins. The likelihood is minimised with L-BFGS-B (Nelder–Mead fallback); errors come from the numerical Hessian. Toy studies confirm unit pulls for the reference model.

**ADR-9 · Data–simulation fit models.**
- J/ψ: Crystal Ball + Gaussian.
- Υ: double Gaussian, with Υ(2S) and Υ(3S) fitted together in data over 8.5–11.5 GeV. Without them the 9.0–9.8 GeV fit fails (χ²/ndf ≈ 17).
- Z: relativistic BW ⊗ Crystal Ball in data, because the final-state-radiation tail gives χ²/ndf 2.2 with a Gaussian resolution and 1.05 with CB; BW ⊗ Gaussian in simulation (no radiation simulated).

All shapes were chosen by fit quality, not by agreement with simulation.

**ADR-10 · CMS `pt`/`eta` columns are not used for cuts.** For about 3 % of muons the published `pt` disagrees with √(px² + py²), while the published `M` agrees with the four-vectors to within 0.6 MeV. The rows are kept and counted as warnings, and every cut uses quantities recomputed from (px, py, pz). Rejecting them would have removed 6 % of events for no physical reason.

**ADR-11 · No multiprocessing by default.** A 4-process pool was slower than one process on the reference laptop (48 s vs 29 s for 100k Z events), because of Windows process start-up and CPU contention. The option remains, with results identical to the serial path (tested). No Numba: the hot paths are already vectorised NumPy.

**ADR-12 · No external generators or units library at run time.** Pythia 8 and Geant4 are assessed in `research/LIMITATIONS.md` as future extensions. Pint or Astropy units would slow the vectorised inner loops; the units module plus boundary conversion is the proportionate choice.

**ADR-13 · 3D event display with PyVista, loaded lazily.** PyVista/VTK was chosen for detector-quality 3D rendering, using pyvistaqt's `QtInteractor` inside the existing PySide6 window (QtPy is pinned to PySide6 through `QT_API`). Measured on the reference laptop (Intel integrated graphics, 15.7 GB RAM):

| Cost | Value |
|---|---|
| Disk | about 330 MB (VTK 309 MB) |
| Memory | +about 150 MB |
| Import | 7–12 s warm, 18 s cold |
| First frame of 41 translucent surfaces | 1.9 s |
| Redraw | 5 ms |
| Event update | 60–80 ms |

Because of the import time:
- PyVista is imported in a background thread the first time the 3D tab is shown, and the VTK widgets are then created on the GUI thread. A test confirms that the import runs off the GUI thread and the event loop keeps ticking.
- The detector meshes are merged into one actor per group and cached per geometry. A new event replaces only the track, hit, vertex and arrow actors.

The 2D displays were kept and now draw the same `DisplayEvent` as the 3D view.

**ADR-14 · Shared display event.** One event is generated per request (Next event, or a finished simulation) by `visualization.make_event` in a worker thread, using the unchanged generator, hit simulation and track fit. It is redrawn until it would enter the mass spectrum (both muons reconstructed, opposite charge, analysis cuts). The mass-spectrum pipeline does not store per-hit data, so after a run the display shows a new event generated with that run's process, detector settings and seed, not one of the run's own events.

**ADR-15 · Hit residual magnification (display only).** Smearing (10–45 µm) and multiple-scattering offsets (up to millimetres) are invisible at metre scale. The 3D view can draw each hit at true crossing + k × (measured − true crossing), with k = 10, 100 or 1000, plus a connecting segment. The scene text states when this is active. The simulation data are never altered.


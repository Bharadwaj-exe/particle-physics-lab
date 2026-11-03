# Independent audit scripts

The evidence behind `docs/validation/FINAL_INDEPENDENT_AUDIT.md`. These scripts only read the production code and data; none of them modifies `particlelab/`, `tests/` or `outputs/`. Baseline: commit `440ef55`.

| Script | What it checks | Output |
|---|---|---|
| `a1_kinematics.py` | symbolic two-body decay (sympy), boosts against explicit 4×4 matrices, invariant mass against 50-digit arithmetic, angular distributions with an independently built rest frame and Collins–Soper axes, spectrum and line-shape sampling | `results/a1_kinematics.json` |
| `a2_detector.py` | helices against a Runge–Kutta Lorentz-force integration, the 0.299792458 constant from SI values, Highland scattering (analytic and empirical), missed-hit bookkeeping, fit independence from truth, curvature offset, acceptance and material against η | `results/a2_detector.json` |
| `a2b_curvature_cause.py` | whether the forward curvature offset comes from the disk radial error the circle fit ignores | `results/a2b_curvature_cause.json` |
| `a3_reco_and_display.py` | reconstructed mass against independent 40-digit four-vectors, reconstruction edge cases, 2D/3D display coordinates against independent helix geometry, 3D scene behaviour, memory over 300 updates | `results/a3_reco_and_display.json` |
| `a4_fits_data_toys.py` | fit-model dependence of the masses and widths, model-independent widths, CMS mass offsets by detector region and muon type, the published `pt` column, kinematic discrepancies, toys with independent truth, simulated subsamples | `results/a4_fits_data_toys.json` |
| `a5_architecture.py` | import graph, cycles, layer violations, broad exception handlers, duplicated constants | `results/a5_architecture.json` |
| `rerun_default.json` | the default analysis configuration with the output moved to `results/rerun_default/`, for the bit-for-bit reproducibility check | `results/rerun_default/` |

Run each one from the repository root:

```bash
python audit/a1_kinematics.py
```

`a4` regenerates the full simulated samples with the seeds of `run_analysis.py` (about 4 minutes) and caches them in `results/sim_cache.npz`. That cache is about 8 MB and does not need to be kept.

# Particle Physics Lab

Simulates J/ψ, Υ(1S) and Z decays to μ⁺μ⁻ and passes them through a CMS-inspired tracker model. It then reconstructs the muons, fits the mass peaks with extended likelihood fits, and compares the results with CMS 2011 open data (CERN Open Data record 545) using the same selection.

The model is educational: every step has a stated equation, assumption and test. It is not a CMS simulation.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -c "from particlelab.data import cms; cms.fetch()"
```

```bash
python gui.py
```

```bash
python run_analysis.py
```

```bash
python -m pytest
```

## Contents

| Path | What |
|---|---|
| `particlelab/domain` | PDG 2026 particle database (`particles_pdg.json`) and the conservation-law engine |
| `particlelab/physics` | production line shapes and spectra, two-body decay (Collins–Soper / isotropic), Lorentz boosts, invariant mass |
| `particlelab/detector` | CMS-inspired and uniform-reference geometries, hit simulation (multiple scattering, smearing, efficiency), weighted circle and line fits |
| `particlelab/analysis` | memory-bounded simulation pipeline, likelihood fitter and line shapes, toy MC, resolution studies, data–simulation comparison |
| `particlelab/data` | CMS CSV ingestion: checksum, schema and row validation, selection with cut flow |
| `particlelab/visualization` | renderer-independent event-display data (detector shapes, true and reconstructed helices, hits, vertex, momentum arrows) |
| `particlelab/app` | desktop GUI (PySide6, pyqtgraph) with a PyVista 3D event display |
| `run_analysis.py`, `configs/` | reproducible end-to-end analysis |
| `benchmarks/bench.py` | throughput and memory benchmark |
| `tests/` | unit, property-based, integration, physics-validation and regression tests |

Documentation is in `docs/`, covering engineering, physics, validation, research and the user guide. The results of the last full run are in `outputs/report.md`.

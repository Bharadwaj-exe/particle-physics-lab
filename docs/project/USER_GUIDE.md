# User Guide — Particle Physics Lab

## Install (Windows, Python 3.11+)

```bash
pip install -r requirements.txt
```

## Get the CMS data (13.9 MB, CC0 licence)

```bash
python -c "from particlelab.data import cms; cms.fetch()"
```

The file is saved to `data/raw/Dimuon_DoubleMu.csv`. Its size and adler32 checksum are verified against CERN Open Data record 545, and a `.provenance.json` record is written next to it.

## Desktop application

```bash
python gui.py
```

- **Left panel — experiment configuration:** process, number of events, random seed, detector model (CMS-inspired or uniform reference), magnetic field, hit-resolution and material scale factors, hit efficiency, multiple scattering and smearing switches, and muon pT and |η| cuts.
- **Mass spectrum:** *Run simulation* runs the full chain in the background. It shows the simulated peak, or the CMS data with its likelihood fit (signal and background) and the simulation scaled to the fitted yield. The table lists fitted masses, widths, yields, p-values and acceptance. Exports: plot as PNG, masses as CSV, run record as JSON (settings, detector, versions, results).
- **Event display (2D):** transverse and longitudinal views with barrel layers and endcap disks. It shows recorded hits (dots), lost hits (crosses), reconstructed tracks (solid), true trajectories (dashed) and momentum arrows at the vertex. The side panel has the lowest-order Feynman diagram and per-muon details.
- **Event display (3D):** a rotatable PyVista view of the same event as the 2D tab.
  - **Detector and beam:** barrel layers and endcap disks drawn from the detector settings, the beam pipe (for orientation only; not simulated as material) and the beam axis.
  - **Tracks:** true helices (thin light lines) and reconstructed helices (thick tubes; blue μ⁻, red μ⁺).
  - **Points and arrows:** recorded hits with smearing and scattering (black spheres), lost hits (grey squares), the decay vertex (green) and reconstructed momentum arrows.
  - **Controls:** camera presets (3D, along the beam, side) plus mouse rotate, zoom and pan; checkboxes to show or hide each detector group and each event element; and hit-residual magnification ×10/×100/×1000 (display only).
  - **Loading:** the renderer loads the first time the tab is opened (about 10 s on the reference laptop) while the rest of the window stays usable. *Next event* on either tab, or a finished simulation, updates both displays.
- **Detector performance:** σ(pT)/pT against pT at a chosen |η|, with and without multiple scattering (Gluckstern curve for the uniform model), the track efficiency, and the cut flow of the last simulation.
- **Physics explorer:** the particle table with sources and verification flags; a conservation-law checker that shows every law with its class (exact, accidental, approximate, kinematic); and an interactive two-body decay showing rest-frame and lab four-vectors and the invariant mass.

## Reproducible analysis

```bash
python run_analysis.py
```

```bash
python run_analysis.py --config configs/quick.json
```

Outputs go to `outputs/`:
- `figures/` (PNG, and PDF for the main plots);
- `results.json` and `results.csv`;
- `data_validation.json` and `data_cutflow.json`;
- `detector_studies.json` and `fit_validation.json`;
- `run_record.json` (provenance);
- `report.md`.

To change the physics, edit a configuration file. For example:

```json
{"seed": 1, "events": {"J/psi": 200000, "Upsilon(1S)": 50000, "Z": 50000},
 "detector": "cms_inspired", "detector_overrides": {"B": 2.0, "hit_efficiency": 0.95},
 "resolution_scale": 1.5, "material_scale": 0.5, "pt_min": 4.0, "eta_max": 2.1,
 "toys": 50, "output_dir": "outputs_B2T"}
```

## Tests and benchmarks

```bash
python -m pytest
```

```bash
python -m pytest -m "not slow"
```

```bash
python benchmarks/bench.py
```

## Python API

```python
from particlelab.analysis import simulate, fit_peak
from particlelab.detector import cms_inspired
r = simulate("Z", 50_000, cms_inspired().with_(B=3.0), seed=1)
f = fit_peak(r.reco_mass, "Z", background=False, signal="rbw_gauss")
print(f.mass, f.mass_err, r.cutflow)
```

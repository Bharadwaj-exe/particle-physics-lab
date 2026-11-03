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

- **Left panel — experiment configuration:** process (with a ✓/✗ line from the process gate: only a decay that passes the conservation checks and is implemented is simulated), number of events, random seed, detector model (CMS-inspired or uniform reference), magnetic field, hit-resolution and material scale factors, hit efficiency, multiple scattering and smearing switches, muon pT and |η| cuts, and the source of the displayed events (*Simulated events* or *CMS 2011 data events*).
- **Menus:** *File → Save image of the current tab* (Ctrl+S) saves any tab as a PNG. *View → Dark theme* (Ctrl+D) switches the whole application between light and dark. *View → Settings panel* (Ctrl+B) hides or shows the left panel. *View → Reset window layout* restores the default window and pane sizes. *View → Play / pause* (Ctrl+P) controls the event animation. The theme, window size and position, the panel and all pane sizes are remembered between sessions.
- **Window and display scaling:** the window opens maximised on screens smaller than 1450 × 860. It can be resized freely and minimised. Every divider between panes can be dragged. When a tab is too narrow for its panes side by side, they move one above the other, and the tab scrolls only if the window is smaller than its content. The layout follows Windows display scaling: at 100 %, 125 % and 150 %, no tab needs sideways scrolling on a 1366 × 768 screen.
- **Plots:** the mouse wheel or a right-drag zooms and a left-drag pans. Histograms and curves zoom only along x, with the y axis always fitted to what is visible, and no plot can be moved outside its data or the detector. Double-click a plot, or click its *A* button, to restore the default view.
- **Mass spectrum:** *Run simulation* runs the full chain in the background. It shows the simulated peak, or the CMS data with its likelihood fit (signal and background) and the simulation scaled to the fitted yield. The table lists fitted masses, widths, yields, p-values and acceptance. *Export* offers the plot as a PNG, the masses as a CSV file and the run record as JSON (settings, detector, versions, results).
- **Event display (2D):** transverse and longitudinal views with barrel layers and endcap disks. It shows recorded hits (dots), lost hits (crosses), reconstructed tracks (solid), true trajectories (dashed) and momentum arrows at the vertex. The side panel has the lowest-order Feynman diagram, a note explaining why its lines are not trajectories, and per-muon details.
  - **Animation:** the bar under the buttons plays the event (▶/⏸, ⏮ replay, time slider, ¼× to 2×). The muons leave the collision point at their real speeds, so the clock shows the physical time after the collision in ns (a muon crosses the 1.1 m tracker in about 4 ns). Each hit appears when its surface is crossed and is highlighted briefly; the fitted tracks and momentum arrows appear when both muons have left the tracker. *auto-play* replays every new event automatically. The 2D and 3D tabs share one clock.
  - **CMS data events:** with *CMS 2011 data events* selected on the left, the displays show a real dimuon event from the selected resonance's mass window (J/ψ 3.0–3.2 GeV, Υ 9.3–9.6 GeV, Z 85–97 GeV; some events in the window are background). The run and event numbers, the CMS muon types and the mass are shown. The tracks are helices drawn from the measured momenta in the 3.8 T field. The file has no hits and no vertex, so the vertex is drawn at the origin, and the hollow purple circles are where these helices cross the model's sensitive layers: computed points, not CMS hits. The drawing uses the CMS-inspired geometry whatever detector is selected.
  - **Compare simulation with CMS data:** shows a simulated event and a real event of the same resonance one above the other, on the same clock.
  - **Outer detector and muon system** (*View → Outer detector and muon system*, Ctrl+M; on by default). The muons are followed beyond the tracker through a schematic CMS-like detector: the electromagnetic calorimeter (green), the hadron calorimeter (amber), the solenoid (grey), the iron return yoke (red) and the muon stations MB1–MB4 (barrel) and ME1–ME4 (endcaps).
    - Outside the coil the magnetic field points the other way, so a muon bends back. This is the "S" shape seen in CMS transverse views.
    - Each muon loses energy in the material. Orange squares are muon-chamber hits; a red square marks where a muon ran out of energy and stopped.
    - At η = 0 a muon needs about 3 GeV to reach the muon stations and about 4 GeV to cross them all. This is one reason muon analyses cut at pT ≈ 3 GeV.
    - The side panel says which stations each muon reached. *Whole detector* / *Tracker* switches the zoom of both views.
    - The fitted tracks appear as soon as both muons have left the tracker, while the muons are still flying outwards.
    - This part is a display model only: the field is simplified (no endcap return field), the energy loss is the average rate (no fluctuations, no radiation), and nothing in the analysis uses it.
  - *Save image* saves the event views and side panel as a PNG.
- **Event display (3D):** a rotatable PyVista view of the same event as the 2D tab.
  - **Detector and beam:** barrel layers and endcap disks drawn from the detector settings, the beam pipe (for orientation only; not simulated as material) and the beam axis.
  - **Tracks:** true helices (thin light lines) and reconstructed helices (thick tubes; blue μ⁻, red μ⁺).
  - **Points and arrows:** recorded hits with smearing and scattering (black spheres), lost hits (grey squares), the decay vertex (green) and reconstructed momentum arrows.
  - **Controls:** camera presets (3D, along the beam, side) plus mouse rotate, zoom and pan; checkboxes to show or hide each detector group and each event element; and hit-residual magnification ×10/×100/×1000 (display only). The *Controls* button next to the animation bar shows or hides this column; in a narrow window the column folds away by itself.
  - **Outer detector:** drawn with a quarter cut away, as in the CMS drawings, with its own visibility check boxes (calorimeters, solenoid, return yoke, muon chambers; muons outside the tracker, muon-chamber hits). *Whole detector* shows all of it, *Tracker* zooms in on the tracker.
  - **Animation, data events, theme and images:** the same animation bar, event source and theme as the 2D tab. While the muons fly, a bright line and ball show each muon and its path so far, and the hits appear as they are made. *Save image* writes the current 3D view to a PNG.
  - **Loading:** the renderer loads the first time the tab is opened (about 10 s on the reference laptop) while the rest of the window stays usable. *Next event* on either tab, or a finished simulation, updates both displays.
- **Reconstruct it yourself:** for the displayed event (simulated, or a real CMS event; for a CMS event the chord ends at the last computed layer crossing), the tracker's measurements of each muon (chord L to the last hit, sagitta s, φ, η, charge) and the field. Work out R = L²/(8s) + s/2, pT = 0.2998·B·R, E = √(pT²cosh²η + m_μ²) and the pair mass; *Check my answers* marks each value and names common mistakes without giving the answer away. *Show worked solution* reveals every step; *Add my mass to the histogram* builds your own mass histogram over several events.
- **Detector performance:** σ(pT)/pT against pT at a chosen |η|, with and without multiple scattering (Gluckstern curve for the uniform model), the track efficiency, and the cut flow of the last simulation.
- **Physics explorer:** the particle table with sources and verification flags; a conservation-law checker that shows every law with its class (exact, accidental, approximate, kinematic) and, for an allowed and implemented decay, a *Simulate this process* button; and an interactive two-body decay showing rest-frame and lab four-vectors and the invariant mass.

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

# System Architecture — Particle Physics Lab

## Layers

```
particle-physics-lab/
├── gui.py                      entry point → particlelab.app.desktop
├── run_analysis.py             full reproducible analysis (config → figures, tables, run record)
├── benchmarks/                 performance benchmark; speed and accuracy against a Delphes-style baseline
├── rust/                       optional compiled kernels (hit simulation, track fit) used when installed
├── configs/                    run configurations (JSON)
├── data/raw/                   downloaded CMS file + provenance JSON (never modified)
├── references/                 digitised CMS reference curves (material budget, muon resolution) with provenance
├── calibration/                material fit to the CMS X0(η) reference (no dimuon data used)
├── audit/                      independent audit scripts and their evidence files
├── particlelab/
│   ├── units.py                unit conventions (GeV, m, T; c = 1) and physical constants
│   ├── provenance.py           RunConfig, environment and run records
│   ├── domain/                 what exists: particles (PDG database JSON), conservation laws
│   ├── physics/                how events are made: kinematics, production, decay
│   ├── detector/               how they are seen: geometry, hit simulation, track fitting
│   ├── analysis/               what is measured: pipeline, line shapes, likelihood fits, studies, data–sim comparison
│   ├── data/                   experimental data: CMS open-data ingestion and validation
│   ├── visualization/          renderer-independent display data (event geometry, tracks, hits)
│   ├── education/              learner logic without GUI: guided mass reconstruction with diagnosed mistakes, process gate
│   └── app/                    desktop interface (PySide6/pyqtgraph; 3D view with PyVista/VTK;
│                               shared animation clock in animation.py, light/dark themes in theme.py,
│                               font-relative sizes, responsive panes and bounded plot views in layout.py)
└── tests/                      unit · integration · physics_validation · regression
```

Dependency direction (an arrow means "imports"):

```
app ──► visualization ──► detector, physics   (display data only; no physics changes)
app ──► analysis ──► detector ──► physics ──► domain
 │         │            │                       ▲
 │         └──► data ───┴───────────────────────┘
 └──► provenance ──► detector, domain
units ◄── physics, detector
```

Rules:

- `domain` and `physics` import nothing from the layers above them.
- `app` is the only module that imports Qt or PyVista. All physics is testable without a GUI.
- `visualization` turns simulation outputs into plain arrays (NumPy only). Both the 2D (pyqtgraph) and 3D (PyVista) displays draw the same `DisplayEvent`.
- Particle constants live only in `domain/particles_pdg.json`.
- Units are fixed: GeV, metres, tesla. Conversions happen at the boundaries (GUI fields, data files) using `units.py`.

## Key interfaces

| Function / type | Purpose |
|---|---|
| `physics.generate(name, n, seed, model)` → `DecayBatch` | parent and muon four-vectors |
| `detector.cms_inspired()` / `uniform_barrel()` → `DetectorConfig` | immutable detector description |
| `detector.simulate_hits(p, q, det, rng, z_vertex)` → `Hits` | crossings, scattering, smearing, efficiency |
| `detector.fit_tracks(hits, det)` → `TrackFit` | momenta, charges, impact parameters, hit-error covariance |
| `analysis.simulate(name, n, det, seed)` → `SimResult` | full chain with a cut flow |
| `analysis.fit_peak(masses, name, …)` → `FitResult` | extended binned likelihood fit |
| `analysis.compare.compare(name, data_events, sim)` → `Comparison` | data vs simulation |
| `data.cms.load(path, selection)` → `CMSDataset` | validated, selected CMS events with a report |
| `domain.conservation.check("A -> b c")` → `Verdict` | structured conservation-law result |
| `provenance.RunConfig`, `run_record()` | reproducibility |

## Reproducibility

Every random number comes from `numpy.random.Generator`. In `simulate`, chunk *i* uses child *i* of `SeedSequence(seed)`, so results do not depend on the chunk count processed so far or on the number of worker processes (tested). `run_analysis.py` writes `run_record.json` containing:

- the configuration;
- the detector description;
- package versions and the git commit;
- the PDG edition;
- the data checksum and cut flow.

## Integration with the wider application

The Wave Function Visualizer and Atomic Model modules do not exist in the repository yet (audit, §0). When they are added, they should follow the same pattern:

- a physics package with no GUI imports;
- one `QWidget` tab class registered in a shared `QMainWindow` shell;
- the shared `units.py` and `provenance.py`.

`particlelab.app.desktop.MainWindow` is built so that more top-level tabs can be added.

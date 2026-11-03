# Limitations and Research Extensions

## What the model is and is not

The project is a transparent, testable **educational simulation**. Every step has a closed-form or numerically checked basis. It is **not** a CMS simulation and does not reproduce CMS performance quantitatively.

| Area | Simplification | Expected effect |
|---|---|---|
| Production | phenomenological pT and y shapes; no PDFs, cross sections, pile-up, or non-prompt (B → J/ψ) component | acceptance and muon pT/η spectra are model-dependent; yields are not predicted |
| Decay | λθ = 0 for quarkonia, 1 + cos²θ (Collins–Soper) for Z; no forward–backward asymmetry, no λφ terms | small acceptance effects |
| Z line shape | no parton luminosity, no γ*/Z interference | shape distortion far from the pole |
| Radiation | no final-state radiation | the data show low-mass tails (absorbed by the CB shapes) that the simulation lacks |
| Material | sensitive layers plus a few thin passive service surfaces, calibrated to the CMS X₀(η) curve (within ±12 % in 0.5-wide \|η\| ranges); linearised Gaussian scattering; no energy loss | single-muon resolution within ±15 % of CMS at 10 GeV; about 25 % optimistic at 1 GeV centrally and about 2× optimistic at 100 GeV for \|η\| > 1.5 (forward geometry and hit resolution, no alignment) |
| Tracking | ideal hit-to-track assignment; no fakes; scattering-agnostic weights | efficiency and resolution optimistic |
| Muon ID | no calorimeter or muon-system simulation; muons = tracks passing pT and η cuts | real muon-ID thresholds (p ≳ 3 GeV barrel, lower forward) are approximated by the pT cut |
| Trigger | not simulated; the data come from DoubleMu-triggered events | data pT spectra are shaped by trigger thresholds that the simulation does not have |
| Calibration | no momentum-scale correction in simulation or data | data peaks sit below PDG (J/ψ −3.6 MeV, Υ −16.5 MeV, Z −0.27 GeV), more so in the forward region |

## Level 3 extensions (assessed, not implemented)

**Pythia 8** (ref. 16).
- **Gain:** realistic quarkonium and Drell–Yan production with PDFs, initial-state radiation (Z pT), final-state photon radiation, and the non-prompt J/ψ fraction.
- **Cost:** on Windows, Pythia's Python bindings are most reliably installed through conda; pip wheels are not guaranteed.
- **Integration point:** a `ProductionBackend` that returns `DecayBatch` (parent + muon four-vectors). The detector and analysis need no changes.
- **Effort:** 1–2 weeks for one person. It must remain optional.

**Geant4** (ref. 17).
- **Gain:** full material interactions, secondaries and realistic scattering tails.
- **Cost:** C++ toolchain, geometry description, long run times. Disproportionate for this project; geant4_pybind exists but is a large dependency.
- **Recommended instead:** step-wise propagation with Highland plus Landau energy loss, or a Kalman-filter refit (TECHNICAL_DEBT D2–D3).

**Other extensions:**
- ψ(2S), Υ(2S) and Υ(3S) simulation (the database and fitter already support them);
- unbinned likelihood with iminuit;
- profile-likelihood intervals;
- a momentum-scale calibration study;
- additional channels (e.g. Z → ee with an electromagnetic calorimeter model).

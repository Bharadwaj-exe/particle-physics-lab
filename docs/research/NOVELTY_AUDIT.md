# Novelty Audit

**Date:** 3 November 2026.
**Scope:** every novelty claim made for this project, the sources behind each claim, and whether the software actually delivers what the claim describes.
**Result in one line:** the integration claim survives, in a narrower form. Two of its distinguishing features were missing from the software and have now been built, and one motivating statement overstated its source and has been corrected.

## 1. The claims audited

| # | Claim (as originally written) |
|---|---|
| C1 | **Gap.** "No learner-facing tool connects an allowed process, its kinematic simulation, a detector response, reconstruction and comparison with real collider data in one environment." |
| C2 | **Diagram tools stop at the diagram.** FeynGame and FeynCraft validate Feynman diagrams against a particle model but go no further. |
| C3 | **Reconstruction is hidden in masterclasses.** HYPATIA computes a pair's invariant mass when the learner clicks the tracks; the CMS masterclass has students place pre-computed masses into histogram bins. |
| C4 | **Simulation tools show generator stages, not detector response.** VISTAS (2026) visualises Pythia stages for education. |
| C5 | **Diagrams are misread.** "Feynman diagrams are known to elicit and perpetuate inadequate conceptions, including reading lines as trajectories." |
| C6 | **The contribution (the "diagram-to-detector pipeline").** The learner selects a valid process; the tool generates decays, propagates tracks through a magnetic field and applies detector smearing; **the learner reconstructs the invariant mass from four-vectors**; the histogram is overlaid on CMS open data; the diagram and the curved tracks are shown side by side **as explicitly different objects**. |

**Method.**
- Every cited paper was opened at the publisher, on arXiv or in an institutional record. VISTAS, the masterclass paper and the CMS tracking paper were read in full text, extracted from the PDFs.
- Prior art was searched on the web, arXiv and GitHub (queries in §6).
- The software was checked feature by feature against C6.

## 2. Reference verification

| Ref. | Status | Finding |
|---|---|---|
| FeynGame — R. V. Harlander, S. Y. Klein, M. Lipp, Comput. Phys. Commun. 256 (2020) 107465, arXiv:2003.00896 | **VERIFIED** | Authors, venue and DOI 10.1016/j.cpc.2020.107465 confirmed. Later versions exist: FeynGame-2.1 (PoS EPS-HEP2023, arXiv:2401.12778) and FeynGame 3.0 (arXiv:2501.04651; Bündgen, Harlander, Klein, Schaaf). They add amplitudes and qgraf rendering, but still no events, detectors or data. |
| FeynCraft — J. R. Gaunt, A. Owen, arXiv:2510.14082 (2025) | **VERIFIED** | Browser game: build and validate diagrams, generate the diagrams for a process, export LaTeX. Nothing on detectors or data. |
| VISTAS — B. Assi, C. Bierlich, R. Gambhir, P. Ilten, T. Menzo, S. Mrenna, M. Szewc, M. K. Wilkinson, A. Youssef, J. Zupan, "Vistas: A Visualization Interface for Particle Collision Simulations", arXiv:2606.19524 (2026), submitted to Physics Education | **VERIFIED; read in full** | Displays the Pythia event record (hard process, showers, hadronisation, decays) in the Phoenix viewer. The authors switch off "all particle selection requirements and detector elements". No detector simulation, track fitting, invariant-mass reconstruction or data comparison. The original citation lacked the authors. |
| B. Tomasik, J. Kekelakova, I. Melo, V. Pleskot, "Bridging the gap between cutting-edge science and school…", arXiv:2401.17103 (GIREP 2023 proceedings) | **VERIFIED** | States that after finding the pair, the learner clicks the tracks and HYPATIA calculates the invariant mass automatically. It is a secondary source, so the primary HYPATIA paper is now cited as well. |
| C. Kourkoumelis, S. Vourakis, "HYPATIA — an online tool for ATLAS event visualization", Phys. Educ. 49 (2014) 21 | **ADDED** (primary source for C3) | Event display derived from Atlantis; students select tracks and compute invariant masses. |
| K. Cecire, M. Bardeen, T. McCauley, "The CMS Masterclass and Particle Physics Outreach", EPJ Web Conf. 71 (2014) 00027 | **VERIFIED WITH LIMITATIONS** | Authors and venue confirmed (the citation said "et al."). Students analyse authentic CMS events; the CIMA tool histograms masses and tabulates ratios automatically. "Place pre-computed masses into histogram bins" is a fair paraphrase, but the full text was not obtained. |
| M. N. Dahlkemper, P. Klein, A. Müller, S. M. Schmeling, J. Wiener, Physics 4(4) (2022) 1331, doi:10.3390/physics4040085 | **VERIFIED; claim C5 overstated** | An interview study of **four experts**. They warn that diagrams *can* elicit and reinforce inadequate conceptions. This is expert opinion, not a measured student effect. |
| P. Lindenau, M. Kobel, G. Pospiech, "Learning difficulties in the interpretation of Feynman diagrams", WPCE-3 talk, December 2021 | **VERIFIED as a talk** | Exploratory questionnaire study with teachers and university students; learning difficulties identified and categorised. Not peer-reviewed. |
| M. Dahlkemper, *The Use of Feynman Diagrams in Physics Education: Opportunities, Challenges, Practices*, PhD thesis, University of Göttingen (2024) | **ADDED** (counter-evidence for C5) | In open comprehension tasks, upper-secondary students showed "no inherently stable or widespread inadequate conceptions". Their difficulties were terminology ("weak charge", "interaction particle"), arrows and non-sequential elements. |

## 3. Prior art for the integration claim

| Tool | Process validity check | Event generation | Detector response the learner can change | Learner does the reconstruction | Simulation compared with real data | Diagram vs trajectory made explicit |
|---|---|---|---|---|---|---|
| FeynGame, FeynCraft | yes | — | — | — | — | — |
| VISTAS | — | yes (Pythia) | — (detector elements off) | — | — | — |
| MadGraph | yes | yes | — | — | — | renders diagrams |
| HYPATIA (ATLAS masterclass) | — | — | — (real events) | clicks; mass automatic | data only | — |
| CMS masterclass (iSpy + CIMA) | — | — | — (real events) | selects; mass and histogram automatic | data only | — |
| ATLAS Open Data notebooks | — | — (fixed full-simulation samples) | — | yes, in code | **yes** | — |
| Particle Physics Playground (Siena) | — | — (supplied MC files) | — | yes, in code | partly (MC files vs data) | — |
| "Particle Playground" (single-author GitHub demo) | — | parametrised collisions | — | automatic | separate displays, no overlay | animated diagrams |
| **This project** | **yes (gate)** | **yes** | **yes: field, hit resolution, material, efficiency; resolution comes out of a hit-level fit calibrated to the published CMS material** | **yes: step by step from tracker measurements, with diagnosed mistakes** | **yes: fitted peaks and widths overlaid, with systematics** | **yes** |

**New prior art not in the original literature review:**
- **ATLAS Open Data notebooks.** Learners compute invariant masses from four-vectors themselves and compare full-simulation samples with data. This is the closest prior art for "reconstruction plus comparison with real data". It differs in that the simulation is fixed and pre-made, and the learner cannot change or inspect the detector response.
- **Particle Physics Playground** (Siena College). Code-based exercises on CMS data, with supplied Monte Carlo samples for some analyses.
- **"Particle Playground".** A small browser demo (GitHub, single author, no publication) that combines animated Feynman diagrams, parametrised collisions and real CMS event displays. It has no detector simulation, its masses are computed automatically, and simulation and data are not overlaid.

These tools must appear in any related-work section.

## 4. Verdict per claim

| # | Verdict | Why |
|---|---|---|
| C1 | **SUPPORTED, narrowed** | No tool found does all five steps. The parts exist separately, and ATLAS Open Data covers reconstruction plus a data comparison. The defensible gap is a learner-controlled detector model whose parameters visibly set the reconstructed width, compared quantitatively with real data, behind a process-validity gate. |
| C2 | **VERIFIED** | Still true for FeynGame 3.0 and FeynCraft. |
| C3 | **VERIFIED** (HYPATIA); **VERIFIED WITH LIMITATIONS** (CMS masterclass) | HYPATIA computes the mass on click (primary and secondary sources). For the CMS masterclass, the histogramming is automatic (CIMA). |
| C4 | **VERIFIED** | Confirmed by reading VISTAS in full. |
| C5 | **OVERSTATED → corrected** | The evidence is expert opinion [Dahlkemper 2022] and an exploratory talk [Lindenau 2021]. The larger student study [Dahlkemper 2024] found no stable or widespread misconceptions. The diagram–trajectory contrast is kept as a cheap precaution, not as a response to a documented student problem. |
| C6 | **PARTLY DELIVERED before this audit → now DELIVERED** | See §5. |

## 5. What the software delivered against C6

| Element of C6 | Before the audit | Now |
|---|---|---|
| Learner selects a **valid** process | Fixed list of three processes; the conservation checker was a separate tab with no link to the simulation | **Process gate** (`particlelab/education/process_gate.py`). Only a process that passes charge, B − L, B, L, Le, Lμ, Lτ and the mass threshold, and is implemented, reaches the generator. A forbidden decay is refused with the law it breaks. The conservation checker's "Simulate this process" button sends an allowed process to the simulation. |
| Decay Monte Carlo, magnetic-field propagation, detector response | Yes. Beyond the claim: hit-level simulation with multiple scattering and a circle fit, material calibrated to the published CMS X₀(η) | unchanged |
| **Learner reconstructs** the mass | **Missing.** The mass was computed automatically, the same limitation the claim criticises in HYPATIA | **"Reconstruct it yourself" tab** (`particlelab/education/reconstruction.py`). The learner gets only tracker measurements (chord, sagitta, φ, η, charge, B) and computes R, pT, E and m, with feedback per step. Typical slips are named: missing 0.2998, centimetres, pT instead of p, energies added without momenta, magnitudes instead of vectors, z dropped, degrees vs radians. Correct masses accumulate in the learner's own histogram. The exact chain reproduces the simulation's reconstructed mass to 10⁻⁹. |
| Overlay on CMS open data | Yes | Yes, now with systematic uncertainties and fits by detector region |
| Diagram and tracks as **explicitly different** objects | Side by side, with no explanation | A note under the diagram: lines are not paths, left-to-right is not distance, the Z/γ* never reaches the detector and its mass is *inferred*, and the curves are measured helices of radius pT/(0.3B) |

Tests:
- `tests/unit/test_education.py` (26 tests): the learner chain equals the simulated mass for all three resonances; R from the sagitta is exact; every diagnosed mistake; wrong answers never reveal the value; the gate on allowed, forbidden, unimplemented and unreadable decays.
- `tests/visualization/test_learner_gui.py`: the gate, the explorer button, the reconstruction tab and the diagram note, driven through the real widgets offscreen.

## 6. Revised claim (ready to use)

> **Contribution.** An open-source learning environment in which the learner (i) chooses a decay that must first pass conservation-law checks; (ii) simulates it through a hit-level tracker model, with multiple scattering and track fitting, whose material is calibrated to the published CMS material budget and whose field, hit resolution and material the learner can change; (iii) reconstructs the dimuon mass by hand from tracker measurements (sagitta → R → pT → E → m), with step-by-step diagnostic feedback; and (iv) compares the simulated peak position and width quantitatively with CMS 2011 open data. Feynman diagrams are shown alongside the measured tracks and explicitly distinguished from trajectories.
>
> **Closest prior art.**
> - FeynGame and FeynCraft (diagrams only).
> - VISTAS (generator stages, no detector or measurement).
> - HYPATIA and the CMS masterclass (real events, mass computed automatically).
> - ATLAS Open Data notebooks (learner-computed masses and a data–simulation comparison, but with fixed full-simulation samples and no learner control of the detector).
>
> **Novelty:** plausibly novel as an integration (estimated 0.6). No individual component is claimed as new.

The estimate rises from 0.55 because the two missing features now exist and the search found nothing that combines them. It does not rise further because the new prior art (ATLAS Open Data, "Particle Playground") covers more of the chain than the original review recognised.

## 7. Limits of this audit and remaining checks

- **Searches not run:** Google Scholar, ERIC and IEEE Xplore could not be queried from this environment. Run them with the queries below before submitting a paper.
  - "Feynman diagram" "event display" education
  - "invariant mass" "toy detector" students
  - "open data" "detector simulation" classroom
  - "particle physics" simulation "magnetic field" "track reconstruction" education
- **Not read in full:** the full text of Cecire et al. (2014) and the Dahlkemper thesis (the PDF host refused access; the official CERN summary was used).
- **Learning effect:** none is claimed. Education journals will expect a small study, for example pre/post tests and think-aloud interviews with ethics approval. Without one, the realistic venue is a software journal (e.g. JOSE).

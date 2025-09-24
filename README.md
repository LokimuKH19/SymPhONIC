# 🎵 SymPhONIC
**Symmetric-based Physics Oriented Neural Integral Computation**

SymPhONIC is a novel FEM-PINN / weak-form deep learning framework dedicated to analyzing impellers and rotating turbomachinery flows 🚁.

It leverages **geometric and group symmetry** to accelerate high-quality data generation, and integrates these data with **physics-oriented neural integral (weak-form) formulations** for offering data to the future surrogate modeling which includes both **structural** and **operational** parameters.

## 🔭 A Glimpse to the Future
SymPhONIC aims to provide a practical and provable bridge from high-fidelity CFD to physics-aware neural surrogates for impellers and pumps:  
- use **symmetry & sector/wedge reductions** to cheaply generate high-quality training backbones;
- the **Parametric Boundary** is dealt with hard constraints;
- enforce **weak-form (variational) residuals** so networks only require lower-order AD (better stability at high Re, especially the Main Coolant Pump in the Lead-cooled Fast Reactors);  
- fuse **data anchors + weak physics** to preserve local structures (vortices, boundary layers) while ensuring global conservation.

## Quickstart (placeholder)
This repository is a **living** project. The minimal initial files include:
- `WhyWeakFNO/` — I will continue recording the process of looking for the solver of SymPhONIC. Maybe modified to a novel or movie in the future(XD).

## License & contribution
I used GNU 3.0, temporarily. May change in the future.

> SymPhONIC — where symmetry, physics and neural computation resonate in unison.

> Its original name was Symmetric Impeller Sobolev-Underwritten Kinetics Analyzer with Supervisory (SISUKAS), maybe used in later system?

内忧外患喵

# Ultra Fuel-Efficient 250-Passenger Blended Wing Body Aircraft

**MAE 155A - Senior Design**  
**University of California, San Diego**  
**March 2026**  

**Team 2:** Alexander Burns, Martina Danieli, Anthony Gaeta, Sainirnay Mantrala, Sahil Savasere, Gregorio Zaltzman

---

## Project Overview

The **Manta** is a conceptual **Blended Wing Body (BWB)** aircraft designed to achieve ultra-high fuel efficiency for the 250-passenger long-haul market. By treating the fuselage as the primary lifting body instead of relying on a conventional tube-and-wing configuration, the design delivers a cruise lift-to-drag ratio (**L/D**) of **25.30**, addressing the stagnation in fuel burn per passenger seen in traditional architectures.

### Key Specifications (CoDR Configuration)

| Parameter              | Value                          |
|------------------------|--------------------------------|
| **Range**              | 7,400 nmi                     |
| **Cruise Mach**        | 0.85                          |
| **Cruise Altitude**    | 40,000 ft                     |
| **Passenger Capacity** | 250 (3-class configuration)   |
| **Wingspan**           | 64.67 m (Code E compliant)    |
| **Aspect Ratio**       | 5.03                          |
| **Cruise L/D**         | **25.30**                     |
| **MTOW**               | 1,435,394 N (~146,370 kg)     |
| **Cost per Seat Mile** | $0.0873                       |
| **Engines**            | 2 × aft-mounted ultra-high-bypass turbofans (BPR 9.1) with Boundary Layer Ingestion |

**Mission Profile**: Long-haul route such as Los Angeles (LAX) to Dubai (DXB), including 30 minutes loiter and 45 minutes diversion reserve.

The aircraft features a non-cylindrical pressurized cabin, aft-mounted engines for maximum boundary layer ingestion, twin canted vertical stabilizers, and a fly-by-wire control system with wing-mounted elevons.

---

## Design Analysis & Methodology

### Master Script
A comprehensive **Master Script** was developed to integrate all major disciplinary analyses into a single, consistent workflow. The script handles:

- Mission inputs and Breguet range fuel fraction calculation
- OpenVSP geometry generation, wetted area, and parasitic drag estimation
- AVL aerodynamic analysis (lift, drag, neutral point, and control surface moments)
- Component weight estimation using Raymer empirical models
- Structural stress evaluation (root bending moment and Euler beam theory)
- Operating cost modeling

This integration eliminated input inconsistencies and enabled rapid design iteration across geometry, aerodynamics, weights, structures, and economics.

### Gradient-Free Optimization (Particle Swarm Optimization)

OpenVSP and AVL evaluations lack explicit derivatives, so a **gradient-free Particle Swarm Optimization (PSO)** was implemented using the PySwarms library.

**Design Variables** (defined per reflected section of the aircraft):
- Aspect Ratio (AR)
- Sweep angle (Λ)
- Area (S)
- Taper ratio (λ)

**Objective**: Maximize cruise **L/D**  
**Constraints**:
- Material stress ratio ≤ allowable stress
- Static margin ≥ 0
- Wingspan ≤ 65 m (Code E airport gate limit)

**Optimization Process**:
1. Initial global PSO run with 40 particles over 25 iterations to explore the design space.
2. Refinement run with bounds tightened to ±20% around the best solution from the first run.

**Results**:
- Cruise L/D improved from 24.85 (initial BCR configuration) to **25.30**
- Structural stress ratio was driven to the active constraint limit
- Optimized geometries (cyan = first PSO, orange = second PSO) are shown overlaid on the baseline in `figures/PSO.png`

The global-best swarm topology with tuned cognitive (1), social (2), and inertial (0.5) parameters provided efficient convergence for this high-cost black-box problem.

### Monte Carlo Sensitivity Analysis

A large-scale **Monte Carlo simulation** using **Latin Hypercube Sampling** was performed to assess parameter sensitivities and design robustness.

**Setup**:
- Each geometric design parameter varied by ±40% around the optimized point
- Mission parameters held constant to isolate geometric effects
- Latin Hypercube sampling used instead of pure random sampling for better space coverage with fewer evaluations

**Key Insights** (supported by figures `figures/spearsman.png`, `figures/CorrelationMatrix.png`, `figures/ParallelCoords.png`, `figures/PateroA.png`, `figures/PateroB.png`, and `figures/CGSM.png`):

- **Aspect Ratio** showed the strongest positive Spearman rank correlation with cruise L/D
- **Chord length** was the dominant driver of cost per seat-mile
- Strong trade-off observed between aerodynamic efficiency and structural feasibility near the 65 m wingspan limit
- Pareto fronts illustrated the efficiency vs. wingspan relationship and the structural feasibility map highlighted stress boundaries

These sensitivity results validated the final CoDR geometry, confirmed the dominance of aspect ratio for performance, and justified not pursuing folding-wing solutions.

### Additional Technical Analyses

- **Constraint Analysis**: Carpet plot (`figures/Carpet.png`) evaluating stall, takeoff, landing, climb, ceiling, and maneuver constraints with a 5% safety margin
- **Weight Buildup**: Detailed MTOW breakdown (`figures/MTOWBuildup.png`) showing fuel (~30.8%) and payload (~27.3%) as the largest contributors
- **Aerodynamics**: Drag polars, lift curves, and L/D variation with angle of attack (`figures/AeroAll.png`, `figures/AeroClCm.png`)
- **Static Stability**: Center of gravity excursion and static margin across flight phases (`figures/CoDR_Fig_10_1_CG_and_SM.png`)
- **Dynamic Stability**: Modal parameters for short-period, phugoid, roll subsidence, spiral divergence, and Dutch roll with triplex Fly-By-Wire implementation
- **Structures**: V-n diagram (`figures/Vn.png`), material selection (CFRP primary structure, titanium reinforcements), and detailed structural layout (`figures/Structures.png`)
- **Economics**: Direct operating cost breakdown (`figures/Cost.png`) and break-even analysis requiring 249 aircraft sold at $209.5M unit price (`figures/Breakeven.png`)

---

## Aircraft Concept Highlights

- **Cabin Layout**: Triple-aisle 2-4-4-2 seating with 20 first-class seats in the forward triangular section, premium amenities, crew quarters below first class, and flexible cargo volume (`figures/cabin.png`)
- **Propulsion**: Dual rear-mounted high-bypass turbofans optimized for boundary layer ingestion to minimize drag and reduce cabin noise
- **Structures & Materials**: Carbon fiber reinforced polymer (CFRP) for primary lifting surfaces, high-modulus carbon fiber beams, aluminum ribs, titanium at engine mounts and high-load transitions
- **Controls**: Fly-by-wire system with wing-mounted elevons and twin inclined vertical stabilizers
- **Visuals**: 3-view CAD drawings and high-fidelity render (`figures/CAD.png`, `figures/Render.png`)

---

## Documentation

- **`MAE155_Team2_CoDR_Slides.pdf`**: 19-page presentation slides
- **`MAE155_Team2_CoDR.pdf`**: Full 23+ page technical report (with appendices)
- **`main.tex`**: LaTeX source file for the report

Both PDF files are self-contained and provide full context for every analysis and result.

---

## Team Contributions

- **Alexander Burns**: Master Script development, Monte carlo Sensitivity Analysis, Gradient-Free PSO, AVL & OpenVSP Integration, Controls
- **Sainirnay Mantrala**: Gradient-Free PSO, Structural Layout and Weight Integration
- **Martina Danieli**: Aerodynamics, Design Choices, Break-even Analysis
- **Gregorio Zaltzman**: Initial Sizing, Aerodynamics, Weights, CAD
- **Sahil Savasere**: Passenger Cabin Configuration, Dynamic Stability and Controls
- **Anthony Gaeta**: Propulsion System, CAD, Structures and Materials, Operating Cost

---

## Citation

If you use or reference this work, please cite:

> Burns, A., Danieli, M., Gaeta, A., Mantrala, S., Savasere, S., & Zaltzman, G. (2026). *Manta BWB: Conceptual Design of an Ultra Fuel-Efficient 250-Passenger Blended Wing Body Aircraft*. MAE 155A, University of California, San Diego.

---

**UC San Diego**  
**Department of Mechanical and Aerospace Engineering**  
**MAE 155A – Aerospace Engineering Design I**  
**March 2026**

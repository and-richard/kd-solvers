# kd-solvers
# Numerical Evidence of Kapitza–Dirac Momentum Transfer in Weakly Relativistic Crossed Lasers

Companion numerical simulation code for the article submitted to *Physical Review A*.

**Author:** Richard-Andrei Radu  
**Affiliation:** Faculty of Physics, University of Bucharest, Măgurele, Ilfov, Romania  
**Corresponding Contact:** radu.richard.a@gmail.com  

---

## Overview

This repository contains the production Julia solvers used to model electron dynamics in weakly relativistic crossed laser pulses. The numerical framework investigates momentum-transfer bounds and caustic spatial compression under both semiclassical hyperbolic and complex conservation laws.

### Script Manifest

* **`solver_main.jl`**  
Main solver integrating the 1D semiclassical hyperbolic conservation law for the interaction phase gradient. Performs simulations across pulse durations and initial momentum states to determine the resulting Kapitza–Dirac momentum transfer (generates Figures 2–5 and Table I of the manuscript).

* **`solver_auxiliary.jl`**  
Diagnostic solver integrating the coupled complex conservation law retaining both log-amplitude and phase gradients. Investigates spatial compression dynamics, caustic formation, and computational resolution limits (generates Figure 1 of the manuscript).

---

## Requirements & Environment

* **Language:** [Julia](https://julialang.org/) (v1.9 or higher recommended)
* **Standard Libraries:** `LinearAlgebra`, `Printf`, `Statistics`
* **External Dependencies:**
  * Differential equation solvers: `OrdinaryDiffEq`, `OrdinaryDiffEqSSPRK`
  * Numerical tools & utilities: `DataInterpolations`, `Integrals`, `Polyester`, `StaticArrays`, `SpecialFunctions`, `Parameters`, `JLD2`
  * Visualization: `CairoMakie`, `LaTeXStrings`

### Installation

To install all required packages into your global Julia environment, run:

```julia
using Pkg
Pkg.add([
    "OrdinaryDiffEq",
    "OrdinaryDiffEqSSPRK",
    "DataInterpolations",
    "Integrals",
    "Polyester",
    "StaticArrays",
    "SpecialFunctions",
    "Parameters",
    "JLD2",
    "CairoMakie",
    "LaTeXStrings"
])
```

Alternatively, if using this repository's local project environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## Execution Workflow

Both scripts are self-contained and execute without external CLI configuration files. Simulation parameters, spatial grids, and numerical tolerances are declared directly inside the source code:

1. Open `solver_main.jl` or `solver_auxiliary.jl` in a text editor or IDE (such as VS Code with the Julia extension).
2. Set or inspect the physical parameters (such as potential amplitude $A_0$, pulse shelf duration $L$, standard deviation $\sigma$, initial electron momentum $\mathbf{p}$, etc.) within the `run_config` tuple located in Section 5 (*Simulation Configuration, Execution and Workflow*).
3. Execute the simulation or post-processing plotting functions directly in an interactive Julia REPL.

---

## Data Availability & Supplemental Material

An instance of these scripts is archived in the journal's official Supplemental Material deposit (`code.zip`). Tabular data across varying shelf durations and momentum states are reported in the manuscript and Supplemental Material document.

---

## License

This project is licensed under the [MIT License](LICENSE).

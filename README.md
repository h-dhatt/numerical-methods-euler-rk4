
# Numerical Methods Project — Euler vs RK4 (Logistic Growth & Harmonic Oscillator)

**Author:** Harleen Dhatt (B.Sc. (Honours) Math & Stats, McMaster)  

## Objectives
1. Implement **Euler** and **Runge–Kutta 4 (RK4)** methods for ODEs.
2. Evaluate **accuracy and convergence** against known exact solutions.
3. Visualize **time series**, **phase portrait**, and **error vs. step size** on log–log axes.

## Problems Studied
- **Logistic Growth**: \( y' = r y (1 - y/K) \), closed-form exact solution used for error analysis.
- **Simple Harmonic Oscillator**: \( y'' + \omega^2 y = 0 \), converted to a first-order system with exact solution.

## Highlights
- Built from scratch **Euler** and **RK4** solvers; verified **first-order** (slope ~1) and **fourth-order** (slope ~4) convergence rates on log–log error plots.
- Provided rigorous comparison to exact solutions and included diagnostics (phase portrait for SHO).
- Reproducible code with clear plots and CSV error tables.

## Repository Structure
```
project2_numerical/
├── analysis_script.py
├── logistic_errors.csv
├── sho_errors.csv
├── fig_logistic_timeseries.png
├── fig_logistic_error_convergence.png
├── fig_sho_timeseries.png
├── fig_sho_phase_portrait.png
└── fig_sho_error_convergence.png
```

## How to Run
```bash
pip install numpy pandas matplotlib
python analysis_script.py
```

## Talking Points
- Explain **why RK4 outperforms Euler**: local truncation error \(O(h^5)\) leading to global error \(O(h^4)\) vs. Euler's \(O(h)\).
- Point out the **phase portrait**: RK4 preserves the elliptical orbit of the SHO much better than Euler (which spirals due to numerical damping/instability).
- Discuss trade-offs: Euler is simple but inaccurate; RK4 requires more function evaluations but achieves far smaller error for the same step size.

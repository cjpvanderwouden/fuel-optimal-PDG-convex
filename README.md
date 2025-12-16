# Fuel-Optimal Powered Descent Guidance via Convex Optimization

Implementation of fuel-optimal powered descent guidance using convex optimization, based on the **lossless convexification** approach pioneered by Açıkmeşe & Ploen (2007).

## Overview

This project solves the problem of landing a rocket optimally—minimizing fuel consumption while satisfying safety and physical constraints. The key insight is that the inherently non-convex problem (due to minimum thrust constraints) admits a **lossless convexification**: relaxing the equality constraint to an inequality yields a second-order cone program (SOCP) whose optimal solution satisfies the original constraint.

### Key Features

- **SOCP Formulation**: Transforms non-convex optimal control into a tractable convex program
- **Lossless Relaxation**: Proves that the convex relaxation is tight at optimality
- **Real-Time Capable**: Solve times ~400ms (suitable for onboard guidance with optimization)
- **Constraint Handling**: Glideslope (terrain avoidance), pointing (gimbal limits), thrust bounds
- **Sensitivity Analysis**: Sweeps over throttle limits and glideslope constraints

## Problem Formulation

**Objective**: Minimize ∫‖T(t)‖ dt (fuel consumption)

**Subject to**:
- **Dynamics**: ṙ = v, v̇ = T/m + g, ṁ = -‖T‖/vₑ
- **Thrust bounds**: T_min ≤ ‖T(t)‖ ≤ T_max  (non-convex!)
- **Glideslope**: rz ≥ tan(γ) · ‖(rx, ry)‖
- **Pointing**: Tz ≥ cos(θ_max) · ‖T‖
- **Terminal**: r(tf) = 0, v(tf) = 0

The key insight: relaxing ‖u‖ = σ to ‖u‖ ≤ σ yields an SOCP, and **any slack is suboptimal** (you'd be burning fuel without producing thrust). The optimal solution automatically satisfies ‖u‖ = σ.

## Results

### Optimal Trajectory

| Metric | Value |
|--------|-------|
| Flight time | 25.1 s |
| Fuel consumed | 2,888 kg |
| Final mass | 22,112 kg |
| Solve time | ~400 ms |

The thrust profile exhibits the classic **bang-bang** structure predicted by optimal control theory:
- Maximum thrust initially (aggressive braking)
- Minimum thrust during mid-flight
- Maximum thrust at the end (final braking)

### Grid Search over Flight Time

The problem has a narrow **feasibility window** (tf ∈ [25, 27]s). This is because:
- Too short: Can't reach the target
- Too long: T_min > weight means you can't "coast"—running the engine too long burns all the fuel

### Sensitivity Analysis

There's a sharp **threshold effect** on minimum throttle: above ~30% of max thrust, landing becomes infeasible. This has real engineering implications—SpaceX's Merlin engines throttle to ~40%, which drives their "suicide burn" landing profile.

**Dependencies**:
- numpy
- cvxpy
- clarabel
- matplotlib

## Usage

python main.py --output ./output

This will:
1. Grid search over flight time to find optimal tf
2. Generate all trajectory visualizations
3. Run sensitivity analyses
4. Save results to output/

## Project Structure

    powered-descent-guidance/
    ├── README.md
    ├── requirements.txt
    │
    ├── config.py
    ├── dynamics.py
    ├── problem.py
    ├── solver.py
    ├── plots.py
    ├── sensitivity.py
    ├── main.py
    │
    ├── animate_landing.py
    ├── animate_tmin.py
    ├── animate_dashboard.py
    │
    ├── technical_note.pdf
    │
    └── output/
        ├── figures/
        ├── trajectory.npz
        ├── results.json
        ├── landing_animation.mp4
        ├── tmin_threshold_animation.mp4
        ├── dashboard_animation.mp4
        └── raw_terminal_outputs/

## Technical Notes

### Design Choices

1. **Grid search over tf**: The free-final-time problem has bilinear terms (tf · state), making it non-convex. Instead of successive convexification, we solve a grid of fixed-tf problems and pick the best. This is simpler, robust, and produces the fuel vs. tf curve as a byproduct.

2. **Conservative thrust bounds**: We use ρ_min = T_min/m0 and ρ_max = T_max/m_dry (worst-case over all masses). This guarantees feasibility without iterating. SCP refinement is available but not required for this scenario.

3. **Discretization**: N=50 time steps with zero-order hold on controls. The state transition uses exact integration for the linear-plus-constant-acceleration dynamics.

### Losslessness Caveat

The classic proof of losslessness assumes thrust bounds independent of state. When bounds depend on mass (ρ_min = T_min · e^{-z}), modifying σ changes z, potentially violating bounds. For rigorous treatment, either:
- Use conservative fixed bounds (our approach)
- Use successive convexification (also implemented)

See Section 4.5 of the technical note for details.

References

- Açıkmeşe, B. and Ploen, S.R. (2007). Convex programming approach to powered descent guidance for Mars landing. Journal of Guidance, Control, and Dynamics, 30(5):1353–1366.
- Açıkmeşe, B., Carson, J.M., and Blackmore, L. (2013). Lossless convexification of nonconvex control bound and pointing constraints of the soft landing optimal control problem. IEEE Transactions on Control Systems Technology, 21(6):2104–2113.
- Blackmore, L., Açıkmeşe, B., and Scharf, D.P. (2010). Minimum-landing-error powered-flight guidance for Mars landing using convex optimization. Journal of Guidance, Control, and Dynamics, 33(4):1161–1171.
- Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
- Mao, Y., Szmuk, M., and Açıkmeşe, B. (2016). Successive convexification of non-convex optimal control problems and its convergence properties. In IEEE Conference on Decision and Control, pages 3636–3641.
- Szmuk, M. and Açıkmeşe, B. (2018). Successive convexification for 6-DoF Mars rocket powered landing with free-final-time. In AIAA Guidance, Navigation, and Control Conference.
## Author

Casper J.P. van der Wouden - December 2025

---

*This project demonstrates the intersection of optimal control theory, convex optimization, and aerospace engineering—the same mathematics that enables autonomous rocket landings. This is a 'personal project' out of interest, I am an econ. student..*

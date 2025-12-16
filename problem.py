"""
CVXPY problem formulation for fuel-optimal powered descent.

Implements the lossless convexification approach:
- Relaxes ||u|| = σ to ||u|| <= σ (second-order cone)
- Uses conservative fixed bounds for thrust acceleration
- Solves as SOCP

The relaxation is tight at optimality (any slack would be suboptimal
since we're minimizing σ which appears in mass depletion).
"""

import cvxpy as cp
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from config import DescentConfig
from dynamics import build_state_transition_matrices


@dataclass
class ProblemVariables:
    """Container for CVXPY decision variables."""
    r: cp.Variable      # Position (N+1, 3)
    v: cp.Variable      # Velocity (N+1, 3)
    z: cp.Variable      # Log-mass (N+1,)
    u: cp.Variable      # Thrust acceleration (N, 3)
    sigma: cp.Variable  # Thrust magnitude (N,)


def build_problem(config: DescentConfig, 
                  N: int,
                  tf: float,
                  z_ref: Optional[np.ndarray] = None,
                  use_scp_bounds: bool = False) -> Tuple[cp.Problem, ProblemVariables]:
    """
    Build CVXPY problem for powered descent guidance.
    
    Args:
        config: Problem configuration
        N: Number of time steps
        tf: Fixed final time [s]
        z_ref: Reference log-mass trajectory for SCP bounds (N+1,)
               If None, uses conservative fixed bounds
        use_scp_bounds: If True and z_ref provided, use mass-dependent bounds
        
    Returns:
        prob: CVXPY Problem object
        vars: ProblemVariables container
    """
    dt = tf / N
    
    # Build state transition matrices
    A, B, c = build_state_transition_matrices(config, dt)
    
    # =========================================================================
    # Decision Variables
    # =========================================================================
    r = cp.Variable((N + 1, 3), name='position')
    v = cp.Variable((N + 1, 3), name='velocity')
    z = cp.Variable(N + 1, name='log_mass')
    u = cp.Variable((N, 3), name='thrust_accel')
    sigma = cp.Variable(N, name='thrust_mag')
    
    vars = ProblemVariables(r=r, v=v, z=z, u=u, sigma=sigma)
    
    # =========================================================================
    # Objective: Minimize fuel consumption
    # =========================================================================
    # min Σ σ_k * dt ≈ ∫ σ dt = ∫ ||T||/m dt ∝ propellant mass
    objective = cp.Minimize(cp.sum(sigma) * dt)
    
    # =========================================================================
    # Constraints
    # =========================================================================
    constraints = []
    
    # -------------------------------------------------------------------------
    # Initial conditions
    # -------------------------------------------------------------------------
    constraints.append(r[0] == config.r0_vec)
    constraints.append(v[0] == config.v0_vec)
    constraints.append(z[0] == config.z0)
    
    # -------------------------------------------------------------------------
    # Terminal conditions
    # -------------------------------------------------------------------------
    constraints.append(r[N] == np.zeros(3))      # Land at origin
    constraints.append(v[N] == np.zeros(3))      # Zero velocity
    # z[N] is free (we maximize it by minimizing fuel)
    
    # -------------------------------------------------------------------------
    # Dynamics constraints
    # -------------------------------------------------------------------------
    for k in range(N):
        # State at step k
        x_k = cp.hstack([r[k], v[k], z[k]])
        
        # Control at step k
        w_k = cp.hstack([u[k], sigma[k]])
        
        # Next state
        x_next = A @ x_k + B @ w_k + c
        
        # Enforce dynamics
        constraints.append(r[k + 1] == x_next[:3])
        constraints.append(v[k + 1] == x_next[3:6])
        constraints.append(z[k + 1] == x_next[6])
    
    # -------------------------------------------------------------------------
    # Thrust constraints
    # -------------------------------------------------------------------------
    # Get bounds
    if use_scp_bounds and z_ref is not None:
        # Mass-dependent bounds from reference trajectory
        # ρ_min(k) = T_min * e^{-z_ref(k)}, ρ_max(k) = T_max * e^{-z_ref(k)}
        rho_min = config.T_min * np.exp(-z_ref[:-1])  # (N,)
        rho_max = config.T_max * np.exp(-z_ref[:-1])  # (N,)
    else:
        # Conservative fixed bounds (feasible for all mass values)
        rho_min_const, rho_max_const = config.thrust_bounds_conservative()
        rho_min = np.full(N, rho_min_const)
        rho_max = np.full(N, rho_max_const)
    
    for k in range(N):
        # Second-order cone: ||u_k|| <= σ_k
        # This is the relaxed constraint (equality at optimality by losslessness)
        constraints.append(cp.norm(u[k]) <= sigma[k])
        
        # Thrust magnitude bounds
        constraints.append(sigma[k] >= rho_min[k])
        constraints.append(sigma[k] <= rho_max[k])
        
        # Pointing constraint: u_z >= cos(θ_max) * σ
        constraints.append(u[k, 2] >= config.cos_point * sigma[k])
    
    # -------------------------------------------------------------------------
    # Glideslope constraint: r_z >= tan(θ_gs) * ||(r_x, r_y)||
    # -------------------------------------------------------------------------
    # Note: This implicitly requires r_z >= 0 (we're above ground)
    for k in range(N + 1):
        constraints.append(r[k, 2] >= config.tan_gs * cp.norm(r[k, :2]))
    
    # -------------------------------------------------------------------------
    # Mass bounds: z_dry <= z <= z_0
    # -------------------------------------------------------------------------
    constraints.append(z >= config.z_dry)
    constraints.append(z <= config.z0)
    
    # =========================================================================
    # Build and return problem
    # =========================================================================
    prob = cp.Problem(objective, constraints)
    
    return prob, vars


def extract_solution(vars: ProblemVariables) -> dict:
    """
    Extract solution values from CVXPY variables.
    
    Args:
        vars: ProblemVariables with solved values
        
    Returns:
        Dictionary with numpy arrays
    """
    return {
        'r': vars.r.value,
        'v': vars.v.value,
        'z': vars.z.value,
        'u': vars.u.value,
        'sigma': vars.sigma.value,
        'm': np.exp(vars.z.value)
    }


def verify_losslessness(u: np.ndarray, sigma: np.ndarray, tol: float = 1e-4) -> dict:
    """
    Verify that the relaxation is tight (||u|| ≈ σ at optimality).
    
    Args:
        u: Optimal thrust acceleration (N, 3)
        sigma: Optimal thrust magnitude (N,)
        tol: Tolerance for relative error
        
    Returns:
        Dictionary with verification results
    """
    u_norm = np.linalg.norm(u, axis=1)
    slack = sigma - u_norm
    relative_slack = slack / sigma
    
    return {
        'is_tight': np.all(relative_slack < tol),
        'max_absolute_slack': np.max(slack),
        'max_relative_slack': np.max(relative_slack),
        'mean_relative_slack': np.mean(relative_slack),
        'u_norm': u_norm,
        'sigma': sigma
    }


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG
    N = 50
    tf = 30.0
    
    print(f"Building problem with N={N}, tf={tf}s...")
    prob, vars = build_problem(config, N, tf)
    
    print(f"Problem built:")
    print(f"  Variables: {sum(v.size for v in prob.variables())}")
    print(f"  Constraints: {len(prob.constraints)}")
    print(f"  Is DCP: {prob.is_dcp()}")


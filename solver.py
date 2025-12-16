"""
Solver for fuel-optimal powered descent problem.

Handles:
- Single solve for fixed tf
- Grid search over tf to find optimal flight time
- SCP iterations for refined mass-dependent bounds
- Diagnostic logging
"""

import cvxpy as cp
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import time

from config import DescentConfig, DEFAULT_CONFIG
from problem import build_problem, extract_solution, verify_losslessness
from dynamics import propagate_trajectory, check_constraints


@dataclass
class SolveResult:
    """Container for solve results."""
    success: bool
    status: str
    fuel_used: float           # kg
    final_mass: float          # kg
    tf: float                  # s
    solve_time: float          # s
    trajectory: dict           # Full trajectory data
    losslessness: dict         # Verification of relaxation tightness
    constraints: dict          # Constraint satisfaction
    
    def summary(self) -> str:
        """Return summary string."""
        if self.success:
            return f"""
Solve Result: {self.status}
==============================
Flight time:   {self.tf:.2f} s
Fuel used:     {self.fuel_used:.1f} kg
Final mass:    {self.final_mass:.1f} kg
Solve time:    {self.solve_time*1000:.1f} ms

Losslessness check:
  Tight: {self.losslessness['is_tight']}
  Max relative slack: {self.losslessness['max_relative_slack']:.2e}
"""
        else:
            return f"Solve FAILED: {self.status}"


def solve_fixed_tf(config: DescentConfig,
                   tf: float,
                   N: Optional[int] = None,
                   solver: str = 'CLARABEL',
                   verbose: bool = False,
                   z_ref: Optional[np.ndarray] = None) -> SolveResult:
    """
    Solve the powered descent problem for fixed final time.
    
    Args:
        config: Problem configuration
        tf: Final time [s]
        N: Number of time steps (default: config.N)
        solver: CVXPY solver to use
        verbose: Print solver output
        z_ref: Reference log-mass for SCP bounds
        
    Returns:
        SolveResult object
    """
    if N is None:
        N = config.N
    
    dt = tf / N
    
    # Build problem
    use_scp = z_ref is not None
    prob, vars = build_problem(config, N, tf, z_ref=z_ref, use_scp_bounds=use_scp)
    
    # Solve
    start_time = time.time()
    try:
        if solver == 'ECOS':
            prob.solve(solver=cp.ECOS, verbose=verbose)
        elif solver == 'CLARABEL':
            prob.solve(solver=cp.CLARABEL, verbose=verbose)
        elif solver == 'SCS':
            prob.solve(solver=cp.SCS, verbose=verbose, max_iters=5000)
        else:
            prob.solve(verbose=verbose)
    except Exception as e:
        return SolveResult(
            success=False,
            status=f"Solver exception: {str(e)}",
            fuel_used=np.inf,
            final_mass=0.0,
            tf=tf,
            solve_time=time.time() - start_time,
            trajectory={},
            losslessness={},
            constraints={}
        )
    
    solve_time = time.time() - start_time
    
    # Check status
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return SolveResult(
            success=False,
            status=prob.status,
            fuel_used=np.inf,
            final_mass=0.0,
            tf=tf,
            solve_time=solve_time,
            trajectory={},
            losslessness={},
            constraints={}
        )
    
    # Extract solution
    sol = extract_solution(vars)
    
    # Compute fuel used
    final_mass = sol['m'][-1]
    fuel_used = config.m0 - final_mass
    
    # Verify losslessness
    lossless = verify_losslessness(sol['u'], sol['sigma'])
    
    # Build full trajectory dict
    trajectory = {
        'r': sol['r'],
        'v': sol['v'],
        'z': sol['z'],
        'm': sol['m'],
        'u': sol['u'],
        'sigma': sol['sigma'],
        't': np.linspace(0, tf, N + 1),
        't_ctrl': np.linspace(0, tf, N + 1)[:-1] + dt / 2,
        'dt': dt,
        'tf': tf,
        'N': N
    }
    
    # Check constraints
    constraints = check_constraints(config, trajectory)
    
    return SolveResult(
        success=True,
        status=prob.status,
        fuel_used=fuel_used,
        final_mass=final_mass,
        tf=tf,
        solve_time=solve_time,
        trajectory=trajectory,
        losslessness=lossless,
        constraints=constraints
    )


def solve_grid_search(config: DescentConfig,
                      tf_min: float = 15.0,
                      tf_max: float = 60.0,
                      n_grid: int = 30,
                      N: Optional[int] = None,
                      solver: str = 'CLARABEL',
                      verbose: bool = False) -> Tuple[SolveResult, List[dict]]:
    """
    Grid search over final time to find fuel-optimal tf.
    
    Since free-final-time introduces bilinear terms (tf * state),
    we solve a sequence of fixed-tf problems and pick the best.
    
    Args:
        config: Problem configuration
        tf_min: Minimum flight time to consider [s]
        tf_max: Maximum flight time to consider [s]
        n_grid: Number of grid points
        N: Time steps per problem
        solver: CVXPY solver
        verbose: Print progress
        
    Returns:
        best_result: SolveResult for optimal tf
        grid_results: List of (tf, fuel, status) for all grid points
    """
    tf_values = np.linspace(tf_min, tf_max, n_grid)
    
    grid_results = []
    best_result = None
    best_fuel = np.inf
    
    if verbose:
        print(f"Grid search over tf ∈ [{tf_min}, {tf_max}] with {n_grid} points...")
    
    for i, tf in enumerate(tf_values):
        result = solve_fixed_tf(config, tf, N=N, solver=solver, verbose=False)
        
        grid_results.append({
            'tf': tf,
            'fuel_used': result.fuel_used if result.success else np.inf,
            'success': result.success,
            'status': result.status
        })
        
        if result.success and result.fuel_used < best_fuel:
            best_fuel = result.fuel_used
            best_result = result
        
        if verbose:
            status_str = f"{result.fuel_used:.1f} kg" if result.success else result.status
            print(f"  tf = {tf:.1f}s: {status_str}")
    
    if best_result is None:
        # All failed - return last failed result
        return SolveResult(
            success=False,
            status="All grid points infeasible",
            fuel_used=np.inf,
            final_mass=0.0,
            tf=0.0,
            solve_time=0.0,
            trajectory={},
            losslessness={},
            constraints={}
        ), grid_results
    
    if verbose:
        print(f"\nOptimal: tf = {best_result.tf:.2f}s, fuel = {best_result.fuel_used:.1f} kg")
    
    return best_result, grid_results


def solve_with_scp(config: DescentConfig,
                   tf: float,
                   N: Optional[int] = None,
                   max_iter: int = 5,
                   tol: float = 1e-3,
                   solver: str = 'CLARABEL',
                   verbose: bool = False) -> SolveResult:
    """
    Solve with successive convex programming for mass-dependent bounds.
    
    Iteratively refines the thrust bounds using the mass trajectory
    from the previous solution.
    
    Args:
        config: Problem configuration
        tf: Final time [s]
        N: Number of time steps
        max_iter: Maximum SCP iterations
        tol: Convergence tolerance on fuel change
        solver: CVXPY solver
        verbose: Print iteration info
        
    Returns:
        Final SolveResult
    """
    if N is None:
        N = config.N
    
    # Initial solve with conservative bounds
    result = solve_fixed_tf(config, tf, N=N, solver=solver, verbose=False)
    
    if not result.success:
        return result
    
    if verbose:
        print(f"SCP iteration 0: fuel = {result.fuel_used:.2f} kg (conservative bounds)")
    
    prev_fuel = result.fuel_used
    
    for i in range(max_iter):
        # Use previous solution's mass trajectory as reference
        z_ref = result.trajectory['z']
        
        # Solve with refined bounds
        result = solve_fixed_tf(config, tf, N=N, solver=solver, 
                               verbose=False, z_ref=z_ref)
        
        if not result.success:
            if verbose:
                print(f"SCP iteration {i+1}: FAILED ({result.status})")
            break
        
        fuel_change = abs(result.fuel_used - prev_fuel)
        
        if verbose:
            print(f"SCP iteration {i+1}: fuel = {result.fuel_used:.2f} kg "
                  f"(Δ = {fuel_change:.4f} kg)")
        
        if fuel_change < tol:
            if verbose:
                print(f"Converged after {i+1} iterations")
            break
        
        prev_fuel = result.fuel_used
    
    return result


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    
    print(config.summary())
    
    # Test single solve
    print("\n" + "="*60)
    print("Testing single solve (tf = 30s)...")
    print("="*60)
    
    result = solve_fixed_tf(config, tf=30.0, verbose=False)
    print(result.summary())
    
    # Test grid search
    print("\n" + "="*60)
    print("Testing grid search...")
    print("="*60)
    
    best, grid = solve_grid_search(config, tf_min=20, tf_max=50, n_grid=15, verbose=True)
    
    if best.success:
        print(best.summary())

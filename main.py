"""
Fuel-Optimal Powered Descent Guidance
======================================

Main entry point for the project. Runs:
1. Grid search over flight time to find optimal tf
2. Generates trajectory visualizations
3. Sensitivity analysis over glideslope and throttle limits

Usage:
    python main.py              # Full run with all outputs
    python main.py --quick      # Quick test (single tf, no sensitivity)
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime

from config import DescentConfig, DEFAULT_CONFIG
from solver import solve_fixed_tf, solve_grid_search, solve_with_scp
from plots import (
    plot_3d_trajectory, plot_2d_trajectory, plot_time_series,
    plot_losslessness, plot_grid_search, create_all_figures
)
from sensitivity import run_all_sensitivity


def run_full(output_dir: str = './output', verbose: bool = True):
    """
    Run complete analysis pipeline.
    
    1. Grid search over tf to find fuel-optimal flight time
    2. Generate all trajectory visualizations
    3. Run sensitivity analyses
    4. Save results to JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    config = DEFAULT_CONFIG
    
    if verbose:
        print("="*70)
        print("FUEL-OPTIMAL POWERED DESCENT GUIDANCE")
        print("="*70)
        print(config.summary())
    
    # =========================================================================
    # Step 1: Grid search over tf
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: Grid Search over Flight Time")
        print("="*70)
    
    best_result, grid_results = solve_grid_search(
        config,
        tf_min=22.0,
        tf_max=32.0,
        n_grid=30,
        verbose=verbose
    )
    
    if not best_result.success:
        print("ERROR: No feasible solution found!")
        return
    
    if verbose:
        print(best_result.summary())
    
    # =========================================================================
    # Step 2: Generate visualizations
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: Generating Visualizations")
        print("="*70)
    
    # 3D trajectory
    fig = plot_3d_trajectory(config, best_result.trajectory)
    fig.savefig(os.path.join(figures_dir, 'trajectory_3d.png'), dpi=150, bbox_inches='tight')
    if verbose:
        print("  Saved: trajectory_3d.png")
    
    # 2D projections
    fig = plot_2d_trajectory(config, best_result.trajectory)
    fig.savefig(os.path.join(figures_dir, 'trajectory_2d.png'), dpi=150, bbox_inches='tight')
    if verbose:
        print("  Saved: trajectory_2d.png")
    
    # Time series
    fig = plot_time_series(config, best_result.trajectory)
    fig.savefig(os.path.join(figures_dir, 'time_series.png'), dpi=150, bbox_inches='tight')
    if verbose:
        print("  Saved: time_series.png")
    
    # Losslessness verification
    fig = plot_losslessness(best_result.trajectory)
    fig.savefig(os.path.join(figures_dir, 'losslessness.png'), dpi=150, bbox_inches='tight')
    if verbose:
        print("  Saved: losslessness.png")
    
    # Grid search curve
    fig = plot_grid_search(grid_results)
    fig.savefig(os.path.join(figures_dir, 'grid_search.png'), dpi=150, bbox_inches='tight')
    if verbose:
        print("  Saved: grid_search.png")
    
    # =========================================================================
    # Step 3: Sensitivity Analysis
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: Sensitivity Analysis")
        print("="*70)
    
    sensitivity_results = run_all_sensitivity(
        config, 
        tf=best_result.tf,
        output_dir=figures_dir,
        verbose=verbose
    )
    
    # =========================================================================
    # Step 4: Save results to JSON
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: Saving Results")
        print("="*70)
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'm0': config.m0,
            'm_dry': config.m_dry,
            'Isp': config.Isp,
            'T_max': config.T_max,
            'T_min': config.T_min,
            'gamma_gs_deg': config.gamma_gs_deg,
            'theta_max_deg': config.theta_max_deg,
            'r0': list(config.r0),
            'v0': list(config.v0),
        },
        'optimal_solution': {
            'tf': float(best_result.tf),
            'fuel_used_kg': float(best_result.fuel_used),
            'final_mass_kg': float(best_result.final_mass),
            'solve_time_ms': float(best_result.solve_time * 1000),
            'losslessness': {
                'is_tight': bool(best_result.losslessness['is_tight']),
                'max_relative_slack': float(best_result.losslessness['max_relative_slack'])
            }
        },
        'grid_search': {
            'tf_values': [float(r['tf']) for r in grid_results],
            'fuel_values': [float(r['fuel_used']) if r['success'] else None for r in grid_results],
            'feasible': [bool(r['success']) for r in grid_results]
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    if verbose:
        print(f"  Saved: results.json")
    
    # Save trajectory data as NPZ
    np.savez(
        os.path.join(output_dir, 'trajectory.npz'),
        t=best_result.trajectory['t'],
        r=best_result.trajectory['r'],
        v=best_result.trajectory['v'],
        m=best_result.trajectory['m'],
        u=best_result.trajectory['u'],
        sigma=best_result.trajectory['sigma']
    )
    
    if verbose:
        print(f"  Saved: trajectory.npz")
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print(f"""
Results saved to: {output_dir}/
  - results.json     : Summary of optimal solution
  - trajectory.npz   : Full trajectory data (numpy format)
  - figures/         : All visualizations

Key results:
  Optimal flight time:  {best_result.tf:.2f} s
  Fuel consumed:        {best_result.fuel_used:.1f} kg
  Final mass:           {best_result.final_mass:.1f} kg
  Propellant fraction:  {100 * best_result.fuel_used / config.max_propellant:.1f}% of available

Relaxation verification:
  Tight at optimality:  {best_result.losslessness['is_tight']}
  Max relative slack:   {best_result.losslessness['max_relative_slack']:.2e}
""")
    
    return best_result, grid_results, sensitivity_results


def run_quick(verbose: bool = True):
    """Quick test run with single tf value."""
    config = DEFAULT_CONFIG
    
    if verbose:
        print("="*70)
        print("QUICK TEST: Powered Descent Guidance")
        print("="*70)
        print(config.summary())
    
    # Find a feasible tf first
    result = solve_fixed_tf(config, tf=25.0, verbose=False)
    
    if verbose:
        print(result.summary())
        
        if result.success:
            # Quick constraint check
            c = result.constraints
            print("Constraint satisfaction:")
            print(f"  Glideslope: {'OK' if c['glideslope']['satisfied'] else 'VIOLATED'}")
            print(f"  Pointing:   {'OK' if c['pointing']['satisfied'] else 'VIOLATED'}")
            print(f"  Thrust cone: {'OK' if c['thrust_cone']['satisfied'] else 'VIOLATED'}")
            print(f"  Terminal position error: {c['terminal']['position_error']:.4f} m")
            print(f"  Terminal velocity error: {c['terminal']['velocity_error']:.4f} m/s")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuel-Optimal Powered Descent Guidance')
    parser.add_argument('--quick', action='store_true', help='Quick test (single tf)')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    # SHOW POP-UP GRAPHICS
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not show plot pop-up windows (save only)")   
    parser.set_defaults(show=True)

    args = parser.parse_args()
    
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.quick:
        run_quick(verbose=not args.quiet)
    else:
        run_full(output_dir=args.output, verbose=not args.quiet)

    if args.show:
        plt.show()

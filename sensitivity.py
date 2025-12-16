"""
Sensitivity analysis for powered descent guidance.

Sweeps over key parameters to understand:
- Fuel vs glideslope angle
- Fuel/feasibility vs T_min (throttle limit)
- Fuel vs initial conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import replace
import os

from config import DescentConfig, DEFAULT_CONFIG
from solver import solve_fixed_tf, solve_grid_search


def sweep_glideslope(config: DescentConfig,
                     gamma_gs_range: Tuple[float, float] = (3.0, 20.0),
                     n_points: int = 15,
                     tf: float = 30.0,
                     verbose: bool = True) -> dict:
    """
    Sweep glideslope angle and measure fuel consumption.
    
    Smaller angles = more permissive (can fly more horizontally).
    Larger angles = more restrictive (must stay closer to vertical above target).
    
    Args:
        config: Base configuration
        gamma_gs_range: (min, max) glideslope angle from horizontal [deg]
        n_points: Number of sweep points
        tf: Fixed flight time [s]
        verbose: Print progress
        
    Returns:
        Dictionary with sweep results
    """
    gamma_values = np.linspace(gamma_gs_range[0], gamma_gs_range[1], n_points)
    
    results = {
        'gamma_gs_deg': [],
        'fuel_used': [],
        'success': [],
        'status': []
    }
    
    if verbose:
        print(f"Sweeping glideslope angle γ_gs ∈ [{gamma_gs_range[0]}, {gamma_gs_range[1]}]°")
    
    for gamma in gamma_values:
        # Create modified config
        cfg = DescentConfig(
            m0=config.m0, m_dry=config.m_dry,
            Isp=config.Isp, T_max=config.T_max, T_min=config.T_min,
            g0=config.g0,
            gamma_gs_deg=gamma, theta_max_deg=config.theta_max_deg,
            r0=config.r0, v0=config.v0,
            N=config.N
        )
        
        result = solve_fixed_tf(cfg, tf=tf)
        
        results['gamma_gs_deg'].append(gamma)
        results['fuel_used'].append(result.fuel_used if result.success else np.nan)
        results['success'].append(result.success)
        results['status'].append(result.status)
        
        if verbose:
            fuel_str = f"{result.fuel_used:.1f} kg" if result.success else "INFEASIBLE"
            print(f"  γ_gs = {gamma:.1f}°: {fuel_str}")
    
    # Convert to arrays
    results['gamma_gs_deg'] = np.array(results['gamma_gs_deg'])
    results['fuel_used'] = np.array(results['fuel_used'])
    results['success'] = np.array(results['success'])
    
    return results


def sweep_tmin(config: DescentConfig,
               tmin_range: Tuple[float, float] = (200000, 500000),
               n_points: int = 15,
               tf: float = 30.0,
               verbose: bool = True) -> dict:
    """
    Sweep minimum thrust and measure fuel consumption / feasibility.
    
    Higher T_min = less throttle authority = harder to land efficiently.
    This often creates a threshold effect where infeasibility jumps suddenly.
    
    Args:
        config: Base configuration
        tmin_range: (min, max) minimum thrust values [N]
        n_points: Number of sweep points
        tf: Fixed flight time [s]
        verbose: Print progress
        
    Returns:
        Dictionary with sweep results
    """
    tmin_values = np.linspace(tmin_range[0], tmin_range[1], n_points)
    
    results = {
        'T_min': [],
        'T_min_pct': [],  # As % of T_max
        'fuel_used': [],
        'success': [],
        'status': []
    }
    
    if verbose:
        print(f"Sweeping T_min ∈ [{tmin_range[0]/1000:.0f}, {tmin_range[1]/1000:.0f}] kN")
    
    for tmin in tmin_values:
        # Create modified config
        cfg = DescentConfig(
            m0=config.m0, m_dry=config.m_dry,
            Isp=config.Isp, T_max=config.T_max, T_min=tmin,
            g0=config.g0,
            gamma_gs_deg=config.gamma_gs_deg, theta_max_deg=config.theta_max_deg,
            r0=config.r0, v0=config.v0,
            N=config.N
        )
        
        result = solve_fixed_tf(cfg, tf=tf)
        
        results['T_min'].append(tmin)
        results['T_min_pct'].append(100 * tmin / config.T_max)
        results['fuel_used'].append(result.fuel_used if result.success else np.nan)
        results['success'].append(result.success)
        results['status'].append(result.status)
        
        if verbose:
            pct = 100 * tmin / config.T_max
            fuel_str = f"{result.fuel_used:.1f} kg" if result.success else "INFEASIBLE"
            print(f"  T_min = {tmin/1000:.0f} kN ({pct:.0f}%): {fuel_str}")
    
    # Convert to arrays
    results['T_min'] = np.array(results['T_min'])
    results['T_min_pct'] = np.array(results['T_min_pct'])
    results['fuel_used'] = np.array(results['fuel_used'])
    results['success'] = np.array(results['success'])
    
    return results


def sweep_initial_altitude(config: DescentConfig,
                           z0_range: Tuple[float, float] = (1000, 4000),
                           n_points: int = 15,
                           tf: Optional[float] = None,
                           verbose: bool = True) -> dict:
    """
    Sweep initial altitude.
    
    Args:
        config: Base configuration
        z0_range: (min, max) initial altitude [m]
        n_points: Number of sweep points
        tf: Fixed flight time (if None, use grid search)
        verbose: Print progress
        
    Returns:
        Dictionary with sweep results
    """
    z0_values = np.linspace(z0_range[0], z0_range[1], n_points)
    
    results = {
        'z0': [],
        'fuel_used': [],
        'tf_optimal': [],
        'success': [],
        'status': []
    }
    
    if verbose:
        mode = "fixed tf" if tf else "optimizing tf"
        print(f"Sweeping initial altitude z0 ∈ [{z0_range[0]}, {z0_range[1]}] m ({mode})")
    
    for z0 in z0_values:
        # Modify initial position
        r0_new = (config.r0[0], config.r0[1], z0)
        
        cfg = DescentConfig(
            m0=config.m0, m_dry=config.m_dry,
            Isp=config.Isp, T_max=config.T_max, T_min=config.T_min,
            g0=config.g0,
            gamma_gs_deg=config.gamma_gs_deg, theta_max_deg=config.theta_max_deg,
            r0=r0_new, v0=config.v0,
            N=config.N
        )
        
        if tf is not None:
            result = solve_fixed_tf(cfg, tf=tf)
            tf_used = tf
        else:
            # Scale tf range with altitude
            tf_scale = z0 / 2000  # 2000m is reference
            result, _ = solve_grid_search(
                cfg, 
                tf_min=15 * tf_scale, 
                tf_max=60 * tf_scale,
                n_grid=15, 
                verbose=False
            )
            tf_used = result.tf if result.success else np.nan
        
        results['z0'].append(z0)
        results['fuel_used'].append(result.fuel_used if result.success else np.nan)
        results['tf_optimal'].append(tf_used)
        results['success'].append(result.success)
        results['status'].append(result.status)
        
        if verbose:
            fuel_str = f"{result.fuel_used:.1f} kg" if result.success else "INFEASIBLE"
            print(f"  z0 = {z0:.0f}m: {fuel_str}")
    
    # Convert to arrays
    results['z0'] = np.array(results['z0'])
    results['fuel_used'] = np.array(results['fuel_used'])
    results['tf_optimal'] = np.array(results['tf_optimal'])
    results['success'] = np.array(results['success'])
    
    return results


def plot_sensitivity_glideslope(results: dict,
                                save_path: Optional[str] = None) -> plt.Figure:
    """Plot glideslope sensitivity results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gamma = results['gamma_gs_deg']
    fuel = results['fuel_used']
    success = results['success']
    
    if np.any(success):
        ax.plot(gamma[success], fuel[success], 'b.-', markersize=10, linewidth=1.5)
    
    if np.any(~success):
        # Place infeasible markers at bottom of plot
        y_infeas = np.nanmin(fuel) * 0.95 if np.any(success) else 1.0
        ax.scatter(gamma[~success], 
                  np.full(np.sum(~success), y_infeas),
                  color='red', marker='x', s=80, label='Infeasible')
    
    ax.set_xlabel('Glideslope angle γ_gs [deg from horizontal]')
    ax.set_ylabel('Fuel consumption [kg]')
    ax.set_title('Sensitivity: Fuel vs Glideslope Constraint')
    
    if np.any(success) or np.any(~success):
        ax.legend()
    
    # Add annotation only if we have feasible points
    if np.any(success) and np.sum(success) > 1:
        ax.annotate('More permissive\n(can fly horizontal)', 
                    xy=(gamma.min() + 1, fuel[success].max()),
                    fontsize=9, alpha=0.7)
        ax.annotate('More restrictive\n(must stay near vertical)', 
                    xy=(gamma.max() - 5, fuel[success].min()),
                    fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sensitivity_tmin(results: dict,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Plot T_min sensitivity results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tmin_pct = results['T_min_pct']
    fuel = results['fuel_used']
    success = results['success']
    
    if np.any(success):
        ax.plot(tmin_pct[success], fuel[success], 'b.-', markersize=10, linewidth=1.5)
    
    if np.any(~success):
        # Show infeasible region
        infeas_pct = tmin_pct[~success]
        if np.any(success):
            ax.axvspan(infeas_pct.min(), infeas_pct.max() + 2, 
                       alpha=0.2, color='red', label='Infeasible region')
            ax.scatter(infeas_pct, 
                      np.full(len(infeas_pct), np.nanmax(fuel) * 1.05),
                      color='red', marker='x', s=80)
        else:
            ax.scatter(infeas_pct, np.ones(len(infeas_pct)),
                      color='red', marker='x', s=80, label='Infeasible')
    
    ax.set_xlabel('Minimum thrust T_min [% of T_max]')
    ax.set_ylabel('Fuel consumption [kg]')
    ax.set_title('Sensitivity: Fuel vs Throttle Limit')
    
    if np.any(success) or np.any(~success):
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def run_all_sensitivity(config: DescentConfig,
                        tf: float,
                        output_dir: str = './figures',
                        verbose: bool = True) -> dict:
    """
    Run all sensitivity analyses and save figures.
    
    Args:
        config: Base configuration
        tf: Fixed flight time for sweeps
        output_dir: Directory to save figures
        verbose: Print progress
        
    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Glideslope sweep
    if verbose:
        print("\n" + "="*60)
        print("Glideslope Sensitivity")
        print("="*60)
    
    gs_results = sweep_glideslope(config, tf=tf, verbose=verbose)
    all_results['glideslope'] = gs_results
    
    fig = plot_sensitivity_glideslope(gs_results)
    fig.savefig(os.path.join(output_dir, 'sensitivity_glideslope.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # T_min sweep
    if verbose:
        print("\n" + "="*60)
        print("Throttle Limit Sensitivity")
        print("="*60)
    
    tmin_results = sweep_tmin(config, tf=tf, verbose=verbose)
    all_results['tmin'] = tmin_results
    
    fig = plot_sensitivity_tmin(tmin_results)
    fig.savefig(os.path.join(output_dir, 'sensitivity_tmin.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return all_results


if __name__ == "__main__":
    from solver import solve_grid_search
    
    config = DEFAULT_CONFIG
    
    print(config.summary())
    
    # First find the optimal tf
    print("Finding optimal tf...")
    best, _ = solve_grid_search(config, tf_min=22, tf_max=32, n_grid=15, verbose=False)
    
    if best.success:
        print(f"Using optimal tf = {best.tf:.1f}s")
        # Run sensitivity analysis with the correct tf
        results = run_all_sensitivity(config, tf=best.tf, verbose=True)
    else:
        print("ERROR: Could not find feasible solution")
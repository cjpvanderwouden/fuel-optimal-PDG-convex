"""
Visualization for powered descent trajectories.

Generates:
- 3D trajectory with glideslope cone and thrust vectors
- Time series plots (thrust, velocity, mass)
- Grid search fuel vs tf curve
- Sensitivity analysis plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, List
import os

from config import DescentConfig, DEFAULT_CONFIG


# Style settings
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_3d_trajectory(config: DescentConfig,
                       trajectory: dict,
                       show_glideslope: bool = True,
                       show_thrust: bool = True,
                       thrust_scale: float = 50.0,
                       n_thrust_vectors: int = 10,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D trajectory with glideslope cone and thrust vectors.
    
    Args:
        config: Problem configuration
        trajectory: Trajectory dictionary from solver
        show_glideslope: Overlay translucent glideslope cone
        show_thrust: Show thrust direction arrows
        thrust_scale: Scale factor for thrust arrows
        n_thrust_vectors: Number of thrust vectors to show
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    r = trajectory['r']
    u = trajectory['u']
    sigma = trajectory['sigma']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(r[:, 0], r[:, 1], r[:, 2], 'b-', linewidth=2, label='Trajectory')
    
    # Mark start and end
    ax.scatter(*r[0], color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(*r[-1], color='red', s=100, marker='*', label='Target', zorder=5)
    
    # Glideslope cone
    if show_glideslope:
        # Create cone surface: rz = tan(θ_gs) * sqrt(rx² + ry²)
        max_z = np.max(r[:, 2]) * 1.1
        max_r = max_z / config.tan_gs
        
        theta_cone = np.linspace(0, 2 * np.pi, 50)
        z_cone = np.linspace(0, max_z, 30)
        Theta, Z = np.meshgrid(theta_cone, z_cone)
        R = Z / config.tan_gs  # At height z, radius is z/tan(θ_gs)
        
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        ax.plot_surface(X, Y, Z, alpha=0.15, color='orange', 
                       label='Glideslope cone')
    
    # Thrust vectors
    if show_thrust and u is not None:
        N = len(sigma)
        indices = np.linspace(0, N - 1, n_thrust_vectors, dtype=int)
        
        for i in indices:
            # Position at this time (use state index)
            pos = r[i]
            
            # Thrust direction (normalized) scaled by magnitude
            thrust_dir = u[i] / (np.linalg.norm(u[i]) + 1e-10)
            arrow_len = thrust_scale * sigma[i] / np.max(sigma)
            
            ax.quiver(pos[0], pos[1], pos[2],
                     -thrust_dir[0] * arrow_len,
                     -thrust_dir[1] * arrow_len,
                     -thrust_dir[2] * arrow_len,
                     color='red', alpha=0.7, arrow_length_ratio=0.2)
    
    # Labels and formatting
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z (altitude) [m]')
    ax.set_title('Fuel-Optimal Powered Descent Trajectory')
    ax.legend(loc='upper left')
    
    # Equal aspect ratio (approximately)
    max_range = np.max([
        np.max(r[:, 0]) - np.min(r[:, 0]),
        np.max(r[:, 1]) - np.min(r[:, 1]),
        np.max(r[:, 2]) - np.min(r[:, 2])
    ]) / 2
    
    mid_x = (np.max(r[:, 0]) + np.min(r[:, 0])) / 2
    mid_y = (np.max(r[:, 1]) + np.min(r[:, 1])) / 2
    mid_z = (np.max(r[:, 2]) + np.min(r[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, max_z if show_glideslope else mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_time_series(config: DescentConfig,
                     trajectory: dict,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series: thrust magnitude, velocity, mass.
    
    Args:
        config: Problem configuration
        trajectory: Trajectory dictionary
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    t = trajectory['t']
    t_ctrl = trajectory['t_ctrl']
    r = trajectory['r']
    v = trajectory['v']
    m = trajectory['m']
    sigma = trajectory['sigma']
    u = trajectory['u']
    
    # Compute derived quantities
    speed = np.linalg.norm(v, axis=1)
    altitude = r[:, 2]
    u_norm = np.linalg.norm(u, axis=1)
    
    # Thrust in physical units
    m_ctrl = m[:-1]  # Mass at control times
    T_mag = sigma * m_ctrl / 1000  # Convert to kN
    
    # Bounds
    rho_min, rho_max = config.thrust_bounds_conservative()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Thrust magnitude ---
    ax = axes[0, 0]
    ax.step(t_ctrl, sigma, where='mid', label=r'$\sigma$ (accel)', linewidth=2)
    ax.axhline(rho_min, color='r', linestyle='--', alpha=0.7, label=r'$\rho_{min}$')
    ax.axhline(rho_max, color='r', linestyle='--', alpha=0.7, label=r'$\rho_{max}$')
    ax.fill_between(t_ctrl, rho_min, rho_max, alpha=0.1, color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Thrust acceleration $\sigma$ [m/s²]')
    ax.set_title('Thrust Profile')
    ax.legend()
    ax.set_xlim(0, t[-1])
    
    # --- Velocity ---
    ax = axes[0, 1]
    ax.plot(t, speed, 'b-', label='Speed ||v||', linewidth=2)
    ax.plot(t, v[:, 2], 'g--', label='$v_z$ (vertical)', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('Velocity Profile')
    ax.legend()
    ax.set_xlim(0, t[-1])
    
    # --- Mass ---
    ax = axes[1, 0]
    ax.plot(t, m / 1000, 'b-', linewidth=2)
    ax.axhline(config.m_dry / 1000, color='r', linestyle='--', 
               alpha=0.7, label='Dry mass')
    ax.fill_between(t, config.m_dry / 1000, m / 1000, alpha=0.2, color='blue')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Mass [tonnes]')
    ax.set_title(f'Mass Depletion (fuel used: {config.m0 - m[-1]:.0f} kg)')
    ax.legend()
    ax.set_xlim(0, t[-1])
    ax.set_ylim(config.m_dry / 1000 * 0.98, config.m0 / 1000 * 1.02)
    
    # --- Altitude ---
    ax = axes[1, 1]
    ax.plot(t, altitude, 'b-', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Altitude Profile')
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=-50)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_losslessness(trajectory: dict,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot verification that relaxation is tight (||u|| ≈ σ).
    
    Args:
        trajectory: Trajectory dictionary
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    t_ctrl = trajectory['t_ctrl']
    u = trajectory['u']
    sigma = trajectory['sigma']
    
    u_norm = np.linalg.norm(u, axis=1)
    slack = sigma - u_norm
    relative_slack = slack / sigma
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute comparison
    ax = axes[0]
    ax.plot(t_ctrl, sigma, 'b-', label=r'$\sigma$', linewidth=2)
    ax.plot(t_ctrl, u_norm, 'r--', label=r'$\|u\|$', linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Thrust acceleration [m/s²]')
    ax.set_title(r'Relaxation Tightness: $\|u\| \leq \sigma$')
    ax.legend()
    
    # Relative slack
    ax = axes[1]
    ax.semilogy(t_ctrl, relative_slack, 'b-', linewidth=2)
    ax.axhline(1e-4, color='r', linestyle='--', alpha=0.7, label='Tolerance')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Relative slack $(\sigma - \|u\|) / \sigma$')
    ax.set_title('Relaxation Slack (should be near machine precision)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_grid_search(grid_results: List[dict],
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot fuel consumption vs flight time from grid search.
    
    Args:
        grid_results: List of dicts with 'tf', 'fuel_used', 'success'
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    tf_vals = np.array([r['tf'] for r in grid_results])
    fuel_vals = np.array([r['fuel_used'] for r in grid_results])
    success = np.array([r['success'] for r in grid_results])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot feasible points
    ax.plot(tf_vals[success], fuel_vals[success], 'b.-', 
            markersize=8, linewidth=1.5, label='Feasible')
    
    # Mark infeasible
    if np.any(~success):
        ax.scatter(tf_vals[~success], 
                  np.full(np.sum(~success), np.min(fuel_vals[success]) * 0.95),
                  color='red', marker='x', s=50, label='Infeasible')
    
    # Mark optimal
    if np.any(success):
        opt_idx = np.argmin(fuel_vals[success])
        opt_tf = tf_vals[success][opt_idx]
        opt_fuel = fuel_vals[success][opt_idx]
        ax.scatter([opt_tf], [opt_fuel], color='green', marker='*', 
                  s=200, zorder=5, label=f'Optimal: tf={opt_tf:.1f}s')
    
    ax.set_xlabel('Flight time $t_f$ [s]')
    ax.set_ylabel('Fuel consumption [kg]')
    ax.set_title('Fuel vs Flight Time (Grid Search)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_2d_trajectory(config: DescentConfig,
                       trajectory: dict,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D projections of trajectory.
    
    Args:
        config: Problem configuration
        trajectory: Trajectory dictionary
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    r = trajectory['r']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # XZ plane (side view)
    ax = axes[0]
    ax.plot(r[:, 0], r[:, 2], 'b-', linewidth=2)
    ax.scatter(r[0, 0], r[0, 2], color='green', s=100, marker='o', 
               label='Start', zorder=5)
    ax.scatter(r[-1, 0], r[-1, 2], color='red', s=100, marker='*',
               label='Target', zorder=5)
    
    # Glideslope lines
    x_gs = np.linspace(-500, 2000, 100)
    z_gs = config.tan_gs * np.abs(x_gs)
    ax.fill_between(x_gs, 0, z_gs, alpha=0.1, color='orange', 
                    label='Glideslope region')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z (altitude) [m]')
    ax.set_title('Trajectory (XZ plane - side view)')
    ax.legend()
    ax.set_ylim(bottom=-50)
    ax.set_aspect('equal')
    
    # XY plane (top view)
    ax = axes[1]
    ax.plot(r[:, 0], r[:, 1], 'b-', linewidth=2)
    ax.scatter(r[0, 0], r[0, 1], color='green', s=100, marker='o',
               label='Start', zorder=5)
    ax.scatter(r[-1, 0], r[-1, 1], color='red', s=100, marker='*',
               label='Target', zorder=5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory (XY plane - top view)')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_all_figures(config: DescentConfig,
                       result,
                       grid_results: Optional[List[dict]] = None,
                       output_dir: str = './figures') -> List[str]:
    """
    Generate and save all figures.
    
    Args:
        config: Problem configuration
        result: SolveResult from solver
        grid_results: Optional grid search results
        output_dir: Directory to save figures
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    
    trajectory = result.trajectory
    
    # 3D trajectory
    fig = plot_3d_trajectory(config, trajectory)
    path = os.path.join(output_dir, 'trajectory_3d.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    
    # 2D projections
    fig = plot_2d_trajectory(config, trajectory)
    path = os.path.join(output_dir, 'trajectory_2d.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    
    # Time series
    fig = plot_time_series(config, trajectory)
    path = os.path.join(output_dir, 'time_series.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    
    # Losslessness verification
    fig = plot_losslessness(trajectory)
    path = os.path.join(output_dir, 'losslessness.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    
    # Grid search (if available)
    if grid_results is not None:
        fig = plot_grid_search(grid_results)
        path = os.path.join(output_dir, 'grid_search.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
    
    return saved


if __name__ == "__main__":
    # Test with dummy data
    from solver import solve_fixed_tf
    
    config = DEFAULT_CONFIG
    result = solve_fixed_tf(config, tf=25.1)
    
    print("success:", result.success)
    print("status:", getattr(result, "status", None))
    print("message:", getattr(result, "message", None))

    if result.success:

        fig = plot_2d_trajectory(config, result.trajectory)
        plt.show()

        fig = plot_3d_trajectory(config, result.trajectory)
        plt.show()
        
        fig = plot_time_series(config, result.trajectory)
        plt.show()


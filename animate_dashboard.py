"""
Mission Control Dashboard Animation
====================================

Real-time visualization of optimal powered descent showing:
- Rocket trajectory (2D side view)
- Thrust gauge with min/max bounds
- Fuel tank depletion
- Velocity profile
- Flight phase annotations

Demonstrates the bang-bang optimal control structure:
MAX thrust → MIN thrust → MAX thrust

Usage:
    python animate_dashboard.py
    python animate_dashboard.py --output dashboard.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import argparse
import os

from config import DescentConfig, DEFAULT_CONFIG


def load_trajectory(path: str = './output/trajectory.npz') -> dict:
    """Load saved trajectory data."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def draw_rocket_2d(ax, x, z, angle, thrust_level, scale=30):
    """Draw a 2D rocket with flame."""
    artists = []
    
    # Rocket body
    body_w, body_h = 0.25 * scale, 1.0 * scale
    nose_h = 0.35 * scale
    
    # Body points (centered at bottom)
    body = np.array([
        [-body_w/2, 0],
        [body_w/2, 0],
        [body_w/2, body_h],
        [-body_w/2, body_h],
    ])
    
    # Nose cone
    nose = np.array([
        [-body_w/2, body_h],
        [body_w/2, body_h],
        [0, body_h + nose_h],
    ])
    
    # Fins
    fin_l = np.array([
        [-body_w/2, 0],
        [-body_w/2 - 0.2*scale, -0.15*scale],
        [-body_w/2, 0.2*scale],
    ])
    fin_r = np.array([
        [body_w/2, 0],
        [body_w/2 + 0.2*scale, -0.15*scale],
        [body_w/2, 0.2*scale],
    ])
    
    # Flame
    flame_h = thrust_level * 0.6 * scale
    flame = np.array([
        [-body_w/4, 0],
        [body_w/4, 0],
        [0, -flame_h],
    ])
    
    # Rotation
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    
    def transform(pts):
        return pts @ R.T + np.array([x, z])
    
    # Draw flame
    if thrust_level > 0.1:
        for i, color in enumerate(['#FF4500', '#FF8C00', '#FFD700']):
            f_scaled = flame * (1 - i*0.2)
            poly = Polygon(transform(f_scaled), fc=color, ec='none', 
                          alpha=0.9-i*0.2, zorder=2)
            ax.add_patch(poly)
            artists.append(poly)
    
    # Draw body
    poly = Polygon(transform(body), fc='white', ec='black', lw=1.5, zorder=3)
    ax.add_patch(poly)
    artists.append(poly)
    
    # Draw nose
    poly = Polygon(transform(nose), fc='#CC0000', ec='black', lw=1.5, zorder=3)
    ax.add_patch(poly)
    artists.append(poly)
    
    # Draw fins
    for fin in [fin_l, fin_r]:
        poly = Polygon(transform(fin), fc='#404040', ec='black', lw=1, zorder=3)
        ax.add_patch(poly)
        artists.append(poly)
    
    return artists


def draw_gauge(ax, value, min_val, max_val, label, unit, 
               bounds=None, danger_zone=None):
    """
    Draw a vertical gauge with current value indicator.
    
    bounds: (lower_bound, upper_bound) to show as green region
    danger_zone: values above this are red
    """
    ax.clear()
    
    # Normalize value
    norm_val = (value - min_val) / (max_val - min_val)
    norm_val = np.clip(norm_val, 0, 1)
    
    # Background
    ax.add_patch(Rectangle((0.2, 0), 0.6, 1, fc='#E0E0E0', ec='black', lw=2))
    
    # Bounds region (if specified)
    if bounds:
        b_lo = (bounds[0] - min_val) / (max_val - min_val)
        b_hi = (bounds[1] - min_val) / (max_val - min_val)
        ax.add_patch(Rectangle((0.2, b_lo), 0.6, b_hi - b_lo, 
                               fc='#90EE90', ec='none', alpha=0.5))
        # Bound lines
        ax.axhline(b_lo, color='green', lw=2, linestyle='--', xmin=0.15, xmax=0.85)
        ax.axhline(b_hi, color='green', lw=2, linestyle='--', xmin=0.15, xmax=0.85)
    
    # Fill bar
    if danger_zone and value > danger_zone:
        bar_color = '#FF4444'
    elif bounds and (value < bounds[0] or value > bounds[1]):
        bar_color = '#FFaa00'
    else:
        bar_color = '#4CAF50'
    
    ax.add_patch(Rectangle((0.25, 0.02), 0.5, norm_val * 0.96, 
                           fc=bar_color, ec='none'))
    
    # Current value indicator
    ax.plot([0.1, 0.9], [norm_val, norm_val], 'k-', lw=3)
    ax.plot(0.5, norm_val, 'ko', markersize=10)
    
    # Labels
    ax.text(0.5, 1.08, label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(0.5, -0.08, f'{value:.1f} {unit}', ha='center', va='top', fontsize=10)
    
    # Min/max labels
    ax.text(0.05, 0, f'{min_val:.0f}', ha='right', va='center', fontsize=8)
    ax.text(0.05, 1, f'{max_val:.0f}', ha='right', va='center', fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.15)
    ax.axis('off')


def draw_fuel_tank(ax, fuel_fraction, fuel_used, max_fuel):
    """Draw a fuel tank gauge."""
    ax.clear()
    
    # Tank outline
    tank_w, tank_h = 0.6, 0.85
    x0 = (1 - tank_w) / 2
    
    # Tank body
    ax.add_patch(FancyBboxPatch((x0, 0.05), tank_w, tank_h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc='#E8E8E8', ec='black', lw=2))
    
    # Fuel level
    fuel_h = fuel_fraction * (tank_h - 0.04)
    if fuel_fraction > 0.01:
        ax.add_patch(Rectangle((x0 + 0.02, 0.07), tank_w - 0.04, fuel_h,
                               fc='#FF6B00', ec='none', alpha=0.8))
    
    # Percentage markers
    for pct in [0.25, 0.5, 0.75]:
        y = 0.07 + pct * (tank_h - 0.04)
        ax.plot([x0, x0 + 0.08], [y, y], 'k-', lw=1, alpha=0.5)
        ax.plot([x0 + tank_w - 0.08, x0 + tank_w], [y, y], 'k-', lw=1, alpha=0.5)
    
    # Labels
    ax.text(0.5, 0.97, 'FUEL', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.5, f'{fuel_fraction*100:.0f}%', ha='center', va='center', 
           fontsize=14, fontweight='bold')
    ax.text(0.5, -0.02, f'Used: {fuel_used:.0f} kg', ha='center', va='top', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.05)
    ax.axis('off')


def get_flight_phase(t, tf, sigma, rho_min, rho_max):
    """Determine current flight phase based on thrust level."""
    # Normalize thrust
    sigma_norm = (sigma - rho_min) / (rho_max - rho_min)
    
    if t < 0.1 * tf:
        return "IGNITION", "#FF4444"
    elif sigma_norm > 0.8:
        if t < 0.3 * tf:
            return "INITIAL BRAKING", "#FF8C00"
        else:
            return "TERMINAL BRAKING", "#FF4444"
    elif sigma_norm < 0.3:
        return "FUEL-OPTIMAL COAST", "#4CAF50"
    else:
        return "TRANSITION", "#FFD700"


def create_dashboard_animation(config: DescentConfig,
                               trajectory: dict,
                               output_path: str = './output/dashboard.gif',
                               fps: int = 20,
                               duration: float = 10.0):
    """Create mission control dashboard animation."""
    
    # Extract data
    t = trajectory['t']
    r = trajectory['r']
    v = trajectory['v']
    m = trajectory['m']
    sigma = trajectory['sigma']
    
    tf = t[-1]
    N = len(t) - 1
    n_frames = int(fps * duration)
    
    # Compute derived quantities
    speed = np.linalg.norm(v, axis=1)
    rho_min, rho_max = config.thrust_bounds_conservative()
    fuel_capacity = config.m0 - config.m_dry
    
    # Set up figure with grid layout
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1, 1], 
                          width_ratios=[2, 0.5, 0.5, 1.2],
                          hspace=0.3, wspace=0.3)
    
    # Main trajectory view
    ax_traj = fig.add_subplot(gs[:, 0])
    
    # Gauges
    ax_thrust = fig.add_subplot(gs[0, 1])
    ax_fuel = fig.add_subplot(gs[0, 2])
    
    # Velocity plot
    ax_vel = fig.add_subplot(gs[1, 1:3])
    
    # Altitude plot  
    ax_alt = fig.add_subplot(gs[2, 1:3])
    
    # Info panel
    ax_info = fig.add_subplot(gs[:, 3])
    
    # Trajectory limits
    x_min, x_max = -100, max(r[:, 0].max(), 200) + 150
    z_max = r[:, 2].max() * 1.15
    
    def animate(frame):
        # Current time index
        t_current = frame / n_frames * tf
        idx = min(int(t_current / tf * N), N - 1)
        
        # === TRAJECTORY VIEW ===
        ax_traj.clear()
        
        # Ground and pad
        ax_traj.axhline(0, color='#2E7D32', lw=4)
        ax_traj.fill_between([x_min, x_max], -80, 0, color='#1B5E20', alpha=0.3)
        ax_traj.add_patch(Rectangle((-30, -5), 60, 10, fc='gray', ec='black'))
        ax_traj.plot(0, 0, 'g*', markersize=25)
        
        # Trajectory trail
        ax_traj.plot(r[:idx+1, 0], r[:idx+1, 2], 'b-', lw=2, alpha=0.6)
        
        # Ghost trajectory (future, faded)
        ax_traj.plot(r[idx:, 0], r[idx:, 2], 'b--', lw=1, alpha=0.2)
        
        # Rocket
        x, z = r[idx, 0], r[idx, 2]
        vx, vz = v[idx, 0], v[idx, 2]
        angle = np.arctan2(-vx, max(1, -vz)) * 0.5  # Slight tilt based on velocity
        
        thrust_level = sigma[min(idx, N-1)] / rho_max if idx < N else 0
        draw_rocket_2d(ax_traj, x, z, angle, thrust_level, scale=45)
        
        # Velocity vector
        v_scale = 3
        ax_traj.arrow(x, z, vx * v_scale, vz * v_scale,
                     head_width=20, head_length=10, fc='blue', ec='blue', alpha=0.7)
        
        ax_traj.set_xlim(x_min, x_max)
        ax_traj.set_ylim(-80, z_max)
        ax_traj.set_xlabel('Downrange [m]', fontsize=11)
        ax_traj.set_ylabel('Altitude [m]', fontsize=11)
        ax_traj.set_title('Trajectory View', fontsize=12, fontweight='bold')
        ax_traj.set_aspect('equal')
        ax_traj.grid(True, alpha=0.3)
        
        # === THRUST GAUGE ===
        current_sigma = sigma[min(idx, N-1)] if idx < N else rho_min
        draw_gauge(ax_thrust, current_sigma, 0, rho_max * 1.1, 
                  'THRUST', 'm/s²',
                  bounds=(rho_min, rho_max))
        
        # === FUEL TANK ===
        current_m = m[idx]
        fuel_remaining = current_m - config.m_dry
        fuel_used = config.m0 - current_m
        fuel_fraction = fuel_remaining / fuel_capacity
        draw_fuel_tank(ax_fuel, fuel_fraction, fuel_used, fuel_capacity)
        
        # === VELOCITY PLOT ===
        ax_vel.clear()
        ax_vel.plot(t, speed, 'b-', lw=1, alpha=0.3, label='Speed')
        ax_vel.plot(t[:idx+1], speed[:idx+1], 'b-', lw=2)
        ax_vel.axhline(0, color='g', lw=1, alpha=0.5)
        ax_vel.scatter(t[idx], speed[idx], c='red', s=80, zorder=5)
        ax_vel.set_xlim(0, tf)
        ax_vel.set_ylim(0, speed.max() * 1.1)
        ax_vel.set_ylabel('Speed [m/s]', fontsize=9)
        ax_vel.set_title('Velocity', fontsize=10, fontweight='bold')
        ax_vel.grid(True, alpha=0.3)
        
        # === ALTITUDE PLOT ===
        ax_alt.clear()
        ax_alt.plot(t, r[:, 2], 'b-', lw=1, alpha=0.3)
        ax_alt.plot(t[:idx+1], r[:idx+1, 2], 'b-', lw=2)
        ax_alt.axhline(0, color='g', lw=2)
        ax_alt.scatter(t[idx], r[idx, 2], c='red', s=80, zorder=5)
        ax_alt.set_xlim(0, tf)
        ax_alt.set_ylim(-50, r[:, 2].max() * 1.1)
        ax_alt.set_xlabel('Time [s]', fontsize=9)
        ax_alt.set_ylabel('Altitude [m]', fontsize=9)
        ax_alt.set_title('Altitude', fontsize=10, fontweight='bold')
        ax_alt.grid(True, alpha=0.3)
        
        # === INFO PANEL ===
        ax_info.clear()
        ax_info.axis('off')
        
        # Flight phase
        phase, phase_color = get_flight_phase(t_current, tf, current_sigma, rho_min, rho_max)
        
        # Title
        ax_info.text(
    0.5, 1.02, "MISSION STATUS",
    transform=ax_info.transAxes,
    ha="center", va="bottom",
    fontsize=14, fontweight="bold",
    clip_on=False
    )
        
        # Phase indicator
        ax_info.add_patch(FancyBboxPatch((0.05, 0.82), 0.9, 0.12,
                                         boxstyle="round,pad=0.02",
                                         fc=phase_color, ec='black', lw=2, alpha=0.8))
        ax_info.text(0.5, 0.88, phase, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')
        
        # Time
        ax_info.text(0.5, 0.72, f'T = {t_current:.1f}s / {tf:.1f}s', 
                    ha='center', fontsize=12)
        
        # Progress bar
        progress = t_current / tf
        ax_info.add_patch(Rectangle((0.1, 0.65), 0.8, 0.04, fc='#E0E0E0', ec='black'))
        ax_info.add_patch(Rectangle((0.1, 0.65), 0.8 * progress, 0.04, fc='#2196F3', ec='none'))
        
        # Stats
        stats = [
            ('Altitude', f'{r[idx, 2]:.0f} m'),
            ('Speed', f'{speed[idx]:.1f} m/s'),
            ('Downrange', f'{r[idx, 0]:.0f} m'),
            ('', ''),
            ('Thrust', f'{current_sigma:.1f} m/s²'),
            ('Fuel Used', f'{fuel_used:.0f} kg'),
            ('Fuel Left', f'{fuel_remaining:.0f} kg'),
        ]
        
        y_pos = 0.55
        for label, value in stats:
            if label:
                ax_info.text(0.1, y_pos, label + ':', fontsize=10, va='center')
                ax_info.text(0.9, y_pos, value, fontsize=10, va='center', ha='right',
                           fontweight='bold')
            y_pos -= 0.065
        
        # Thrust profile explanation
        ax_info.text(0.5, 0.12, 'OPTIMAL CONTROL:', ha='center', fontsize=9, 
                    fontweight='bold', alpha=0.7)
        ax_info.text(0.5, 0.06, 'MAX → MIN → MAX thrust', ha='center', fontsize=9, 
                    alpha=0.7, style='italic')
        ax_info.text(0.5, 0.00, '(Bang-Bang Control)', ha='center', fontsize=8, 
                    alpha=0.5)
        
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        
        return []
    
    print(f"Generating {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames,
                        interval=1000/fps, blit=False)
    
    print(f"Saving to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    print(f"Done! Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mission control dashboard animation')
    parser.add_argument('--output', type=str, default='./output/dashboard.gif',
                       help='Output file path')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Animation duration (seconds)')
    parser.add_argument('--solve', action='store_true', help='Solve trajectory first')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG
    
    # Get trajectory
    if args.solve or not os.path.exists('./output/trajectory.npz'):
        print("Solving optimal trajectory...")
        from solver import solve_grid_search
        
        result, _ = solve_grid_search(config, tf_min=22, tf_max=32, n_grid=20, verbose=True)
        
        if not result.success:
            print("ERROR: Could not find feasible solution")
            exit(1)
        
        trajectory = result.trajectory
        
        # Save
        os.makedirs('./output', exist_ok=True)
        np.savez('./output/trajectory.npz', **{
            't': trajectory['t'],
            'r': trajectory['r'],
            'v': trajectory['v'],
            'm': trajectory['m'],
            'u': trajectory['u'],
            'sigma': trajectory['sigma']
        })
    else:
        print("Loading saved trajectory...")
        trajectory = load_trajectory('./output/trajectory.npz')
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    create_dashboard_animation(config, trajectory, args.output, args.fps, args.duration)
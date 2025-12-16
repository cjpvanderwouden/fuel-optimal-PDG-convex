"""
T_min Threshold Animation
=========================

Visualizes the critical relationship between minimum throttle and landing feasibility.
Shows 6-7 rockets with different T_min values - some land successfully, others crash.

This demonstrates why SpaceX Falcon 9 does "suicide burns" - with high minimum throttle,
you can't start slowing down early. Time it wrong and you crash.

Usage:
    python animate_tmin.py
    python animate_tmin.py --output throttle_demo.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
from matplotlib.collections import PatchCollection
import matplotlib.transforms as transforms
import argparse
import os

from config import DescentConfig, DEFAULT_CONFIG
from solver import solve_fixed_tf


# =============================================================================
# Rocket Drawing
# =============================================================================

def draw_rocket(ax, x, z, angle, thrust_level, scale=40, color='white'):
    """
    Draw a rocket at position (x, z) with given angle and thrust.
    
    Args:
        ax: Matplotlib axes
        x, z: Position
        angle: Rotation angle in radians (0 = pointing up)
        thrust_level: 0-1, controls flame size
        scale: Size multiplier
        color: Rocket body color
    """
    # Rocket body dimensions (in local coords, pointing up)
    body_width = 0.3 * scale
    body_height = 1.0 * scale
    nose_height = 0.4 * scale
    fin_width = 0.25 * scale
    fin_height = 0.3 * scale
    
    # Body rectangle
    body = np.array([
        [-body_width/2, 0],
        [body_width/2, 0],
        [body_width/2, body_height],
        [-body_width/2, body_height],
    ])
    
    # Nose cone (triangle)
    nose = np.array([
        [-body_width/2, body_height],
        [body_width/2, body_height],
        [0, body_height + nose_height],
    ])
    
    # Left fin
    fin_l = np.array([
        [-body_width/2, 0],
        [-body_width/2 - fin_width, -fin_height],
        [-body_width/2, fin_height],
    ])
    
    # Right fin
    fin_r = np.array([
        [body_width/2, 0],
        [body_width/2 + fin_width, -fin_height],
        [body_width/2, fin_height],
    ])
    
    # Flame (if thrusting)
    flame_height = thrust_level * 0.8 * scale
    flame = np.array([
        [-body_width/3, 0],
        [body_width/3, 0],
        [0, -flame_height],
    ])
    
    # Rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    
    def transform(pts):
        rotated = pts @ R.T
        return rotated + np.array([x, z])
    
    # Draw components
    artists = []
    
    # Flame first (behind rocket)
    if thrust_level > 0.05:
        flame_colors = ['#FF6B00', '#FF9500', '#FFCC00']
        for i, fc in enumerate(flame_colors):
            flame_scaled = flame * (1 - i*0.25)
            flame_poly = Polygon(transform(flame_scaled), 
                                facecolor=fc, edgecolor='none', 
                                alpha=0.8 - i*0.2, zorder=2)
            ax.add_patch(flame_poly)
            artists.append(flame_poly)
    
    # Body
    body_poly = Polygon(transform(body), facecolor=color, 
                       edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(body_poly)
    artists.append(body_poly)
    
    # Nose
    nose_poly = Polygon(transform(nose), facecolor='red', 
                       edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(nose_poly)
    artists.append(nose_poly)
    
    # Fins
    for fin in [fin_l, fin_r]:
        fin_poly = Polygon(transform(fin), facecolor='darkgray', 
                          edgecolor='black', linewidth=1, zorder=3)
        ax.add_patch(fin_poly)
        artists.append(fin_poly)
    
    return artists


def draw_explosion(ax, x, z, frame, scale=50):
    """Draw explosion effect."""
    artists = []
    
    # Expanding circles
    for i in range(3):
        radius = scale * (0.3 + frame * 0.1) * (1 + i * 0.5)
        alpha = max(0, 0.7 - frame * 0.05 - i * 0.15)
        colors = ['#FF4500', '#FF6B00', '#FFD700']
        circle = Circle((x, z), radius, facecolor=colors[i], 
                        edgecolor='none', alpha=alpha, zorder=4)
        ax.add_patch(circle)
        artists.append(circle)
    
    # Debris particles
    np.random.seed(42)  # Consistent debris
    n_debris = 8
    for j in range(n_debris):
        angle = 2 * np.pi * j / n_debris + 0.3
        dist = scale * 0.5 * frame * 0.15
        dx = x + dist * np.cos(angle)
        dz = z + dist * np.sin(angle) - 0.5 * 9.8 * (frame * 0.05)**2 * scale * 0.1
        if dz > 0:
            debris = Circle((dx, dz), scale * 0.08, facecolor='gray', 
                           alpha=0.6, zorder=5)
            ax.add_patch(debris)
            artists.append(debris)
    
    return artists


# =============================================================================
# Trajectory Simulation
# =============================================================================

def simulate_crash_trajectory(config: DescentConfig, tf: float, T_min: float) -> dict:
    """
    Simulate a trajectory for infeasible T_min.
    Uses a simple heuristic: max braking until fuel runs out, then ballistic.
    
    Returns trajectory dict with crash point marked.
    """
    dt = 0.1
    N = int(tf * 2 / dt)  # Extra time to show crash
    
    # State
    r = np.zeros((N + 1, 3))
    v = np.zeros((N + 1, 3))
    m = np.zeros(N + 1)
    thrust = np.zeros(N)
    
    # Initial conditions
    r[0] = np.array(config.r0)
    v[0] = np.array(config.v0)
    m[0] = config.m0
    
    crashed = False
    crash_idx = N
    
    for k in range(N):
        if crashed:
            # Continue with last values
            r[k + 1] = r[k]
            v[k + 1] = v[k]
            m[k + 1] = m[k]
            continue
            
        # Simple control: point thrust opposite to velocity (try to brake)
        speed = np.linalg.norm(v[k])
        if speed > 0.1:
            thrust_dir = -v[k] / speed
        else:
            thrust_dir = np.array([0, 0, 1])  # Point up
        
        # Apply thrust at T_min if we have fuel
        if m[k] > config.m_dry:
            T_mag = T_min
            thrust[k] = T_mag / m[k]
            
            # Ensure pointing constraint (mostly upward)
            if thrust_dir[2] < config.cos_point:
                thrust_dir = np.array([0, 0, 1])
            
            accel = thrust_dir * T_mag / m[k] + config.g
            m[k + 1] = m[k] - config.alpha * T_mag * dt
        else:
            # Out of fuel - ballistic
            thrust[k] = 0
            accel = config.g
            m[k + 1] = m[k]
        
        # Integrate
        v[k + 1] = v[k] + accel * dt
        r[k + 1] = r[k] + v[k] * dt + 0.5 * accel * dt**2
        
        # Check for ground collision
        if r[k + 1, 2] <= 0:
            r[k + 1, 2] = 0
            crashed = True
            crash_idx = k + 1
    
    # Check if this is actually a crash (high velocity at ground)
    impact_speed = np.linalg.norm(v[crash_idx]) if crashed else 0
    
    return {
        't': np.arange(N + 1) * dt,
        'r': r,
        'v': v,
        'm': m,
        'thrust': thrust,
        'crashed': crashed or impact_speed > 5,
        'crash_idx': crash_idx,
        'impact_speed': impact_speed
    }


def get_trajectory(config: DescentConfig, T_min: float, tf: float) -> dict:
    """
    Get trajectory for given T_min. Either solve optimally or simulate crash.
    """
    # Create config with this T_min
    cfg = DescentConfig(
        m0=config.m0, m_dry=config.m_dry,
        Isp=config.Isp, T_max=config.T_max, T_min=T_min,
        g0=config.g0,
        gamma_gs_deg=config.gamma_gs_deg, theta_max_deg=config.theta_max_deg,
        r0=config.r0, v0=config.v0,
        N=config.N
    )
    
    # Try to solve
    result = solve_fixed_tf(cfg, tf=tf)
    
    if result.success:
        traj = result.trajectory
        traj['crashed'] = False
        traj['crash_idx'] = None
        traj['impact_speed'] = np.linalg.norm(traj['v'][-1])
        traj['thrust'] = traj['sigma']
        return traj
    else:
        # Simulate crash
        return simulate_crash_trajectory(cfg, tf, T_min)


# =============================================================================
# Animation
# =============================================================================

def create_tmin_animation(output_path: str = './output/tmin_threshold.gif',
                          fps: int = 20,
                          duration: float = 10.0):
    """
    Create animation showing rockets with different T_min values.
    """
    config = DEFAULT_CONFIG
    
    # T_min scenarios (as fraction of T_max)
    # We want some successes and some crashes
    tmin_fractions = [0.20, 0.24, 0.28, 0.31, 0.35, 0.42, 0.50]
    tmin_values = [f * config.T_max for f in tmin_fractions]
    
    n_rockets = len(tmin_values)
    
    # Colors for each rocket
    colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800', '#FF5722', '#F44336']
    
    # Find optimal tf for the easiest case first
    print("Finding feasible flight time...")
    from solver import solve_grid_search
    best, _ = solve_grid_search(config, tf_min=22, tf_max=35, n_grid=15, verbose=False)
    
    if not best.success:
        print("ERROR: Couldn't find any feasible solution")
        return
    
    tf = best.tf
    print(f"Using tf = {tf:.1f}s")
    
    # Get trajectories for each T_min
    print("\nComputing trajectories:")
    trajectories = []
    for i, (tmin, frac) in enumerate(zip(tmin_values, tmin_fractions)):
        print(f"  T_min = {frac*100:.0f}%...", end=" ")
        traj = get_trajectory(config, tmin, tf)
        traj['tmin_frac'] = frac
        traj['color'] = colors[i]
        trajectories.append(traj)
        status = "CRASH" if traj['crashed'] else f"OK (fuel: {config.m0 - traj['m'][-1]:.0f}kg)"
        print(status)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Determine axis limits
    all_x = np.concatenate([t['r'][:, 0] for t in trajectories])
    all_z = np.concatenate([t['r'][:, 2] for t in trajectories])
    
    x_margin = 200
    x_min, x_max = min(all_x.min(), -100) - x_margin, max(all_x.max(), 100) + x_margin
    z_max = max(all_z.max(), config.r0[2]) * 1.1
    
    # Animation timing
    n_frames = int(fps * duration)
    max_t = max(t['t'][-1] for t in trajectories)
    
    # Pre-compute crash frames for explosion effects
    crash_frames = {}
    for i, traj in enumerate(trajectories):
        if traj['crashed']:
            crash_time = traj['t'][traj['crash_idx']]
            crash_frames[i] = int(crash_time / max_t * n_frames)
    
    def animate(frame):
        ax.clear()
        
        current_t = frame / n_frames * max_t
        
        # Ground
        ax.axhline(0, color='#2E7D32', linewidth=4, zorder=1)
        ax.fill_between([x_min, x_max], -100, 0, color='#1B5E20', alpha=0.3, zorder=0)
        
        # Landing pad
        pad_width = 60
        ax.fill_between([-pad_width/2, pad_width/2], -5, 5, color='gray', zorder=1)
        ax.plot(0, 0, 'g*', markersize=20, zorder=2)
        
        # Draw each rocket
        legend_entries = []
        
        for i, traj in enumerate(trajectories):
            t = traj['t']
            r = traj['r']
            thrust = traj['thrust']
            frac = traj['tmin_frac']
            color = traj['color']
            
            # Find current position (interpolate)
            idx = np.searchsorted(t, current_t)
            idx = min(idx, len(t) - 1)
            
            # Check if crashed
            is_crashed = traj['crashed'] and idx >= traj['crash_idx']
            
            if is_crashed:
                # Draw explosion
                frames_since_crash = frame - crash_frames.get(i, frame)
                if frames_since_crash < 20:
                    crash_x = r[traj['crash_idx'], 0]
                    draw_explosion(ax, crash_x, 0, frames_since_crash, scale=40)
                
                # Draw trajectory up to crash (faded)
                crash_idx = traj['crash_idx']
                ax.plot(r[:crash_idx, 0], r[:crash_idx, 2], '--', 
                       color=color, alpha=0.5, linewidth=1.5)
                
                status = f"T_min={frac*100:.0f}% - CRASH"
                legend_entries.append((color, status, True))
            else:
                # Draw trajectory so far
                ax.plot(r[:idx+1, 0], r[:idx+1, 2], '-', 
                       color=color, alpha=0.7, linewidth=2)
                
                # Get rocket state
                x, z = r[idx, 0], r[idx, 2]
                vx, vz = traj['v'][idx, 0], traj['v'][idx, 2]
                
                # Rocket angle (point opposite to velocity for braking)
                if idx < len(thrust):
                    thrust_level = thrust[idx] / (config.T_max / config.m_dry) if thrust[idx] > 0 else 0
                else:
                    thrust_level = 0
                
                # Angle: point thrust opposite to velocity
                angle = np.arctan2(-vx, -vz) if np.sqrt(vx**2 + vz**2) > 1 else 0
                
                # Draw rocket
                draw_rocket(ax, x, z, angle, min(thrust_level, 1.0), 
                           scale=35, color=color)
                
                status = f"T_min={frac*100:.0f}%"
                if not traj['crashed'] and idx == len(t) - 1:
                    status += " ✓"
                legend_entries.append((color, status, False))
        
        # Legend
        legend_y = z_max - 50
        for i, (color, label, crashed) in enumerate(legend_entries):
            style = 'italic' if crashed else 'normal'
            alpha = 0.5 if crashed else 1.0
            ax.plot(x_max - 350, legend_y - i * 80, 's', color=color, 
                   markersize=15, alpha=alpha)
            ax.text(x_max - 320, legend_y - i * 80, label, fontsize=11,
                   va='center', style=style, alpha=alpha)
        
        # Formatting
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-50, z_max)
        ax.set_xlabel('Downrange [m]', fontsize=12)
        ax.set_ylabel('Altitude [m]', fontsize=12)
        ax.set_title(f'Minimum Throttle Threshold Demo  |  t = {current_t:.1f}s\n'
                    f'Why can\'t rockets hover? Engine minimum thrust matters!', 
                    fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.text(x_min + 50, z_max - 50, 
               f'Flight time: {tf:.1f}s\nT_max: {config.T_max/1000:.0f} kN',
               fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return []
    
    print(f"\nGenerating {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, 
                        interval=1000/fps, blit=False)
    
    print(f"Saving to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    print(f"Done! Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T_min threshold animation')
    parser.add_argument('--output', type=str, default='./output/tmin_threshold.gif',
                       help='Output file path')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')
    parser.add_argument('--duration', type=float, default=10.0, 
                       help='Animation duration (seconds)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    create_tmin_animation(args.output, args.fps, args.duration)
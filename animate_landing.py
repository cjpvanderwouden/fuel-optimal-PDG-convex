"""
Animated visualization of fuel-optimal powered descent.

Generates an animated GIF/MP4 showing the rocket landing trajectory
with thrust vectors, glideslope cone, and real-time telemetry.

Usage:
    python animate.py                    # Generate animation from saved trajectory
    python animate.py --solve            # Solve first, then animate
    python animate.py --output rocket.gif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

from config import DescentConfig, DEFAULT_CONFIG


def load_trajectory(path: str = './output/trajectory.npz') -> dict:
    """Load saved trajectory data."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def create_animation(config: DescentConfig,
                     trajectory: dict,
                     output_path: str = './output/landing_animation.gif',
                     fps: int = 20,
                     duration: float = 8.0) -> None:
    """
    Create animated visualization of the landing.
    
    Args:
        config: Problem configuration
        trajectory: Dict with t, r, v, m, u, sigma arrays
        output_path: Where to save the animation
        fps: Frames per second
        duration: Animation duration in seconds
    """
    # Extract data
    t = trajectory['t']
    r = trajectory['r']
    v = trajectory['v']
    m = trajectory['m']
    sigma = trajectory['sigma']
    u = trajectory['u']
    
    tf = t[-1]
    N = len(t) - 1
    
    # Total frames
    n_frames = int(fps * duration)
    
    # Map animation frames to trajectory indices
    frame_to_idx = np.linspace(0, N, n_frames).astype(int)
    
    # Set up figure with subplots
    fig = plt.figure(figsize=(16, 9))
    
    # 3D trajectory plot (left side, larger)
    ax3d = fig.add_subplot(121, projection='3d')
    
    # Telemetry panel (right side)
    ax_thrust = fig.add_subplot(322)
    ax_velocity = fig.add_subplot(324)
    ax_altitude = fig.add_subplot(326)
    
    # =========================================================================
    # Set up 3D plot
    # =========================================================================
    max_z = np.max(r[:, 2]) * 1.2
    max_xy = max(np.max(np.abs(r[:, 0])), np.max(np.abs(r[:, 1]))) * 1.5
    
    # Glideslope cone (static)
    theta_cone = np.linspace(0, 2 * np.pi, 50)
    z_cone = np.linspace(0, max_z, 30)
    Theta, Z = np.meshgrid(theta_cone, z_cone)
    R_cone = Z / config.tan_gs
    X_cone = R_cone * np.cos(Theta)
    Y_cone = R_cone * np.sin(Theta)
    
    def init():
        ax3d.clear()
        ax3d.set_xlim(-max_xy, max_xy)
        ax3d.set_ylim(-max_xy, max_xy)
        ax3d.set_zlim(0, max_z)
        ax3d.set_xlabel('X [m]')
        ax3d.set_ylabel('Y [m]')
        ax3d.set_zlabel('Altitude [m]')
        ax3d.set_title('Fuel-Optimal Powered Descent')
        return []
    
    # Store plot elements
    trajectory_line = None
    rocket_point = None
    thrust_arrow = None
    
    def animate(frame):
        nonlocal trajectory_line, rocket_point, thrust_arrow
        
        idx = frame_to_idx[frame]
        current_t = t[min(idx, N)]
        
        # Clear and redraw 3D plot
        ax3d.clear()
        
        # Glideslope cone
        ax3d.plot_surface(X_cone, Y_cone, Z, alpha=0.1, color='orange')
        
        # Full trajectory (faded)
        ax3d.plot(r[:, 0], r[:, 1], r[:, 2], 'b-', alpha=0.3, linewidth=1)
        
        # Trajectory up to current point
        ax3d.plot(r[:idx+1, 0], r[:idx+1, 1], r[:idx+1, 2], 
                 'b-', linewidth=2, label='Trajectory')
        
        # Current rocket position
        ax3d.scatter(*r[idx], color='red', s=100, marker='o', zorder=5)
        
        # Thrust vector (pointing opposite to thrust direction)
        if idx < N:
            thrust_scale = 100 * sigma[idx] / np.max(sigma)
            thrust_dir = u[idx] / (np.linalg.norm(u[idx]) + 1e-10)
            ax3d.quiver(r[idx, 0], r[idx, 1], r[idx, 2],
                       -thrust_dir[0] * thrust_scale,
                       -thrust_dir[1] * thrust_scale,
                       -thrust_dir[2] * thrust_scale,
                       color='orange', linewidth=2, arrow_length_ratio=0.2)
        
        # Target
        ax3d.scatter(0, 0, 0, color='green', s=200, marker='*', label='Target')
        
        # Formatting
        ax3d.set_xlim(-max_xy, max_xy)
        ax3d.set_ylim(-max_xy, max_xy)
        ax3d.set_zlim(0, max_z)
        ax3d.set_xlabel('X [m]')
        ax3d.set_ylabel('Y [m]')
        ax3d.set_zlabel('Altitude [m]')
        ax3d.set_title(f'Powered Descent  |  t = {current_t:.1f}s / {tf:.1f}s')
        
        # =====================================================================
        # Telemetry plots
        # =====================================================================
        
        # Thrust
        ax_thrust.clear()
        ax_thrust.plot(t[:-1], sigma, 'b-', alpha=0.3)
        ax_thrust.plot(t[:idx], sigma[:idx] if idx > 0 else [], 'b-', linewidth=2)
        if idx < N:
            ax_thrust.scatter(t[idx], sigma[idx], color='red', s=50, zorder=5)
        ax_thrust.axhline(config.T_min / config.m0, color='r', linestyle='--', alpha=0.5)
        ax_thrust.set_ylabel('Thrust [m/s²]')
        ax_thrust.set_title('Thrust Profile')
        ax_thrust.set_xlim(0, tf)
        ax_thrust.grid(True, alpha=0.3)
        
        # Velocity
        speed = np.linalg.norm(v, axis=1)
        ax_velocity.clear()
        ax_velocity.plot(t, speed, 'b-', alpha=0.3)
        ax_velocity.plot(t[:idx+1], speed[:idx+1], 'b-', linewidth=2)
        ax_velocity.scatter(t[idx], speed[idx], color='red', s=50, zorder=5)
        ax_velocity.set_ylabel('Speed [m/s]')
        ax_velocity.set_title('Velocity')
        ax_velocity.set_xlim(0, tf)
        ax_velocity.grid(True, alpha=0.3)
        
        # Altitude
        ax_altitude.clear()
        ax_altitude.plot(t, r[:, 2], 'b-', alpha=0.3)
        ax_altitude.plot(t[:idx+1], r[:idx+1, 2], 'b-', linewidth=2)
        ax_altitude.scatter(t[idx], r[idx, 2], color='red', s=50, zorder=5)
        ax_altitude.axhline(0, color='g', linestyle='-', alpha=0.5)
        ax_altitude.set_xlabel('Time [s]')
        ax_altitude.set_ylabel('Altitude [m]')
        ax_altitude.set_title('Altitude')
        ax_altitude.set_xlim(0, tf)
        ax_altitude.grid(True, alpha=0.3)
        
        # Add telemetry text
        fuel_used = config.m0 - m[idx]
        info_text = f'Alt: {r[idx, 2]:.0f} m\nSpeed: {speed[idx]:.1f} m/s\nFuel: {fuel_used:.0f} kg'
        
        return []
    
    # Create animation
    print(f"Generating {n_frames} frames at {fps} fps...")
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=n_frames, interval=1000/fps, blit=False)
    
    # Save
    print(f"Saving to {output_path}...")
    
    if output_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    else:
        # MP4 requires ffmpeg
        anim.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    
    print(f"Animation saved: {output_path}")
    plt.close()


def create_simple_animation(config: DescentConfig,
                            trajectory: dict,
                            output_path: str = './output/landing_simple.gif',
                            fps: int = 15) -> None:
    """
    Create a simpler, faster animation (just 3D view).
    """
    t = trajectory['t']
    r = trajectory['r']
    sigma = trajectory['sigma']
    u = trajectory['u']
    
    tf = t[-1]
    N = len(t) - 1
    
    # Frames - one per timestep
    n_frames = N + 1
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    max_z = np.max(r[:, 2]) * 1.1
    max_xy = max(np.max(np.abs(r[:, 0])), np.max(np.abs(r[:, 1])), 200) * 1.3
    
    # Glideslope cone
    theta_cone = np.linspace(0, 2 * np.pi, 30)
    z_cone = np.linspace(0, max_z, 20)
    Theta, Z = np.meshgrid(theta_cone, z_cone)
    R_cone = Z / config.tan_gs
    X_cone = R_cone * np.cos(Theta)
    Y_cone = R_cone * np.sin(Theta)
    
    def animate(idx):
        ax.clear()
        
        # Cone
        ax.plot_surface(X_cone, Y_cone, Z, alpha=0.1, color='orange')
        
        # Trajectory so far
        ax.plot(r[:idx+1, 0], r[:idx+1, 1], r[:idx+1, 2], 'b-', linewidth=2)
        
        # Rocket
        ax.scatter(*r[idx], color='red', s=150, marker='^', zorder=5)
        
        # Thrust vector
        if idx < N:
            scale = 80 * sigma[idx] / np.max(sigma)
            thrust_dir = u[idx] / (np.linalg.norm(u[idx]) + 1e-10)
            ax.quiver(r[idx, 0], r[idx, 1], r[idx, 2],
                     -thrust_dir[0] * scale,
                     -thrust_dir[1] * scale,
                     -thrust_dir[2] * scale,
                     color='orange', linewidth=3, arrow_length_ratio=0.15)
        
        # Target
        ax.scatter(0, 0, 0, color='green', s=300, marker='*')
        
        # Formatting
        ax.set_xlim(-max_xy, max_xy)
        ax.set_ylim(-max_xy, max_xy)
        ax.set_zlim(0, max_z)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Altitude [m]')
        ax.set_title(f'Fuel-Optimal Landing  |  t = {t[idx]:.1f}s', fontsize=14)
        
        return []
    
    print(f"Generating {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=False)
    
    print(f"Saving to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print("Done!")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Animate powered descent trajectory')
    parser.add_argument('--solve', action='store_true', help='Solve problem first')
    parser.add_argument('--output', type=str, default='./output/landing.gif', 
                        help='Output file path')
    parser.add_argument('--simple', action='store_true', help='Simple animation (faster)')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG
    
    # Solve if needed
    if args.solve or not os.path.exists('./output/trajectory.npz'):
        print("Solving optimal trajectory...")
        from solver import solve_grid_search
        
        result, _ = solve_grid_search(config, tf_min=22, tf_max=32, n_grid=20, verbose=True)
        
        if not result.success:
            print("ERROR: Could not find feasible solution")
            exit(1)
        
        trajectory = result.trajectory
        
        # Save for future use
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
    
    # Create animation
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    if args.simple:
        create_simple_animation(config, trajectory, args.output, fps=args.fps)
    else:
        create_animation(config, trajectory, args.output, fps=args.fps)
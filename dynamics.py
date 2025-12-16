"""
Discrete-time dynamics for powered descent.

State vector: x = [r (3), v (3), z (1)] ∈ R^7
Control vector: w = [u (3), σ (1)] ∈ R^4

Dynamics:
    r_{k+1} = r_k + v_k * dt + 0.5 * (u_k + g) * dt^2
    v_{k+1} = v_k + (u_k + g) * dt
    z_{k+1} = z_k - α * σ_k * dt
"""

import numpy as np
from config import DescentConfig


def build_state_transition_matrices(config: DescentConfig, dt: float) -> tuple:
    """
    Build discrete state transition matrices for zero-order hold.
    
    x_{k+1} = A @ x_k + B @ w_k + c
    
    where:
        x = [r, v, z] ∈ R^7
        w = [u, σ] ∈ R^4
    
    Args:
        config: Problem configuration
        dt: Time step [s]
        
    Returns:
        A: State transition matrix (7 x 7)
        B: Control input matrix (7 x 4)  
        c: Constant term from gravity (7,)
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    
    # State transition matrix A (7 x 7)
    # [r]     [I   dt*I  0] [r]
    # [v]  =  [0   I     0] [v]  + control + gravity
    # [z]     [0   0     1] [z]
    A = np.block([
        [I3,      dt * I3,  np.zeros((3, 1))],
        [Z3,      I3,       np.zeros((3, 1))],
        [np.zeros((1, 3)), np.zeros((1, 3)), np.array([[1.0]])]
    ])
    
    # Control input matrix B (7 x 4)
    # Control w = [u (3), σ (1)]
    # r gets 0.5 * dt^2 * u
    # v gets dt * u
    # z gets -α * dt * σ
    B = np.block([
        [0.5 * dt**2 * I3,  np.zeros((3, 1))],
        [dt * I3,           np.zeros((3, 1))],
        [np.zeros((1, 3)),  np.array([[-config.alpha * dt]])]
    ])
    
    # Gravity constant term c (7,)
    # r gets 0.5 * dt^2 * g
    # v gets dt * g
    # z gets 0
    c = np.concatenate([
        0.5 * dt**2 * config.g,
        dt * config.g,
        np.array([0.0])
    ])
    
    return A, B, c


def propagate_trajectory(config: DescentConfig, 
                         u: np.ndarray, 
                         sigma: np.ndarray,
                         tf: float) -> dict:
    """
    Forward propagate trajectory given control sequence.
    
    Args:
        config: Problem configuration
        u: Thrust acceleration vectors (N, 3)
        sigma: Thrust magnitudes (N,)
        tf: Final time [s]
        
    Returns:
        Dictionary with trajectory arrays
    """
    N = len(sigma)
    dt = tf / N
    
    A, B, c = build_state_transition_matrices(config, dt)
    
    # Initialize state trajectory
    r = np.zeros((N + 1, 3))
    v = np.zeros((N + 1, 3))
    z = np.zeros(N + 1)
    
    # Initial conditions
    r[0] = config.r0_vec
    v[0] = config.v0_vec
    z[0] = config.z0
    
    # Propagate
    for k in range(N):
        x_k = np.concatenate([r[k], v[k], [z[k]]])
        w_k = np.concatenate([u[k], [sigma[k]]])
        
        x_next = A @ x_k + B @ w_k + c
        
        r[k + 1] = x_next[:3]
        v[k + 1] = x_next[3:6]
        z[k + 1] = x_next[6]
    
    # Convert log-mass to mass
    m = np.exp(z)
    
    # Time vector
    t = np.linspace(0, tf, N + 1)
    t_ctrl = t[:-1] + dt / 2  # Control times (midpoints)
    
    return {
        't': t,
        't_ctrl': t_ctrl,
        'r': r,
        'v': v,
        'z': z,
        'm': m,
        'u': u,
        'sigma': sigma,
        'dt': dt,
        'tf': tf
    }


def check_constraints(config: DescentConfig, traj: dict) -> dict:
    """
    Check constraint satisfaction for a trajectory.
    
    Args:
        config: Problem configuration
        traj: Trajectory dictionary from propagate_trajectory
        
    Returns:
        Dictionary with constraint violation info
    """
    r = traj['r']
    u = traj['u']
    sigma = traj['sigma']
    z = traj['z']
    
    violations = {}
    
    # Glideslope: rz >= tan(θ_gs) * ||(rx, ry)||
    r_horiz = np.linalg.norm(r[:, :2], axis=1)
    gs_slack = r[:, 2] - config.tan_gs * r_horiz
    violations['glideslope'] = {
        'satisfied': np.all(gs_slack >= -1e-6),
        'min_slack': np.min(gs_slack),
        'violations': np.sum(gs_slack < -1e-6)
    }
    
    # Pointing: uz >= cos(θ_max) * σ
    pointing_slack = u[:, 2] - config.cos_point * sigma
    violations['pointing'] = {
        'satisfied': np.all(pointing_slack >= -1e-6),
        'min_slack': np.min(pointing_slack),
        'violations': np.sum(pointing_slack < -1e-6)
    }
    
    # Thrust cone: ||u|| <= σ
    u_norm = np.linalg.norm(u, axis=1)
    cone_slack = sigma - u_norm
    violations['thrust_cone'] = {
        'satisfied': np.all(cone_slack >= -1e-6),
        'min_slack': np.min(cone_slack),
        'violations': np.sum(cone_slack < -1e-6)
    }
    
    # Mass bounds: z_dry <= z <= z0 
    mass_lower = z - config.z_dry
    mass_upper = config.z0 - z
    violations['mass'] = {
        'satisfied': np.all(mass_lower >= -1e-6) and np.all(mass_upper >= -1e-6),
        'min_mass': np.exp(np.min(z)),
        'max_mass': np.exp(np.max(z))
    }
    
    # Terminal conditions
    r_final = r[-1]
    v_final = traj['v'][-1]
    violations['terminal'] = {
        'position_error': np.linalg.norm(r_final),
        'velocity_error': np.linalg.norm(v_final)
    }
    
    return violations


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG
    dt = 0.5
    
    A, B, c = build_state_transition_matrices(config, dt)
    
    print("State transition matrix A:")
    print(A)
    print("\nControl matrix B:")
    print(B)
    print("\nGravity term c:")
    print(c)

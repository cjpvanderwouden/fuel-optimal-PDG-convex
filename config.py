"""
Problem parameters for fuel-optimal powered descent guidance.
Based on Falcon 9-inspired scenario from the technical writeup.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DescentConfig:
    """Configuration for powered descent problem."""
    
    # Mass properties
    m0: float = 25000.0          # Initial wet mass [kg]
    m_dry: float = 22000.0       # Dry mass [kg]
    
    # Propulsion
    Isp: float = 282.0           # Specific impulse [s]
    T_max: float = 845000.0      # Maximum thrust [N]
    T_min: float = 250000.0      # Minimum thrust [N] (~30% throttle)
    
    # Environment
    g0: float = 9.807            # Gravitational acceleration [m/s^2]
    
    # Constraints
    # Glideslope: angle from horizontal defining the cone the vehicle must stay inside
    # Small angle = permissive (can fly almost horizontal)
    # Large angle = restrictive (must stay close to vertical above target)
    gamma_gs_deg: float = 6.0    # Glideslope angle from horizontal [deg] (6° = permissive)
    theta_max_deg: float = 20.0  # Max thrust pointing angle from vertical [deg]
    
    # Initial conditions
    r0: tuple = (500.0, 100.0, 1500.0)     # Initial position [m]
    v0: tuple = (20.0, 5.0, -75.0)         # Initial velocity [m/s]
    
    # Discretization
    N: int = 50                  # Number of time steps
    
    def __post_init__(self):
        """Compute derived quantities."""
        # Exhaust velocity
        self.v_e = self.Isp * self.g0  # [m/s]
        
        # Mass depletion rate per unit thrust
        self.alpha = 1.0 / self.v_e    # [s/m]
        
        # Gravity vector (z-up convention)
        self.g = np.array([0.0, 0.0, -self.g0])
        
        # Convert angles to radians
        self.gamma_gs = np.deg2rad(self.gamma_gs_deg)
        self.theta_max = np.deg2rad(self.theta_max_deg)
        
        # Glideslope tangent (for constraint rz >= tan(gamma_gs) * ||(rx, ry)||)
        # Small gamma_gs = permissive glideslope (shallow cone, can fly more horizontally)
        self.tan_gs = np.tan(self.gamma_gs)
        
        # Pointing cosine (for constraint uz >= cos(theta_max) * sigma)
        self.cos_point = np.cos(self.theta_max)
        
        # Initial/final log-mass bounds
        self.z0 = np.log(self.m0)
        self.z_dry = np.log(self.m_dry)
        
        # Convert initial conditions to arrays
        self.r0_vec = np.array(self.r0)
        self.v0_vec = np.array(self.v0)
    
    @property
    def max_propellant(self) -> float:
        """Maximum available propellant [kg]."""
        return self.m0 - self.m_dry
    
    def thrust_bounds_conservative(self) -> tuple:
        """
        Conservative (fixed) thrust acceleration bounds.
        Uses worst-case mass values to ensure feasibility for all mass.
        
        Returns:
            (rho_min, rho_max): acceleration bounds [m/s^2]
        """
        rho_min = self.T_min / self.m0      # Min thrust at max mass
        rho_max = self.T_max / self.m_dry   # Max thrust at min mass
        return rho_min, rho_max
    
    def thrust_bounds_at_mass(self, m: float) -> tuple:
        """
        Exact thrust acceleration bounds at given mass.
        
        Args:
            m: Vehicle mass [kg]
            
        Returns:
            (rho_min, rho_max): acceleration bounds [m/s^2]
        """
        rho_min = self.T_min / m
        rho_max = self.T_max / m
        return rho_min, rho_max
    
    def summary(self) -> str:
        """Return a summary string of key parameters."""
        rho_min, rho_max = self.thrust_bounds_conservative()
        return f"""
Powered Descent Configuration
=============================
Mass:       m0 = {self.m0:.0f} kg, m_dry = {self.m_dry:.0f} kg
Propellant: {self.max_propellant:.0f} kg available
Thrust:     T_min = {self.T_min/1000:.0f} kN, T_max = {self.T_max/1000:.0f} kN
Isp:        {self.Isp:.0f} s (v_e = {self.v_e:.1f} m/s)

Accel bounds (conservative):
  rho_min = {rho_min:.2f} m/s^2
  rho_max = {rho_max:.2f} m/s^2

Constraints:
  Glideslope: {self.gamma_gs_deg:.1f}° from horizontal (tan = {self.tan_gs:.3f})
  Pointing:   {self.theta_max_deg:.1f}° max gimbal (cos = {self.cos_point:.3f})

Initial state:
  r0 = {self.r0} m
  v0 = {self.v0} m/s
"""


# Default configuration instance
DEFAULT_CONFIG = DescentConfig()


if __name__ == "__main__":
    config = DescentConfig()
    print(config.summary())

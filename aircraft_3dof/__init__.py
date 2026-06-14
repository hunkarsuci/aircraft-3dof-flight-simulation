"""
Core 3DOF Aircraft Flight Dynamics Simulation Package.
Provides modules for US Standard Atmosphere, compressible aerodynamics, jet propulsion models,
aircraft definitions, aircraft dynamics equations, controls, and simulation interfaces.
"""

from aircraft_3dof.atmosphere import StandardAtmosphere
from aircraft_3dof.aerodynamics import Aerodynamics
from aircraft_3dof.propulsion import Propulsion
from aircraft_3dof.aircraft import Aircraft, get_aircraft_presets
from aircraft_3dof.dynamics import flight_dynamics_equations
from aircraft_3dof.controls import Autopilot
from aircraft_3dof.simulation import FlightSimulator
from aircraft_3dof.animation import animate_trajectory

__all__ = [
    'StandardAtmosphere',
    'Aerodynamics',
    'Propulsion',
    'Aircraft',
    'get_aircraft_presets',
    'flight_dynamics_equations',
    'Autopilot',
    'FlightSimulator',
    'animate_trajectory'
]

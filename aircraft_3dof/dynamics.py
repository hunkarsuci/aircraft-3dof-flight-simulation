import numpy as np
from aircraft_3dof.aircraft import Aircraft
from aircraft_3dof.atmosphere import StandardAtmosphere

def flight_dynamics_equations(t: float, state: np.ndarray, aircraft: Aircraft,
                              throttle_cmd: float, cl_cmd: float, bank_cmd: float) -> np.ndarray:
    """
    Computes derivative vector (state rates) representing the Three-Degrees-of-Freedom (3DOF)
    Point-Mass aircraft flight equations of motion in standard Flight Path (wind) coordinates.

    State Vector elements:
        state[0] = x: Easting horizontal coordinate [m]
        state[1] = y: Northing horizontal coordinate [m]
        state[2] = h: Altitude above sea level [m]
        state[3] = V: True Airspeed (TAS) [m/s]
        state[4] = gamma: Flight-path angle [rad]
        state[5] = chi: Velocity heading angle [rad]
        state[6] = m: Integrated current aircraft mass [kg]

    Input Controllers:
        throttle_cmd (float): Throttle command [0.0, 1.0]
        cl_cmd (float): Commanded lift coefficient CL
        bank_cmd (float): Flight-path bank angle mu [rad]

    Returns:
        np.ndarray: [dx/dt, dy/dt, dh/dt, dV/dt, dgamma/dt, dchi/dt, dm/dt]
    """
    x, y, h, V, gamma, chi, m = state
    g0 = 9.80665

    # 1. Structural Crash & Operational Boundaries Safeties Check
    if h <= 0.0:
        # Aircraft is on the ground; velocity falls to zero, rates freeze to prevent solver tunneling
        return np.zeros_like(state)

    if m <= aircraft.empty_mass:
        # Engines flameout due to zero fuel, absolute zero throttle forced
        throttle_cmd = 0.0
        m = aircraft.empty_mass

    # 2. Physics & Engine Force Model Updates
    thrust, fuel_flow = aircraft.prop.compute_thrust_and_fuel_flow(h, V, throttle_cmd)

    # 3. Aerodynamics Force Calculations with Dynamic Stall Limiters
    # Apply stall limits dynamically to commands
    cl_clamped = max(aircraft.cl_min, min(aircraft.cl_max, cl_cmd))
    lift, drag = aircraft.aero.compute_forces(h, V, cl_clamped)

    # Apply structural G limits: n = lift / (m * g)
    n = lift / (m * g0)
    if n > aircraft.n_lim_max:
        # Clamps lift force to respect structural breaking strength (G-limits protection)
        lift = aircraft.n_lim_max * m * g0
        # Recalculate induced drag using inverted polar details
        # CD = Cd0 + K * CL^2 => CD = Cd0 + K * ( n * m * g0 / (q * S) )^2
        q = StandardAtmosphere.dynamic_pressure(h, V)
        cl_clamped = lift / (q * aircraft.S)
        mach = StandardAtmosphere.mach_number(h, V)
        drag = q * aircraft.S * (
            aircraft.aero.get_cd0(mach) + aircraft.aero.get_k(mach) * cl_clamped**2
        )
    elif n < aircraft.n_lim_min:
        lift = aircraft.n_lim_min * m * g0
        q = StandardAtmosphere.dynamic_pressure(h, V)
        cl_clamped = lift / (q * aircraft.S)
        mach = StandardAtmosphere.mach_number(h, V)
        drag = q * aircraft.S * (
            aircraft.aero.get_cd0(mach) + aircraft.aero.get_k(mach) * cl_clamped**2
        )

    # 4. Point-Mass 3DOF ODE Evaluator
    # Dynamic singularity avoidance: ensure V is non-zero to prevent division by zero in angular rates
    V_safe = max(5.0, V)

    # State derivatives formulation:
    dx_dt = V * np.cos(gamma) * np.sin(chi)
    dy_dt = V * np.cos(gamma) * np.cos(chi)
    dh_dt = V * np.sin(gamma)

    # Acceleration: dV/dt = (T - D) / m - g * sin(gamma)
    dV_dt = (thrust - drag) / m - g0 * np.sin(gamma)

    # Flight-path angle rate: dgamma/dt = (L * cos(mu) - m * g * cos(gamma)) / (m * V)
    dgamma_dt = (lift * np.cos(bank_cmd) - m * g0 * np.cos(gamma)) / (m * V_safe)

    # Heading angle rate: dchi/dt = (L * sin(mu)) / (m * V * cos(gamma))
    # Coordinated singularity mitigation around vertical flight angles (gamma = +-90 deg => cos(gamma) = 0)
    cos_gamma_safe = np.cos(gamma)
    if np.abs(cos_gamma_safe) < 1e-4:
        cos_gamma_safe = 1e-4 if cos_gamma_safe >= 0.0 else -1e-4

    dchi_dt = (lift * np.sin(bank_cmd)) / (m * V_safe * cos_gamma_safe)

    # Fuel mass consumption rate: dm/dt = - FuelFlow
    dm_dt = -fuel_flow

    return np.array([
        dx_dt,
        dy_dt,
        dh_dt,
        dV_dt,
        dgamma_dt,
        dchi_dt,
        dm_dt
    ])

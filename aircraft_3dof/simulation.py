import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from aircraft_3dof.aircraft import Aircraft
from aircraft_3dof.dynamics import flight_dynamics_equations
from aircraft_3dof.controls import Autopilot

class FlightSimulator:
    """
    Simulation Executive Manager integrating the 3DOF point-mass equations of motion
    using Scipy solve_ivp Runge-Kutta solver over complex flight profiles.
    """
    def __init__(self, aircraft: Aircraft, autopilot: Autopilot):
        self.aircraft = aircraft
        self.ap = autopilot

    def run_simulation(self, t_end: float, initial_state: np.ndarray,
                       v_cmd: float, h_cmd: float, chi_cmd: float,
                       step_size: float = 0.1) -> pd.DataFrame:
        """
        Executes standard fixed-step numerical integration using scipy.integrate.solve_ivp.

        Initial State Structure:
            state = [x, y, h, V, gamma, chi, m]

        Returns:
            pd.DataFrame: Simulation history logs
        """
        self.ap.reset()

        # Setup output history list
        history = []

        # Current state values
        current_state = np.array(initial_state, dtype=float)
        t = 0.0

        # Iteration loop for simulated guidance steps
        while t < t_end:
            # Check terminal bounds (ground crash safety)
            if current_state[2] <= 0.0:
                current_state[2] = 0.0 # Force soft landing level
                history.append(self._create_log_entry(t, current_state, 0.0, 0.0, 0.0))
                break

            # 1. Sample Guidance autopilot commands
            throttle, cl_cmd, bank_cmd = self.ap.compute_controls(
                t=t,
                state=current_state,
                aircraft=self.aircraft,
                v_cmd=v_cmd,
                h_cmd=h_cmd,
                chi_cmd=chi_cmd,
                dt=step_size
            )

            # 2. Define standard ODE derivative adapter holding control inputs constant over step dt
            def ode_rhs_adapter(t_eval, state_eval):
                return flight_dynamics_equations(
                    t=t_eval,
                    state=state_eval,
                    aircraft=self.aircraft,
                    throttle_cmd=throttle,
                    cl_cmd=cl_cmd,
                    bank_cmd=bank_cmd
                )

            # 3. Integrate step ahead by dt using RK45 method
            sol = solve_ivp(
                fun=ode_rhs_adapter,
                t_span=(t, t + step_size),
                y0=current_state,
                method='RK45',
                rtol=1e-6,
                atol=1e-6
            )

            # Log current step updates
            history.append(self._create_log_entry(t, current_state, throttle, cl_cmd, bank_cmd))

            # Progress state
            current_state = sol.y[:, -1]
            t += step_size

        # Convert array history into a tabular clean Pandas DataFrame
        dfCols = ['time', 'x', 'y', 'altitude', 'velocity', 'gamma', 'chi', 'mass',
                  'throttle', 'CL', 'bank_angle', 'mach', 'g_load', 'fuel_burn_rate']
        return pd.DataFrame(history, columns=dfCols)

    def _create_log_entry(self, t: float, state: np.ndarray,
                          throttle: float, cl: float, bank: float) -> list:
        """
        Utility logging adapter extracting standard derived aerospace variables.
        """
        x, y, h, V, gamma, chi, m = state
        from aircraft_3dof.atmosphere import StandardAtmosphere

        mach = StandardAtmosphere.mach_number(h, V)
        q = StandardAtmosphere.dynamic_pressure(h, V)
        lift = q * self.aircraft.S * cl
        g_load = lift / (m * 9.80665)

        # Calculate instantaneous fuel burn rate
        _, fuel_flow = self.aircraft.prop.compute_thrust_and_fuel_flow(h, V, throttle)

        return [
            t, x, y, h, V, gamma, chi, m,
            throttle, cl, bank, mach, g_load, fuel_flow
        ]

    def run_waypoint_mission(self, waypoints: list, initial_state: np.ndarray,
                             v_cmd: float, step_size: float = 0.1) -> pd.DataFrame:
        """
        Executes a 3D spatial trajectory path tracking guidance simulation, transitioning
        dynamic waypoints sequentially when entering custom target radius boxes.

        waypoints (list): list of dict milestones like [{'x': 100, 'y': 200, 'h': 500}, ...]
        """
        self.ap.reset()
        history = []
        current_state = np.array(initial_state, dtype=float)
        t = 0.0
        wp_idx = 0
        target_radius = 250.0 # [m] switch distance bounding box

        # Loop flight simulation steps
        max_t = 2000.0 # Safety timeout limit

        while t < max_t:
            if wp_idx >= len(waypoints):
                # Completed all guidance milestones
                break

            x, y, h, V, gamma, chi, m = current_state

            # Ground safety collision catch
            if h <= 0.0:
                current_state[2] = 0.0
                history.append(self._create_log_entry(t, current_state, 0.0, 0.0, 0.0))
                break

            # Current milestone targets
            target_wp = waypoints[wp_idx]
            tx, ty, th = target_wp['x'], target_wp['y'], target_wp['h']

            # Compute horizontal distance error bounds
            dist_to_wp = np.sqrt((tx - x)**2 + (ty - y)**2)
            if dist_to_wp < target_radius:
                # Transition to next target
                wp_idx += 1
                if wp_idx >= len(waypoints):
                    break
                target_wp = waypoints[wp_idx]
                tx, ty, th = target_wp['x'], target_wp['y'], target_wp['h']

            # Target course heading steering: chi is measured clockwise from North.
            chi_cmd = np.arctan2(tx - x, ty - y)

            # Sample controls tracking targets
            throttle, cl_cmd, bank_cmd = self.ap.compute_controls(
                t=t,
                state=current_state,
                aircraft=self.aircraft,
                v_cmd=v_cmd,
                h_cmd=th,
                chi_cmd=chi_cmd,
                dt=step_size
            )

            # Integrate step adapter
            def ode_adapter(t_eval, state_eval):
                return flight_dynamics_equations(
                    t=t_eval,
                    state=state_eval,
                    aircraft=self.aircraft,
                    throttle_cmd=throttle,
                    cl_cmd=cl_cmd,
                    bank_cmd=bank_cmd
                )

            sol = solve_ivp(
                fun=ode_adapter,
                t_span=(t, t + step_size),
                y0=current_state,
                method='RK45'
            )

            # Log
            history.append(self._create_log_entry(t, current_state, throttle, cl_cmd, bank_cmd))

            # Transition state
            current_state = sol.y[:, -1]
            t += step_size

        dfCols = ['time', 'x', 'y', 'altitude', 'velocity', 'gamma', 'chi', 'mass',
                  'throttle', 'CL', 'bank_angle', 'mach', 'g_load', 'fuel_burn_rate']
        return pd.DataFrame(history, columns=dfCols)

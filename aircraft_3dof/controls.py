import numpy as np
from aircraft_3dof.atmosphere import StandardAtmosphere

class Autopilot:
    """
    Closed-loop Guidance and Control Law System (Autopilot) for a 3DOF Point-Mass Flight Model.
    Includes:
      1. Speed Hold / Auto-Throttle (PI logic with anti-windup)
      2. Altitude Hold / Flight-Path Control (Dynamic inversion pitch steering)
      3. Heading Hold / Waypoint Guidance (Coordinated turn roll steering)
    """
    def __init__(self, kp_v: float = 0.05, ki_v: float = 0.005,
                 kp_gamma: float = 0.15, max_g_load: float = 4.5):
        """
        Parameters:
            kp_v (float): Speed-hold proportional gain
            ki_v (float): Speed-hold integral gain
            kp_gamma (float): Pitch flight-path tracking proportional gain
            max_g_load (float): Target maneuvering limit during climbs/turns
        """
        self.kp_v = kp_v
        self.ki_v = ki_v
        self.kp_gamma = kp_gamma
        self.max_g_limit = max_g_load

        # Anti-windup tracking states
        self.v_error_integral = 0.0

    def reset(self):
        """
        Resets autopilot integral states.
        """
        self.v_error_integral = 0.0

    def compute_controls(self, t: float, state: np.ndarray, aircraft,
                         v_cmd: float, h_cmd: float, chi_cmd: float, dt: float) -> tuple:
        """
        Computes the target control vector [throttle, CL, bank_angle] to track guidance path.

        State Vector:
            state = [x, y, h, V, gamma, chi, m]
        """
        x, y, h, V, gamma, chi, m = state
        g0 = 9.80665

        # ----------------------------------------------------
        # 1. Auto-Throttle / Speed Hold Loop (PI Control)
        # ----------------------------------------------------
        v_error = v_cmd - V
        self.v_error_integral += v_error * dt
        # Clamp integral bounds (anti-windup protection)
        self.v_error_integral = max(-10.0, min(10.0, self.v_error_integral))

        # PI control value
        throttle_raw = self.kp_v * v_error + self.ki_v * self.v_error_integral
        # Scale to standard normalized engine bounds
        throttle = max(0.0, min(1.0, throttle_raw))

        # ----------------------------------------------------
        # 2. Pitch / Flight-Path / Altitude Altitude Hold Loop
        # ----------------------------------------------------
        # Guide flight path angle based on altitude error: gamma_cmd = K * altitude_error
        h_error = h_cmd - h

        # Max safe climb slope: limit flight path target to standard safe values [-20 deg, 30 deg]
        gamma_limit = np.radians(30.0) if h_error > 0 else np.radians(-20.0)
        gamma_cmd_target = 0.02 * (h_error / 100.0)  # Proportional drift to track vertical target
        gamma_cmd = max(-np.radians(20.0), min(gamma_limit, gamma_cmd_target))

        # Back out required pitch rate to match commanded flight path angle
        gamma_error = gamma_cmd - gamma
        # Pitch rate command: elevator/pitch axis rate
        gamma_dot_cmd = self.kp_gamma * gamma_error

        # Leverage flight path EOM pitch-axis inversion:
        # L = m * V * gamma_dot + m * g * cos(gamma)
        q = StandardAtmosphere.dynamic_pressure(h, V)
        q = max(100.0, q) # Mitigate divide by zero dynamic pressure during stalls

        # ----------------------------------------------------
        # 3. Roll / Turn / Heading Hold Loop
        # ----------------------------------------------------
        # Calculate heading error with circular wraparound correction
        chi_error = chi_cmd - chi
        chi_error = (chi_error + np.pi) % (2 * np.pi) - np.pi # Norm to [-pi, pi]

        # Commanded turn rate: chi_dot_cmd = K * heading_error
        chi_dot_cmd = 0.15 * chi_error

        # Under coordinated turn flight mechanics assumptions:
        # Tan(bank_angle) = V * chi_dot / g
        tan_bank = (V * chi_dot_cmd) / g0
        bank_cmd = np.arctan(tan_bank)

        # Limit banking angle for structural & pilot safety (standard 45 deg limit)
        max_bank = np.radians(45.0)
        bank_cmd = max(-max_bank, min(max_bank, bank_cmd))

        # Coordinated load correction (must lift more during banks to maintain vertical path).
        bank_cos = max(0.2, np.cos(bank_cmd))
        lift_req = m * (V * gamma_dot_cmd + g0 * np.cos(gamma)) / bank_cos
        n_req = lift_req / (m * g0)
        n_req = max(aircraft.n_lim_min, min(self.max_g_limit, aircraft.n_lim_max, n_req))
        lift_req = n_req * m * g0

        # Convert requested force back to commanded lift coefficient: CL = L / (q * S)
        cl_cmd = lift_req / (q * aircraft.S)

        # Standard aerodynamic wing limits protection
        cl_cmd = max(aircraft.cl_min, min(aircraft.cl_max, cl_cmd))

        return throttle, cl_cmd, bank_cmd

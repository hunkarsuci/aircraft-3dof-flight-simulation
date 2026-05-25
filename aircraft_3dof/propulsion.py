import numpy as np
from aircraft_3dof.atmosphere import StandardAtmosphere

class Propulsion:
    """
    Turbofan/Jet Engine Propulsion Model.
    Computes ambient-density and Mach-dependent maximum thrust limits:
    T_max(h, M) = T_max0 * (rho(h)/rho0)^density_exp * (1 + mach_recovery * M^mach_exp)
    And models fuel mass flow via Specific Fuel Consumption (SFC).
    """
    def __init__(self, max_thrust_sl: float, sfc_sl: float, density_exp: float,
                 mach_recovery: float, mach_exp: float, sfc_mach_lapse: float):
        """
        Parameters:
            max_thrust_sl (float): Sea-level static maximum thrust [N]
            sfc_sl (float): Sea-level static SFC [kg/(N*s)]
            density_exp (float): Exponent for thrust density lapse (rho/rho0)^n
            mach_recovery (float): Dynamic ram recovery scaling parameter
            mach_exp (float): Power exponent for Mach recovery scaling
            sfc_mach_lapse (float): Increase in SFC with Mach number (d)
        """
        self.T_max0 = max_thrust_sl
        self.sfc0 = sfc_sl
        self.n_density = density_exp
        self.mach_recovery = mach_recovery
        self.mach_exp = mach_exp
        self.sfc_mach_lapse = sfc_mach_lapse

    def get_max_thrust(self, h: float, M: float) -> float:
        """
        Computes max thrust limits at altitude h and Mach number M.
        """
        atoms = StandardAtmosphere.get_properties(h)
        density_ratio = atoms['density'] / StandardAtmosphere.rho0

        # Max thrust scale combining atmospheric density falloff & Mach air inhalation compression
        thrust_density_lapse = (density_ratio) ** self.n_density
        mach_thrust_recovery = 1.0 + self.mach_recovery * (M ** self.mach_exp)

        return self.T_max0 * thrust_density_lapse * mach_thrust_recovery

    def get_sfc(self, M: float) -> float:
        """
        Computes Specific Fuel Consumption [kg/(N*s)] as a function of Mach speed.
        """
        return self.sfc0 * (1.0 + self.sfc_mach_lapse * M)

    def compute_thrust_and_fuel_flow(self, h: float, V: float, throttle: float) -> tuple:
        """
        Computes engine thrust output and corresponding fuel mass burn rate.

        Parameters:
            h (float): Altitude [m]
            V (float): Airspeed [m/s]
            throttle (float): Action command throttle efficiency in [0.0, 1.0]

        Returns:
            tuple: (Thrust [N], Fuel Flow Rate [kg/s])
        """
        # Normal throttle bounds constraint
        throttle = max(0.0, min(1.0, float(throttle)))

        M = StandardAtmosphere.mach_number(h, V)
        max_thrust = self.get_max_thrust(h, M)

        # Realized thrust force
        thrust = throttle * max_thrust

        # SFC unit check: SFC here is [kg/(N*s)].
        # Fuel Flow rate: dm/dt = - SFC * Thrust
        sfc = self.get_sfc(M)
        fuel_flow = sfc * thrust

        return thrust, fuel_flow

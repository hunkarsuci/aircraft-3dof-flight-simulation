import numpy as np

class StandardAtmosphere:
    """
    US Standard Atmosphere 1976 model implementation.
    Valid up to 20,000m.
    Computes altitude-dependent pressure, temperature, density, and speed of sound.
    """
    # Universal Physical Constants (Metric System)
    g0 = 9.80665      # Sea-level acceleration of gravity [m/s^2]
    R = 287.05287     # Specific gas constant for air [J/(kg*K)]
    gamma = 1.4       # Specific heat ratio for air
    T0 = 288.15       # Sea-level standard temperature [K]
    P0 = 101325.0     # Sea-level standard pressure [Pa]
    rho0 = 1.225      # Sea-level standard density [kg/m^3]

    # Stratosphere Boundary Constants (Lapse rules change at 11,000m)
    h_tropopause = 11000.0  # [m]
    T_tropopause = 216.65   # Temperature in stratosphere isothermal region [K]
    lambda_trop = 0.0065    # Troposphere lapse rate [K/m]

    @classmethod
    def get_properties(cls, h: float):
        """
        Computes atmospheric properties for a given geopotential altitude.

        Parameters:
            h (float): Altitude above sea level [m]

        Returns:
            dict: {
                'temperature': Temperature [K],
                'pressure': Pressure [Pa],
                'density': Density [kg/m^3],
                'speed_of_sound': Speed of sound [m/s]
            }
        """
        # Constrain altitude to valid ranges for stability
        h = max(0.0, float(h))

        # Geopotential/Geometric altitude difference is negligible below 20km for 3DOF
        if h < cls.h_tropopause:
            # Troposphere: standard linear lapse rate
            T = cls.T0 - cls.lambda_trop * h
            P = cls.P0 * (T / cls.T0) ** (cls.g0 / (cls.lambda_trop * cls.R))
        else:
            # Lower Stratosphere: Temperature is constant
            T = cls.T_tropopause
            # Pressure calculation at Tropopause boundary height
            P_trop = cls.P0 * (cls.T_tropopause / cls.T0) ** (cls.g0 / (cls.lambda_trop * cls.R))
            P = P_trop * np.exp(-cls.g0 * (h - cls.h_tropopause) / (cls.R * cls.T_tropopause))

        # Ideal Gas Law equation: P = rho * R * T => rho = P / (R * T)
        rho = P / (cls.R * T)

        # Speed of sound in ideal gas: a = sqrt(gamma * R * T)
        a = np.sqrt(cls.gamma * cls.R * T)

        return {
            'temperature': T,
            'pressure': P,
            'density': rho,
            'speed_of_sound': a
        }

    @classmethod
    def dynamic_pressure(cls, h: float, V: float) -> float:
        """
        Computes dynamic pressure: q = 0.5 * rho * V^2
        """
        rho = cls.get_properties(h)['density']
        return 0.5 * rho * (V ** 2)

    @classmethod
    def mach_number(cls, h: float, V: float) -> float:
        """
        Computes Mach number for speed V at altitude h
        """
        a = cls.get_properties(h)['speed_of_sound']
        return V / a

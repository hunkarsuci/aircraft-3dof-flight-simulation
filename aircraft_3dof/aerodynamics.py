import numpy as np
from aircraft_3dof.atmosphere import StandardAtmosphere

class Aerodynamics:
    """
    Compressible Aerodynamics Model representing aerodynamic lift and drag forces.
    Implements a Mach-dependent transonic drag rise polar:
    Cd(M, Cl) = Cd0(M) + K(M) * Cl^2
    """
    def __init__(self, wing_area: float, cd0_submin: float, cd0_wave_peak: float,
                 k_sub: float, k_sup_factor: float, m_crit: float, m_max: float):
        """
        Parameters:
            wing_area (float): Reference Wing Area S [m^2]
            cd0_submin (float): Minimum subsonic zero-lift drag coefficient
            cd0_wave_peak (float): Transonic shockwave drag increment peak (at Mach ~1.05)
            k_sub (float): Subsonic induced drag factor K_sub
            k_sup_factor (float): Factor by which induced drag K increases in supersonic (typically 2.0-2.5)
            m_crit (float): Critical Mach number (drag rise starts, typically 0.8)
            m_max (float): Peak wave drag Mach number location (typically 1.05)
        """
        self.S = wing_area
        self.cd0_submin = cd0_submin
        self.cd0_wave_peak = cd0_wave_peak
        self.k_sub = k_sub
        self.k_sup_factor = k_sup_factor
        self.m_crit = m_crit
        self.m_max = m_max

    def get_cd0(self, M: float) -> float:
        """
        Computes the zero-lift drag coefficient Cd0(M) using a smooth curve modeling wave drag:
          - Subsonic (M < M_crit): constant base drag
          - Transonic (M_crit <= M <= 1.2): smooth cubic transition to peak wave drag rise
          - Supersonic (M > 1.2): decaying wave drag trailing with 1/sqrt(M^2 - 1) behavior
        """
        if M < self.m_crit:
            return self.cd0_submin

        elif M <= 1.2:
            # Transonic rise transition blending (smooth cubic function)
            t = (M - self.m_crit) / (1.2 - self.m_crit)
            # Smooth Hermite dynamic curve peak
            cd0_rise = self.cd0_wave_peak * (3 * (t**2) - 2 * (t**3))
            return self.cd0_submin + cd0_rise

        else:
            # Supersonic wave drag decay: CD0 peaks then fades out (wave mechanics)
            wave_decay = self.cd0_wave_peak / np.sqrt(M**2 - 1.0)
            # Safe floor normalization
            return self.cd0_submin + max(0.005, min(self.cd0_wave_peak, wave_decay))

    def get_k(self, M: float) -> float:
        """
        Computes the induced drag factor K(M) which increases dynamically transonically
        due to supersonic shockwave generation and the loss of wing leading-edge suction.
        """
        if M < self.m_crit:
            return self.k_sub
        elif M <= 1.2:
            # Interpolate smoothly from subsonic to supersonic values
            t = (M - self.m_crit) / (1.2 - self.m_crit)
            k_sup = self.k_sub * self.k_sup_factor
            return self.k_sub + (k_sup - self.k_sub) * (3 * (t**2) - 2 * (t**3))
        else:
            return self.k_sub * self.k_sup_factor

    def compute_forces(self, h: float, V: float, CL: float) -> tuple:
        """
        Computes physical Lift (L) and Drag (D) forces.

        Parameters:
            h (float): Altitude [m]
            V (float): Airspeed [m/s]
            CL (float): Commanded lift coefficient

        Returns:
            tuple: (Lift [N], Drag [N])
        """
        q = StandardAtmosphere.dynamic_pressure(h, V)
        M = StandardAtmosphere.mach_number(h, V)

        # Calculate lift force: L = q * S * CL
        Lift = q * self.S * CL

        # Calculate compressible drag coefficient
        cd0 = self.get_cd0(M)
        k = self.get_k(M)
        CD = cd0 + k * (CL ** 2)

        # Drag force: D = q * S * CD
        Drag = q * self.S * CD

        return Lift, Drag

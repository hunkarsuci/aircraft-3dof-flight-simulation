from aircraft_3dof.atmosphere import StandardAtmosphere
from aircraft_3dof.aerodynamics import Aerodynamics
from aircraft_3dof.propulsion import Propulsion

class Aircraft:
    """
    Combines aerodynamics, propulsion, mass, and structural limits into a unified model
    defining specific commercial or tactical jet aircraft assets.
    """
    def __init__(self, name: str, empty_mass: float, max_fuel_mass: float, wing_area: float,
                 aerodynamics: Aerodynamics, propulsion: Propulsion,
                 n_lim_min: float, n_lim_max: float, cl_min: float, cl_max: float):
        """
        Parameters:
            name (str): Representative asset name (e.g. F-16 Falcon)
            empty_mass (float): Operating Empty Weight (OEW) [kg]
            max_fuel_mass (float): Internal Fuel Capacity [kg]
            wing_area (float): Reference Wing Surface Area S [m^2]
            aerodynamics (Aerodynamics): Polar parameters
            propulsion (Propulsion): Sea level static / density SFC maps
            n_lim_min (float): Minimum structural load limit [g] (downward force)
            n_lim_max (float): Maximum structural load limit [g] (upward load factor limit)
            cl_min (float): Safe aerodynamic stall lift coefficient limit (lower bound)
            cl_max (float): Peak aerodynamic wing stall lift coefficient limit (upper bound)
        """
        self.name = name
        self.empty_mass = empty_mass
        self.max_fuel_mass = max_fuel_mass
        self.S = wing_area
        self.aero = aerodynamics
        self.prop = propulsion
        self.n_lim_min = n_lim_min
        self.n_lim_max = n_lim_max
        self.cl_min = cl_min
        self.cl_max = cl_max

    def __repr__(self):
        return f"<Aircraft: {self.name} (OEW: {self.empty_mass}kg, S: {self.S}m^2)>"


def get_aircraft_presets() -> dict:
    """
    Dynamic generation of preset configurations representing academic models of:
    1. F-16 Fighting Falcon (High-g supersonic tactical interceptor)
    2. Boeing 737-800 Commercial Jetliner (Highly efficient subsonic transports)
    3. Learjet 35 Business Jet (Swift executive cruiser)
    """

    # 1. F-16 preset definition
    f16_aero = Aerodynamics(
        wing_area=27.8,       # S [m^2]
        cd0_submin=0.015,     # Highly streamlined military fighter profile
        cd0_wave_peak=0.035,  # Moderate transonic wave rise for clean wings
        k_sub=0.13,           # Subsonic lift-to-induced-drag efficiency
        k_sup_factor=2.2,     # High-mach supersonic loss of leading-edge suction
        m_crit=0.82,
        m_max=1.05
    )
    # Typical conversion: SFC = 1.0 lb/(lbf*hr) ~ 2.83e-5 kg/(N*s)
    # With military afterburner, static max thrust is ~127,000 N (127 kN)
    f16_prop = Propulsion(
        max_thrust_sl=127000.0,
        sfc_sl=2.83e-5,
        density_exp=0.85,     # F16 engine core works efficiently in high-density altitude layers
        mach_recovery=0.25,   # Ram recovery increases thrust at high Mach
        mach_exp=1.6,
        sfc_mach_lapse=0.45    # Afterburner/Mach drag-flow penalty
    )
    f16 = Aircraft(
        name="F-16 Falcon",
        empty_mass=8500.0,
        max_fuel_mass=3200.0,
        wing_area=27.8,
        aerodynamics=f16_aero,
        propulsion=f16_prop,
        n_lim_min=-3.0,
        n_lim_max=9.0,        # Maximum pilot load factor limits
        cl_min=-0.4,
        cl_max=1.3            # Tactical agile high lift boundary
    )

    # 2. Boeing 737-800 Preset
    b737_aero = Aerodynamics(
        wing_area=125.0,      # S [m^2]
        cd0_submin=0.020,     # Commercial airliner drag base (instruments, wipers, engine pods)
        cd0_wave_peak=0.090,  # Aggressive wave drag rise past critical mach
        k_sub=0.045,          # Highly efficient high-aspect ratio subsonic wing (extremely low induced drag)
        k_sup_factor=3.0,     # Very high penalty at supersonic speeds (non-designed flight envelope)
        m_crit=0.78,          # Designed cruise around Mach 0.78-0.8
        m_max=1.05
    )
    # Typical high-bypass CFM56 engine: SFC = 0.6 lb/(lbf*hr) ~ 1.7e-5 kg/(N*s)
    # 2x Engines: Max SL thrust = 2 * 117,000 N = 234,000 N
    b737_prop = Propulsion(
        max_thrust_sl=234000.0,
        sfc_sl=1.70e-5,
        density_exp=0.95,     # Sharp dry thrust lapse with altitude density drop
        mach_recovery=0.08,   # Subsonic optimized inlet (minimal high mach recovery)
        mach_exp=1.0,
        sfc_mach_lapse=0.25
    )
    b737 = Aircraft(
        name="Boeing 737-800",
        empty_mass=41000.0,
        max_fuel_mass=20800.0,
        wing_area=125.0,
        aerodynamics=b737_aero,
        propulsion=b737_prop,
        n_lim_min=-1.0,
        n_lim_max=2.5,        # FAA standard transport airworthiness limits
        cl_min=-0.2,
        cl_max=1.6            # Moderately high lift wing profiles
    )

    # 3. Learjet 35 Preset
    learjet_aero = Aerodynamics(
        wing_area=23.5,       # S [m^2]
        cd0_submin=0.018,     # Clean high aspect-ratio executive profile
        cd0_wave_peak=0.075,
        k_sub=0.055,
        k_sup_factor=2.6,
        m_crit=0.79,
        m_max=1.05
    )
    # Max SL Thrust: 2 * 16,000 N = 32,000 N
    # Typical TFE731 Turbofan static SFC: 0.55 lb/(lbf*hr) ~ 1.56e-5 kg/(N*s)
    learjet_prop = Propulsion(
        max_thrust_sl=32000.0,
        sfc_sl=1.56e-5,
        density_exp=0.90,
        mach_recovery=0.12,
        mach_exp=1.2,
        sfc_mach_lapse=0.30
    )
    learjet = Aircraft(
        name="Learjet 35",
        empty_mass=4600.0,
        max_fuel_mass=2800.0,
        wing_area=23.5,
        aerodynamics=learjet_aero,
        propulsion=learjet_prop,
        n_lim_min=-1.0,
        n_lim_max=3.0,
        cl_min=-0.3,
        cl_max=1.5
    )

    return {
        "f16": f16,
        "b737": b737,
        "learjet": learjet
    }

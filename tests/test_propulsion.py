import pytest
from aircraft_3dof.propulsion import Propulsion

def test_thrust_sea_level_static():
    prop = Propulsion(
        max_thrust_sl=100000.0, sfc_sl=2.0e-5, density_exp=0.9,
        mach_recovery=0.1, mach_exp=1.0, sfc_mach_lapse=0.3
    )

    # Static SL thrust at zero Mach and full throttle should equal sea level rating
    thrust, fuel_flow = prop.compute_thrust_and_fuel_flow(h=0.0, V=0.001, throttle=1.0)
    assert thrust == pytest.approx(100000.0, abs=10.0)
    assert fuel_flow == pytest.approx(2.0, abs=1e-2) # fuel flow = SFC * Thrust = 2.0e-5 * 1e5 = 2.0 kg/s

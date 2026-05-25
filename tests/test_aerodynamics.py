import pytest
from aircraft_3dof.aerodynamics import Aerodynamics

def test_subsonic_drag_polar():
    aero = Aerodynamics(
        wing_area=30.0, cd0_submin=0.02, cd0_wave_peak=0.06,
        k_sub=0.05, k_sup_factor=2.0, m_crit=0.8, m_max=1.05
    )

    # Subsonic Cd0 verify
    assert aero.get_cd0(0.5) == 0.02
    assert aero.get_k(0.5) == 0.05

    # Subsonic forces verify at dynamic pressure q = 5000 Pa, CL = 0.4
    # L = q * S * CL = 5000 * 30 * 0.4 = 60000 N
    # CD = 0.02 + 0.05 * 0.4^2 = 0.02 + 0.008 = 0.028
    # D = q * S * CD = 5000 * 30 * 0.028 = 4200 N
    lift, drag = aero.compute_forces(h=0.0, V=90.35, CL=0.4) # dynamic pressure ~5000 Pa
    assert lift == pytest.approx(60000.0, abs=100.0)
    assert drag == pytest.approx(4200.0, abs=50.0)

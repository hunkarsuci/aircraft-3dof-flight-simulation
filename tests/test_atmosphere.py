import pytest
from aircraft_3dof.atmosphere import StandardAtmosphere

def test_sea_level():
    # standard SL conditions checking
    props = StandardAtmosphere.get_properties(0.0)
    assert props['temperature'] == pytest.approx(288.15, abs=1e-2)
    assert props['pressure'] == pytest.approx(101325.0, abs=10.0)
    assert props['density'] == pytest.approx(1.225, abs=1e-3)
    assert props['speed_of_sound'] == pytest.approx(340.29, abs=0.5)

def test_tropopause_boundary():
    # isothermal boundary checking at 11,000 meters
    props = StandardAtmosphere.get_properties(11000.0)
    assert props['temperature'] == pytest.approx(216.65, abs=1e-2)
    assert props['density'] == pytest.approx(0.36391, abs=1e-3)

def test_dynamic_pressure():
    q = StandardAtmosphere.dynamic_pressure(h=0.0, V=100.0)
    # q = 0.5 * 1.225 * 100^2 = 6125 Pa
    assert q == pytest.approx(6125.0, abs=1.0)

import numpy as np
import pytest

from aircraft_3dof.aircraft import get_aircraft_presets
from aircraft_3dof.controls import Autopilot


def level_state(mass=6000.0):
    return np.array([0.0, 0.0, 3000.0, 180.0, 0.0, 0.0, mass])


def test_autopilot_reset_clears_speed_integral():
    autopilot = Autopilot()
    autopilot.v_error_integral = 4.5

    autopilot.reset()

    assert autopilot.v_error_integral == 0.0


def test_autopilot_throttle_saturates_for_large_speed_command():
    aircraft = get_aircraft_presets()["learjet"]
    autopilot = Autopilot(kp_v=0.1, ki_v=0.01)

    throttle, cl_cmd, bank_cmd = autopilot.compute_controls(
        t=0.0,
        state=level_state(mass=aircraft.empty_mass + 1500.0),
        aircraft=aircraft,
        v_cmd=260.0,
        h_cmd=3000.0,
        chi_cmd=0.0,
        dt=0.2,
    )

    assert throttle == pytest.approx(1.0)
    assert aircraft.cl_min <= cl_cmd <= aircraft.cl_max
    assert bank_cmd == pytest.approx(0.0, abs=1e-12)


def test_autopilot_heading_command_generates_limited_bank_angle():
    aircraft = get_aircraft_presets()["f16"]
    autopilot = Autopilot()

    _, _, bank_cmd = autopilot.compute_controls(
        t=0.0,
        state=level_state(mass=aircraft.empty_mass + 2000.0),
        aircraft=aircraft,
        v_cmd=180.0,
        h_cmd=3000.0,
        chi_cmd=np.pi,
        dt=0.2,
    )

    assert abs(bank_cmd) > 0.0
    assert abs(bank_cmd) <= np.radians(45.0)

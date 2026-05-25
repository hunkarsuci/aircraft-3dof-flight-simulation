import pytest
import numpy as np
from aircraft_3dof.aircraft import get_aircraft_presets
from aircraft_3dof.controls import Autopilot
from aircraft_3dof.simulation import FlightSimulator

def test_end_to_end_sim_step():
    presets = get_aircraft_presets()
    learjet = presets['learjet']
    ap = Autopilot()
    sim = FlightSimulator(aircraft=learjet, autopilot=ap)

    # Level cruise state: x, y, h, V, gamma, chi, mass
    state0 = np.array([0.0, 0.0, 10000.0, 200.0, 0.0, 0.0, 6000.0])

    # Run a short integration profile step (3 seconds)
    df = sim.run_simulation(
        t_end=3.0,
        initial_state=state0,
        v_cmd=200.0,
        h_cmd=10000.0,
        chi_cmd=0.0,
        step_size=0.5
    )

    assert len(df) > 0
    # Altitude should hold stable within tight tolerance
    assert df['altitude'].iloc[-1] == pytest.approx(10000.0, abs=15.0)
    # Mass should decrease indicating fuel burn
    assert df['mass'].iloc[-1] < 6000.0

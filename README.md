# Aircraft 3DOF Flight Simulation

Python simulation suite for educational 3-degree-of-freedom (3DOF) point-mass aircraft dynamics. The project includes atmosphere, aerodynamics, propulsion, closed-loop guidance, numerical integration, and four runnable mission scenarios that generate analysis plots.

The model is intended for flight-dynamics learning, controls experiments, and scenario visualization. It is not a certified flight-performance tool.

## Features

- 3DOF point-mass equations of motion in local flat-Earth coordinates
- 1976-style standard atmosphere model up to the lower stratosphere
- Mach-dependent aerodynamic drag polar with transonic drag rise
- Jet propulsion model with density lapse, Mach recovery, and fuel burn
- Closed-loop speed, altitude, and heading guidance
- Preset aircraft models: F-16, Boeing 737-800, and Learjet 35
- Four scenario scripts that save PNG plots to `plots/`
- Pytest coverage for atmosphere, aerodynamics, propulsion, and integration

## Coordinate Convention

The state vector is:

```text
[x, y, h, V, gamma, chi, mass]
```

Where:

- `x`: Easting position in meters
- `y`: Northing position in meters
- `h`: altitude above sea level in meters
- `V`: true airspeed in meters per second
- `gamma`: flight-path angle in radians
- `chi`: heading angle in radians, measured clockwise from north
- `mass`: aircraft mass in kilograms

With this convention:

- `chi = 0` points north
- `chi = pi / 2` points east
- waypoint heading is computed with `atan2(dx, dy)`

## Equations Of Motion

The 3DOF point-mass model projects thrust, drag, lift, and gravity into the velocity-frame dynamics:

```text
dx/dt     = V cos(gamma) sin(chi)
dy/dt     = V cos(gamma) cos(chi)
dh/dt     = V sin(gamma)
dV/dt     = (T - D) / m - g0 sin(gamma)
dgamma/dt = (L cos(mu) - m g0 cos(gamma)) / (m V)
dchi/dt   = L sin(mu) / (m V cos(gamma))
dm/dt     = -fuel_flow
```

Where:

- `T`: thrust force
- `D`: drag force
- `L`: lift force
- `mu`: flight-path bank angle
- `g0 = 9.80665 m/s^2`

The implementation includes guardrails for low speed, ground contact, fuel exhaustion, structural load limits, and near-vertical heading singularities.

## Repository Layout

```text
Aircraft-3dof/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- run_scenario_a.py
|-- run_scenario_b.py
|-- run_scenario_c.py
|-- run_scenario_d.py
|-- aircraft_3dof/
|   |-- __init__.py
|   |-- aerodynamics.py
|   |-- aircraft.py
|   |-- atmosphere.py
|   |-- controls.py
|   |-- dynamics.py
|   |-- propulsion.py
|   `-- simulation.py
`-- tests/
    |-- test_aerodynamics.py
    |-- test_atmosphere.py
    |-- test_propulsion.py
    `-- test_simulation.py
```

## Installation

Create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
cd C:\Users\hunka\Aircraft-3dof
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### macOS / Linux

```bash
cd Aircraft-3dof
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you use Anaconda and see a `PIL._imaging` or Matplotlib DLL error, your base environment likely has a broken Pillow/Matplotlib install. The quickest fix is to use a clean virtual environment. You can also repair the Conda environment with:

```powershell
conda install --force-reinstall pillow matplotlib
```

## Running Tests

```powershell
python -m pytest -q
```

Expected result:

```text
6 passed
```

## Running Scenarios

Each scenario runs a simulation and saves a PNG file in `plots/`.

```powershell
python run_scenario_a.py
python run_scenario_b.py
python run_scenario_c.py
python run_scenario_d.py
```

Scenario outputs:

| Script | Scenario | Output |
| --- | --- | --- |
| `run_scenario_a.py` | F-16 zoom climb and energy trade | `plots/scenario_a_zoom_climb.png` |
| `run_scenario_b.py` | Boeing 737 cruise and Breguet range check | `plots/scenario_b_airliner_cruise.png` |
| `run_scenario_c.py` | F-16 coordinated turn | `plots/scenario_c_coordinated_turn.png` |
| `run_scenario_d.py` | Learjet waypoint tracking and descent | `plots/scenario_d_waypoint_guidance.png` |

On Windows, if `python` points to the wrong interpreter, run with the full path to the interpreter that has the dependencies installed:

```powershell
C:\Users\hunka\AppData\Local\Programs\Python\Python313\python.exe run_scenario_a.py
```

## Scenario Notes

### Scenario A: F-16 Zoom Climb

Runs a two-phase maneuver:

1. Accelerating dash at high thrust
2. Pull-up climb that trades kinetic energy for altitude

The resulting plot shows altitude, true airspeed, kinetic energy, potential energy, total energy, flight-path angle, and load factor.

### Scenario B: Boeing 737 Cruise

Runs a long-range cruise segment and compares simulated range against a Breguet-style estimate. The current controller intentionally remains simple, so small altitude and lift-to-drag oscillations may be visible in the plot.

### Scenario C: F-16 Coordinated Turn

Runs a heading reversal with bank-angle limiting and structural load-factor logging.

### Scenario D: Learjet Waypoint Mission

Tracks a sequence of 3D waypoints and descends toward a final approach waypoint. This scenario is useful for visualizing horizontal guidance, altitude response, and speed-hold behavior.

## Model Components

### Atmosphere

`aircraft_3dof.atmosphere.StandardAtmosphere` computes:

- temperature
- pressure
- density
- speed of sound
- Mach number
- dynamic pressure

### Aerodynamics

`aircraft_3dof.aerodynamics.Aerodynamics` computes lift and drag:

```text
L = q S CL
D = q S CD
CD = CD0(M) + K(M) CL^2
```

### Propulsion

`aircraft_3dof.propulsion.Propulsion` computes thrust and fuel flow using:

```text
Tmax(h, M) = Tmax0 * (rho / rho0)^a * (1 + b M^c)
fuel_flow = SFC(M) * T
```

### Controls

`aircraft_3dof.controls.Autopilot` provides:

- speed hold with PI throttle control
- altitude-to-flight-path command generation
- lift-command inversion with bank compensation
- heading hold and waypoint steering

### Simulation

`aircraft_3dof.simulation.FlightSimulator` integrates the equations of motion with `scipy.integrate.solve_ivp` and returns a pandas `DataFrame` history.

## Example: Programmatic Use

```python
import numpy as np

from aircraft_3dof.aircraft import get_aircraft_presets
from aircraft_3dof.controls import Autopilot
from aircraft_3dof.simulation import FlightSimulator

aircraft = get_aircraft_presets()["learjet"]
autopilot = Autopilot()
sim = FlightSimulator(aircraft=aircraft, autopilot=autopilot)

initial_state = np.array([
    0.0,     # x [m]
    0.0,     # y [m]
    3000.0,  # h [m]
    180.0,   # V [m/s]
    0.0,     # gamma [rad]
    np.pi / 2.0,  # chi [rad], eastbound
    aircraft.empty_mass + 1500.0,
])

df = sim.run_simulation(
    t_end=60.0,
    initial_state=initial_state,
    v_cmd=180.0,
    h_cmd=3000.0,
    chi_cmd=np.pi / 2.0,
    step_size=0.2,
)

print(df.tail())
```

## Limitations

- Flat-Earth, point-mass model only; no attitude dynamics or rotational equations
- Simplified aerodynamic and propulsion maps
- Simplified autopilot intended for scenario demonstrations
- No wind, turbulence, actuator dynamics, or sensor errors
- Preset aircraft parameters are representative educational values, not authoritative aircraft data

## License

This project is configured as MIT-licensed in `pyproject.toml`.

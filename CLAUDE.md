# CLAUDE.md - Developer Guide for 3DOF Flight Dynamics Suite

This document records development instructions, coding patterns, environment setups, and mathematical practices for maintaining the 3DOF point-mass jet aircraft flight dynamics simulation codebase.

## 🛠️ Build and Environment Management

To set up the environment and run the package in editable mode:
```powershell
# Create dynamic python environment (optional/standard)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install requirements and package in development mode
pip install -r requirements.txt
pip install -e .
```

## 🚀 Execution and Simulation Scenarios

All scenario runs are placed under the root directory. Run them to execute mathematical integrations, print performance results, and generate graphics inside the `plots/` directory:
```powershell
# Scenario A: Zoom Climb dynamics (F-16 Fighter Energy Trade)
python run_scenario_a.py

# Scenario B: Cruise Fuel Burn & Range (Boeing 737 Breguet Verification)
python run_scenario_b.py

# Scenario C: Steady High-g Coordinated Defensive Turning (F-16 Loading)
python run_scenario_c.py

# Scenario D: 3D Waypoint guidance follow & Landing glide slope (Learjet 35)
python run_scenario_d.py
```

## 🧪 Testing Suite (`pytest`)

Execute unit tests to confirm mathematical models (Standard Atmosphere equations, aerodynamic lift and zero-lift transonic rise, thrust lapse parameters, dynamic ODE trajectories):
```PowerShell
# Run all unit tests
pytest

# Run tests with terminal logging
pytest -v
```

## 🧭 Mathematical Coding Rules

1. **Unit Consistency**: All physical quantities **MUST** use the metric system (SI) in internal code:
   * Coordinates, Altitude ($x, y, h$): Meters ($m$)
   * Velocity, Airspeed ($V, a$): Meters per second ($m/s$)
   * Thrust, Aerodynamic Drag, Lift ($T, D, L$): Newtons ($N$)
   * Mass ($m$): Kilograms ($kg$)
   * Specific Fuel Consumption ($SFC$): Kilograms per Newton-second ($kg/(N\cdot s)$)
   * Angles ($\alpha, \beta, \gamma, \chi, \mu$): Radians ($rad$)
   * G-load ($n$): Dimensionless units representing acceleration in terms of sea-level gravity ($g_0$)
2. **Coordinate Reference Frame**:
   * Horizontal pathing is handled in the Local Horizon Plane (Flat Earth approximation, positive North-East-Up-ish).
   * Accelerations are resolved along the Velocity Vector (Flight-Path wind-axis frame).
3. **No External Guessing**: Always verify ambient constants, lapse rates, critical Mach calculations, and coordinate conversions directly inside unit tests.

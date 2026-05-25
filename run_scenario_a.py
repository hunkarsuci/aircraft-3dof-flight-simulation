import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from aircraft_3dof.aircraft import get_aircraft_presets
from aircraft_3dof.controls import Autopilot
from aircraft_3dof.simulation import FlightSimulator

def main():
    print("=====================================================================")
    print("Scenario A: F-16 Zoom Climb Dynamics & Energy Trading Simulation")
    print("=====================================================================")

    # Create plots directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    # 1. Initialize fighter aircraft and simulation
    presets = get_aircraft_presets()
    f16 = presets['f16']

    # Auto-throttle & flight path controller
    ap = Autopilot(kp_v=0.08, ki_v=0.005, kp_gamma=0.25, max_g_load=6.0)
    sim = FlightSimulator(aircraft=f16, autopilot=ap)

    # 2. Setup flight state logs representing Zoom Climb initialization point:
    # State: [x (m), y (m), h (m), V (m/s), gamma (rad), chi (rad), mass (kg)]
    # Start at 1,000m altitude, speed Mach 0.9 (~306 m/s) in level flight (gamma=0) heading North
    x0, y0, h0 = 0.0, 0.0, 1000.0
    V0 = 306.0       # ~Mach 0.9 at 1000m
    gamma0 = 0.0     # Level flight slope
    chi0 = 0.0       # Heading North
    fuel0 = 2400.0   # 2400 kg internal fuel remaining
    m0 = f16.empty_mass + fuel0

    state0 = np.array([x0, y0, h0, V0, gamma0, chi0, m0])

    # 3. Simulate Phase 1: High speed dash at max thrust, climbing slightly to Mach 1.2
    # Commands: Max throttle zoom entry setup
    print("\nPhase 1: Max power speed-up dash toward transonic boundary...")
    df_dash = sim.run_simulation(
        t_end=30.0,
        initial_state=state0,
        v_cmd=400.0,      # Request supersonic speed to compress energy state
        h_cmd=1500.0,     # Small climb
        chi_cmd=0.0,
        step_size=0.1
    )

    # Extract dynamic state at dash completion to feed Zoom Climb entry
    last_row = df_dash.iloc[-1]
    dash_end_state = np.array([
        last_row['x'], last_row['y'], last_row['altitude'],
        last_row['velocity'], last_row['gamma'], last_row['chi'],
        last_row['mass']
    ])

    print(f"Dash end: Altitude = {last_row['altitude']:.1f} m, Airspeed = {last_row['velocity']:.1f} m/s (Mach {last_row['mach']:.2f})")

    # 4. Simulate Phase 2: Tactical Pull-up Zoom Climb (steep elevation target)
    print("\nPhase 2: Tactical zoom pull-up to 45 deg steep climb slope...")
    # Command high ceiling to invoke autopilot saturation climbing: target 15,000m altitude
    df_zoom = sim.run_simulation(
        t_end=60.0,
        initial_state=dash_end_state,
        v_cmd=260.0,      # Allow speed decay to convert kinetic into potential energy
        h_cmd=12000.0,    # Target vertical high layer
        chi_cmd=0.0,
        step_size=0.1
    )

    # Combine dataframes for unified plotting
    df_zoom['time'] += df_dash['time'].iloc[-1] # Offset time indexes
    df_full = pd.concat([df_dash, df_zoom], ignore_index=True)

    # 5. Aerospace Energy Analysis (Master's level verification)
    g0 = 9.80665
    df_full['E_kinetic'] = 0.5 * df_full['mass'] * (df_full['velocity']**2) / 1e6  # [MJ]
    df_full['E_potential'] = df_full['mass'] * g0 * df_full['altitude'] / 1e6      # [MJ]
    df_full['E_total'] = df_full['E_kinetic'] + df_full['E_potential']             # [MJ]

    # Generate Performance Plots
    print("\nPlotting energy trade curves and flight envelope history...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Envelope plot: Altitude & Airspeed vs Time
    color = 'tab:blue'
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel('Altitude [m]', color=color)
    axs[0, 0].plot(df_full['time'], df_full['altitude'], color=color, linewidth=2, label='Altitude')
    axs[0, 0].tick_params(axis='y', labelcolor=color)
    axs[0, 0].grid(True)

    ax2 = axs[0, 0].twinx()
    color = 'tab:red'
    ax2.set_ylabel('True Airspeed [m/s]', color=color)
    ax2.plot(df_full['time'], df_full['velocity'], color=color, linestyle='--', linewidth=2, label='TAS')
    ax2.tick_params(axis='y', labelcolor=color)
    axs[0, 0].set_title('Flight Trajectory Envelope profile')

    # Energy conservation plot (Potential vs Kinetic)
    axs[0, 1].plot(df_full['time'], df_full['E_kinetic'], color='r', label='Kinetic Energy (0.5*m*V^2)')
    axs[0, 1].plot(df_full['time'], df_full['E_potential'], color='g', label='Potential Energy (m*g*h)')
    axs[0, 1].plot(df_full['time'], df_full['E_total'], color='b', linestyle='-.', label='Specific Total Energy')
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Energy [MJ]')
    axs[0, 1].set_title('Specific Energy Conservation Curve')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Flight path angle vs Time
    axs[1, 0].plot(df_full['time'], np.degrees(df_full['gamma']), 'purple', linewidth=2)
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel(r'Flight-Path Angle, $\gamma$ [deg]')
    axs[1, 0].set_title('Elevation/Climb Angle steering profile')
    axs[1, 0].grid(True)

    # Load factor G vs Time
    axs[1, 1].plot(df_full['time'], df_full['g_load'], 'orange', linewidth=2)
    axs[1, 1].axhline(y=1.0, color='gray', linestyle='--', label='1g level flight')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Structural G Load Factor [g]')
    axs[1, 1].set_title('Structural Loading')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("plots/scenario_a_zoom_climb.png", dpi=300)
    print("Success! Scenario A plots saved to plots/scenario_a_zoom_climb.png")
    print("Simulation finished successfully.\n")

if __name__ == "__main__":
    main()

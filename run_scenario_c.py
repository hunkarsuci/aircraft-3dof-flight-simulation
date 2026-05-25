import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aircraft_3dof.aircraft import get_aircraft_presets
from aircraft_3dof.controls import Autopilot
from aircraft_3dof.simulation import FlightSimulator

def main():
    print("=====================================================================")
    print("Scenario C: F-16 Tactical High-g Steady Coordinated Turn Performance")
    print("=====================================================================")

    # Create plots directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    # 1. Initialize tactical fighter jet
    presets = get_aircraft_presets()
    f16 = presets['f16']

    # Auto-throttle & flight path controller (maneuver limits set higher, up to 7.0g)
    ap = Autopilot(kp_v=0.10, ki_v=0.005, kp_gamma=0.25, max_g_load=7.0)
    sim = FlightSimulator(aircraft=f16, autopilot=ap)

    # 2. Setup initial stable level flight states at 5,000m altitude
    # Airspeed: Mach 0.85 (~272 m/s at 5000m)
    x0, y0, h0 = 0.0, 0.0, 5000.0
    V0 = 272.0
    gamma0 = 0.0
    chi0 = 0.0       # Heading North (0 rad)
    fuel0 = 2000.0   # 2,000 kg fuel remain
    m0 = f16.empty_mass + fuel0

    state0 = np.array([x0, y0, h0, V0, gamma0, chi0, m0])

    # 3. Simulate high-g banking coordinated turn:
    # Commands: target climb slope at 5000m, speed 272 m/s, command a sharp 180-deg heading turn (pi rad target)
    print("\nSimulating coordinated banking defensive turn to reverse heading...")
    df_turn = sim.run_simulation(
        t_end=40.0,         # 40 seconds duration
        initial_state=state0,
        v_cmd=272.0,        # Speed target
        h_cmd=5000.0,       # Altitude lock
        chi_cmd=np.pi,      # Heading turn clockwise to South (180 deg)
        step_size=0.1
    )

    # 4. Aerospace Steady Turn Mechanics Analysis
    # Coordinated Turn equations: R_turn = V^2 / (g0 * sqrt(n^2 - 1)), turn_rate = g0 * tan(bank_angle) / V
    g0 = 9.80665
    print("\n--------------------------------------------------------------")
    print("           HIGH-G TURNING FLIGHT PERFORMANCE REPORTS")
    print("--------------------------------------------------------------")
    max_bank_deg = np.degrees(np.max(np.abs(df_turn['bank_angle'])))
    max_g = np.max(df_turn['g_load'])
    min_speed = np.min(df_turn['velocity'])

    print(f"Initial Speed      : {V0:.2f} m/s (Mach 0.85)")
    print(f"Minimum Turn Speed : {min_speed:.2f} m/s (demonstrates induced drag energy bleed)")
    print(f"Maximum Bank Angle : {max_bank_deg:.2f} deg")
    print(f"Peak Turn G load   : {max_g:.2f} g")
    print("--------------------------------------------------------------")

    # Generate Performance Plots
    print("\nGenerating ground track and structural loading turn profile plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 2D ground track: Easting (x) vs Northing (y)
    axs[0, 0].plot(df_turn['x'], df_turn['y'], 'b', linewidth=2.5)
    axs[0, 0].set_xlabel('Easting horizontal position, X [m]')
    axs[0, 0].set_ylabel('Northing horizontal position, Y [m]')
    axs[0, 0].set_title('Flat Earth 2D Ground Track horizontal path')
    axs[0, 0].axis('equal')
    axs[0, 0].grid(True)

    # Coordinated Roll Bank angle vs Time
    axs[0, 1].plot(df_turn['time'], np.degrees(df_turn['bank_angle']), 'g', linewidth=2)
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel(r'Coordinated Bank Angle, $\mu$ [deg]')
    axs[0, 1].set_title('Autopilot roll steer profiles')
    axs[0, 1].grid(True)

    # G force loading vs Time
    axs[1, 0].plot(df_turn['time'], df_turn['g_load'], 'orange', linewidth=2)
    axs[1, 0].axhline(y=1.0, color='gray', linestyle='--')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Force load factor, $n$ [g]')
    axs[1, 0].set_title('Structural Loading Factor history')
    axs[1, 0].grid(True)

    # Altitude tracking vs Time
    axs[1, 1].plot(df_turn['time'], df_turn['altitude'], 'r', linewidth=2)
    axs[1, 1].axhline(y=5000.0, color='gray', linestyle='--', label='Command Lock')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('Altitude [m]')
    axs[1, 1].set_title('Altitude Deviation throughout execution')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("plots/scenario_c_coordinated_turn.png", dpi=300)
    print("Success! Scenario C plots saved to plots/scenario_c_coordinated_turn.png")
    print("Simulation finished successfully.\n")

if __name__ == "__main__":
    main()

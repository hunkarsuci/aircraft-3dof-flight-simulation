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
    print("Scenario B: Boeing 737 Fuel Range Cruise & Breguet Range Verification")
    print("=====================================================================")

    # Create plots directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    # 1. Initialize commercial jetliner and simulation
    presets = get_aircraft_presets()
    b737 = presets['b737']

    # Auto-throttle & flight path controller
    ap = Autopilot(kp_v=0.06, ki_v=0.003, kp_gamma=0.06, max_g_load=2.0)
    sim = FlightSimulator(aircraft=b737, autopilot=ap)

    # 2. Setup cruise initial flight states:
    # State: [x (m), y (m), h (m), V (m/s), gamma (rad), chi (rad), mass (kg)]
    # Cruise Altitude: 11,000 m (Stratosphere isothermal layer)
    # Target Speed: Cruise Mach 0.78 (~230 m/s at 11,000m)
    x0, y0, h0 = 0.0, 0.0, 11000.0
    V0 = 230.0
    gamma0 = 0.0
    chi0 = 0.0
    fuel0 = 10000.0   # 10,000 kg cruise fuel configuration
    m0 = b737.empty_mass + fuel0

    state0 = np.array([x0, y0, h0, V0, gamma0, chi0, m0])

    # 3. Simulate Long Range Cruise for 1.5 hours (5400 seconds)
    print("\nSimulating stable commercial cruise at 11,000m, Mach 0.78...")
    df_cruise = sim.run_simulation(
        t_end=5400.0,       # 1.5 hours
        initial_state=state0,
        v_cmd=230.0,        # Cruise TAS command
        h_cmd=11000.0,      # Cruise Altitude command
        chi_cmd=0.0,
        step_size=1.0       # Faster time integration step size for cruise
    )

    # 4. Integrate Breguet analysis comparing actual vs simulated flight distance
    # Breguet Cruise formula: Range = V * L_over_D / (SFC * g0) * ln(m_start / m_end)
    g0 = 9.80665
    m_start = df_cruise['mass'].iloc[0]
    m_end = df_cruise['mass'].iloc[-1]

    # Calculate average simulated L/D and average SFC
    # Realized aerodynamic L/D calculation: Lift / Drag
    from aircraft_3dof.atmosphere import StandardAtmosphere
    q_arr = np.array([
        StandardAtmosphere.dynamic_pressure(h, v)
        for h, v in zip(df_cruise['altitude'].values, df_cruise['velocity'].values)
    ])
    lift_arr = q_arr * b737.S * df_cruise['CL'].values
    # Extract Drag from Aero Polar directly for accuracy
    M_arr = np.array([
        StandardAtmosphere.mach_number(h, v)
        for h, v in zip(df_cruise['altitude'].values, df_cruise['velocity'].values)
    ])
    cd_arr = np.array([b737.aero.get_cd0(M) + b737.aero.get_k(M) * CL**2 for M, CL in zip(M_arr, df_cruise['CL'].values)])
    drag_arr = q_arr * b737.S * cd_arr
    ld_arr = lift_arr / drag_arr
    ld_avg = np.mean(ld_arr)

    # Average Specific Fuel Consumption tracking
    sfc_arr = np.array([b737.prop.get_sfc(M) for M in M_arr])
    sfc_avg = np.mean(sfc_arr)

    # Apply classical Breguet Cruise Range equation:
    breguet_range = (V0 * ld_avg) / (sfc_avg * g0) * np.log(m_start / m_end) # [m]

    simulated_range = np.hypot(df_cruise['x'].iloc[-1], df_cruise['y'].iloc[-1]) # [m]

    print("\n--------------------------------------------------------------")
    print("           BREGUET RANGE ANALYSES VERIFICATION")
    print("--------------------------------------------------------------")
    print(f"Simulated Flight Duration: {5400.0/3600.0:.2f} hours")
    print(f"Initial Flight Mass    : {m_start:.1f} kg (OEW + 10,000 kg Fuel)")
    print(f"Final Flight Mass      : {m_end:.1f} kg")
    print(f"Average Aerodynamic L/D: {ld_avg:.2f}")
    print(f"Average Engine SFC     : {sfc_avg:.5e} kg/(N*s)")
    print(f"Analytical Breguet Range: {breguet_range/1e3:.2f} km")
    print(f"Simulated Dynamic Range: {simulated_range/1e3:.2f} km")
    range_error = np.abs(simulated_range - breguet_range) / breguet_range * 100.0
    print(f"Verification Error %   : {range_error:.4f} %")
    print("--------------------------------------------------------------")

    # Generate Performance Plots
    print("\nGenerating fuel consumption and aerodynamic cruise plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Cruise Altitude hold confirmation
    axs[0, 0].plot(df_cruise['time']/3600.0, df_cruise['altitude'], 'b', linewidth=2)
    axs[0, 0].axhline(y=11000.0, color='gray', linestyle='--', label='Target Altitude')
    axs[0, 0].set_xlabel('Time [hours]')
    axs[0, 0].set_ylabel('Altitude [m]')
    axs[0, 0].set_title('Altitude Steering performance')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Mass decay/fuel burn curve vs time
    axs[0, 1].plot(df_cruise['time']/3600.0, df_cruise['mass'], 'r', linewidth=2)
    axs[0, 1].set_xlabel('Time [hours]')
    axs[0, 1].set_ylabel('Aircraft Total Mass [kg]')
    axs[0, 1].set_title('Aircraft Mass Decay (Fuel Weight loss)')
    axs[0, 1].grid(True)

    # Instantaneous Lift-to-Drag Ratio (L/D) vs time
    axs[1, 0].plot(df_cruise['time']/3600.0, ld_arr, 'g', linewidth=2)
    axs[1, 0].set_xlabel('Time [hours]')
    axs[1, 0].set_ylabel('Aerodynamic Lift-to-Drag Ratio ($L/D$)')
    axs[1, 0].set_title('Instantaneous Cruise Efficiency ($L/D$)')
    axs[1, 0].grid(True)

    # Simulated Flight distance vs time
    ground_range = np.hypot(df_cruise['x'], df_cruise['y'])
    axs[1, 1].plot(df_cruise['time']/3600.0, ground_range/1e3, 'purple', linewidth=2)
    axs[1, 1].set_xlabel('Time [hours]')
    axs[1, 1].set_ylabel('Horizontal Simulated Distance [km]')
    axs[1, 1].set_title('Forward Ground Track Range')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("plots/scenario_b_airliner_cruise.png", dpi=300)
    print("Success! Scenario B plots saved to plots/scenario_b_airliner_cruise.png")
    print("Simulation finished successfully.\n")

if __name__ == "__main__":
    main()

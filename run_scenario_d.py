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
    print("Scenario D: Learjet 35 Closed-loop 3D Waypoint Tracking & Landing glide")
    print("=====================================================================")

    # Create plots directory if it does not exist
    os.makedirs("plots", exist_ok=True)

    # 1. Initialize Business Jet
    presets = get_aircraft_presets()
    learjet = presets['learjet']

    # Guidance Autopilot configuration
    ap = Autopilot(kp_v=0.08, ki_v=0.005, kp_gamma=0.06, max_g_load=3.0)
    sim = FlightSimulator(aircraft=learjet, autopilot=ap)

    # 2. Define standard flight plan (waypoints tracking array):
    # Mission: Flight plan transition paths starting at 3000m, performing a turning pattern,
    # then initiating a glide slope down to airport landing approach at 200m altitude.
    waypoints = [
        {'x': 5000.0,  'y': 0.0,     'h': 3000.0},  # WP 1: Level horizontal climb out
        {'x': 10000.0, 'y': 5000.0,  'h': 2000.0},  # WP 2: Turning step descent
        {'x': 15000.0, 'y': 10000.0, 'h': 1000.0},  # WP 3: Approaching outer marker
        {'x': 20000.0, 'y': 12000.0, 'h': 200.0}    # WP 4: Final approach landing glideslope
    ]

    # Initial stable cruise condition at WP0 waypoint
    # State: [x, y, h, V, gamma, chi, m]
    x0, y0, h0 = 0.0, 0.0, 3000.0
    V0 = 180.0       # ~350 knots speed hold
    gamma0 = 0.0
    chi0 = np.pi / 2.0
    fuel0 = 1500.0
    m0 = learjet.empty_mass + fuel0

    state0 = np.array([x0, y0, h0, V0, gamma0, chi0, m0])

    # 3. Integrate closed-loop mission trajectory run
    print("\nSimulating flight plan tracking waypoints [WP1 -> WP4]...")
    df_mission = sim.run_waypoint_mission(
        waypoints=waypoints,
        initial_state=state0,
        v_cmd=180.0,
        step_size=0.2
    )

    # 4. Generate 3D trajectory visualization plots & path analysis
    print("\nPlotting flight plan trajectory profiles and cross track error charts...")
    fig = plt.figure(figsize=(15, 11))

    # 3D spatial path visualization plot
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot3D(df_mission['x']/1e3, df_mission['y']/1e3, df_mission['altitude'], 'b', linewidth=2.5, label='Flight Path')

    # Draw waypoint coordinates
    wpx = [0.0] + [wp['x']/1e3 for wp in waypoints]
    wpy = [0.0] + [wp['y']/1e3 for wp in waypoints]
    wph = [3000.0] + [wp['h'] for wp in waypoints]
    ax3d.scatter(wpx, wpy, wph, color='r', marker='o', s=80, label='Waypoints')

    # Add numeric labeling identifiers to help visualization
    for idx, (wx, wy, wh) in enumerate(zip(wpx, wpy, wph)):
        ax3d.text(wx, wy, wh + 80, f"WP {idx}", color='red', fontsize=10)

    ax3d.set_xlabel('Easting, X [km]')
    ax3d.set_ylabel('Northing, Y [km]')
    ax3d.set_zlabel('Altitude, Z [m]')
    ax3d.set_title('3D Multi-Waypoint Spatial Trajectory Path')
    ax3d.legend()

    # Altitude tracking vs horizontal range
    ax_alt = fig.add_subplot(2, 2, 2)
    ax_alt.plot(df_mission['time'], df_mission['altitude'], 'teal', linewidth=2.5, label='Actual altitude')
    # Dynamic commanded altitude tracking logs
    # Extract flight goals:
    ax_alt.set_xlabel('Time [s]')
    ax_alt.set_ylabel('Aircraft altitude [m]')
    ax_alt.set_title('3D Glide Slope Descent Profile')
    ax_alt.grid(True)
    ax_alt.legend()

    # Airspeed TAS and Mach stability hold plots
    ax_speed = fig.add_subplot(2, 2, 3)
    ax_speed.plot(df_mission['time'], df_mission['velocity'], 'orange', linewidth=2, label='TAS')
    ax_speed.set_xlabel('Time [s]')
    ax_speed.set_ylabel('True Airspeed [m/s]')
    ax_speed.set_title('Auto-throttle speed hold tracking')
    ax_speed.grid(True)

    # 2D flat earth horizontal geometry track
    ax_track = fig.add_subplot(2, 2, 4)
    ax_track.plot(df_mission['x']/1e3, df_mission['y']/1e3, 'purple', linewidth=2, label='Path')
    ax_track.scatter(wpx, wpy, color='red')
    ax_track.set_xlabel('Easting, X [km]')
    ax_track.set_ylabel('Northing, Y [km]')
    ax_track.set_title('Horizontal Position ground track projection')
    ax_track.grid(True)

    plt.tight_layout()
    plt.savefig("plots/scenario_d_waypoint_guidance.png", dpi=300)
    print("Success! Scenario D plots saved to plots/scenario_d_waypoint_guidance.png")
    print("Simulation finished successfully.\n")

if __name__ == "__main__":
    main()

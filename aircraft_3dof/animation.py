"""
Matplotlib animation helpers for 3DOF aircraft trajectories.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd


REQUIRED_TRAJECTORY_COLUMNS = ("time", "x", "y", "altitude")


def animate_trajectory(
    history: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    interval: int = 50,
    trail_length: Optional[int] = None,
    figsize: tuple = (8, 6),
    dpi: int = 120,
    writer: Optional[object] = None,
) -> FuncAnimation:
    """
    Create a 3D animation from a simulation history DataFrame.

    Args:
        history: Simulation output containing time, x, y, and altitude columns.
        save_path: Optional output path. The file format is inferred by Matplotlib.
        interval: Delay between frames in milliseconds.
        trail_length: Optional number of recent samples to show behind the aircraft.
        figsize: Matplotlib figure size in inches.
        dpi: Dots per inch used when saving.
        writer: Optional Matplotlib animation writer, such as "pillow" or "ffmpeg".

    Returns:
        The Matplotlib FuncAnimation instance.
    """
    _validate_history(history)

    times = history["time"].to_numpy(dtype=float)
    xs = history["x"].to_numpy(dtype=float)
    ys = history["y"].to_numpy(dtype=float)
    altitudes = history["altitude"].to_numpy(dtype=float)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("Altitude [m]")
    ax.set_title("3DOF Aircraft Trajectory")

    ax.set_xlim(*_axis_limits(xs))
    ax.set_ylim(*_axis_limits(ys))
    ax.set_zlim(*_axis_limits(altitudes, lower_bound=0.0))

    trajectory_line, = ax.plot([], [], [], color="tab:blue", linewidth=2)
    aircraft_marker, = ax.plot([], [], [], marker="o", color="tab:red", markersize=7)
    time_label = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        aircraft_marker.set_data([], [])
        aircraft_marker.set_3d_properties([])
        time_label.set_text("")
        return trajectory_line, aircraft_marker, time_label

    def update(frame: int):
        start = 0 if trail_length is None else max(0, frame - trail_length + 1)
        stop = frame + 1

        trajectory_line.set_data(xs[start:stop], ys[start:stop])
        trajectory_line.set_3d_properties(altitudes[start:stop])
        aircraft_marker.set_data([xs[frame]], [ys[frame]])
        aircraft_marker.set_3d_properties([altitudes[frame]])
        time_label.set_text(f"t = {times[frame]:.1f} s")
        return trajectory_line, aircraft_marker, time_label

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        init_func=init,
        interval=interval,
        blit=False,
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output_path, writer=writer, dpi=dpi)

    return animation


def _validate_history(history: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_TRAJECTORY_COLUMNS if column not in history.columns
    ]
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"history is missing required columns: {columns}")

    if history.empty:
        raise ValueError("history must contain at least one row")


def _axis_limits(values: np.ndarray, lower_bound: Optional[float] = None) -> tuple:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("axis data must contain at least one finite value")

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if lower_bound is not None:
        minimum = min(minimum, lower_bound)

    span = maximum - minimum
    padding = max(span * 0.05, 1.0)
    return minimum - padding, maximum + padding

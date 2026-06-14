import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation

from aircraft_3dof.animation import animate_trajectory


def sample_history():
    return pd.DataFrame(
        {
            "time": [0.0, 0.5, 1.0],
            "x": [0.0, 100.0, 220.0],
            "y": [0.0, 20.0, 55.0],
            "altitude": [1000.0, 1010.0, 1030.0],
        }
    )


def test_animate_trajectory_returns_func_animation():
    animation = animate_trajectory(sample_history(), interval=10, trail_length=2)

    assert isinstance(animation, FuncAnimation)
    assert len(list(animation.new_frame_seq())) == 3
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_animate_trajectory_rejects_missing_required_columns():
    history = sample_history().drop(columns=["altitude"])

    with pytest.raises(ValueError, match="altitude"):
        animate_trajectory(history)


def test_animate_trajectory_rejects_empty_history():
    history = sample_history().iloc[0:0]

    with pytest.raises(ValueError, match="at least one row"):
        animate_trajectory(history)

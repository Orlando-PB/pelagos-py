import numpy as np
import pandas as pd
import pytest

from pelagos_py.steps.processing.find_profiles import find_profiles


# Default parameters, mirroring FindProfilesStep.parameter_schema. Individual
# tests override only what they need via **overrides.
DEFAULT_PARAMS = dict(
    time_window_seconds=30,
    target_transect_phase=4,
    velocity_threshold=0.033,
    acceleration_threshold=0.0005,
    transition_buffer_seconds=30,
    min_duration_minutes=5,
    peak_prominence=20,
    min_samples_between_peaks=20,
    gap_threshold_minutes=5,
    surface_depth=20,
    surfacing_threshold=5,
    parking_gradient_threshold=0.005,
)


def make_dive_dataframe(
    n_cycles=3, leg_minutes=10, sample_seconds=10, min_depth=1.0, max_depth=120.0
):
    """Build a clean triangular dive record (descent/ascent legs) for profiling.

    Returns a DataFrame shaped like the one ``FindProfilesStep`` feeds to
    ``find_profiles``: an ``N_MEASUREMENTS`` index column, ``TIME`` and a depth
    column (here ``PRES``).
    """
    leg_samples = int(leg_minutes * 60 / sample_seconds)
    down = np.linspace(min_depth, max_depth, leg_samples)
    up = np.linspace(max_depth, min_depth, leg_samples)

    depth = np.concatenate([leg for _ in range(n_cycles) for leg in (down, up)])
    n = len(depth)
    # Nanosecond resolution to match real OG1 TIME (the step converts TIME to
    # epoch seconds assuming ns, so a coarser dtype would distort velocities).
    offsets = (np.arange(n) * sample_seconds * 1_000_000_000).astype("timedelta64[ns]")
    times = np.datetime64("2024-01-01T00:00:00", "ns") + offsets

    return pd.DataFrame(
        {"N_MEASUREMENTS": np.arange(n), "TIME": times, "PRES": depth}
    )


def run(df, **overrides):
    params = {**DEFAULT_PARAMS, **overrides}
    return find_profiles(df, "PRES", **params)


def test_output_columns_and_length():
    """The step preserves row count and adds the expected derived columns."""
    df = make_dive_dataframe()
    result = run(df)

    expected = {"PROFILE_NUMBER", "PROFILE_DIRECTION", "GRADIENT", "CYCLE", "SCI_PHASE"}
    assert expected.issubset(result.columns)
    assert len(result) == len(df)


def test_ascent_and_descent_detected():
    """Clear up/down legs are classified as ascent (1) and descent (2), and the
    derived direction is consistent with the phase (+1 descent, -1 ascent)."""
    result = run(make_dive_dataframe())
    phases = result["SCI_PHASE"]

    assert (phases == 1).any(), "expected some ascent samples"
    assert (phases == 2).any(), "expected some descent samples"

    descent = result.loc[phases == 2, "PROFILE_DIRECTION"]
    ascent = result.loc[phases == 1, "PROFILE_DIRECTION"]
    assert (descent == 1).all()
    assert (ascent == -1).all()


def test_multiple_profiles_numbered():
    """Several dive cycles produce several distinct, positive profile numbers."""
    result = run(make_dive_dataframe(n_cycles=3))
    profile_numbers = result["PROFILE_NUMBER"].dropna()

    assert profile_numbers.nunique() >= 2
    assert (profile_numbers >= 1).all()


def test_empty_input_returns_defaults():
    """An empty input still yields the derived columns without raising."""
    empty = pd.DataFrame({"N_MEASUREMENTS": [], "TIME": [], "PRES": []})
    result = run(empty)

    for col in ("PROFILE_NUMBER", "PROFILE_DIRECTION", "GRADIENT", "CYCLE", "SCI_PHASE"):
        assert col in result.columns

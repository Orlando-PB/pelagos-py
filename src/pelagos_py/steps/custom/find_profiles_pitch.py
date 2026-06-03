# This file is part of the NOC Autonomy Toolbox.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class definition for exporting data steps."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re


# Phase colours and names, matched to the standard find_profiles diagnostics
PHASE_COLOURS = {
    0: "#9ca3af",
    1: "#22c55e",
    2: "#3b82f6",
    3: "#f97316",
    4: "#a855f7",
    5: "#06b6d4",
    6: "#ef4444",
    7: "#eab308",
}

PHASE_NAMES = {
    0: "0 Unknown",
    1: "1 Ascent",
    2: "2 Descent",
    3: "3 Surfacing",
    4: "4 Parking",
    5: "5 Inflection",
    6: "6 Propelled",
    7: "7 Transition",
}


def _parse_duration_to_kwargs(duration: str) -> dict:
    """Parse a Polars duration string (e.g. '30s', '5m') into timedelta kwargs."""
    m = re.fullmatch(r"(\d+)([smhd])", duration)
    if not m:
        raise ValueError(f"Unsupported duration format: {duration}")
    val, unit = int(m.group(1)), m.group(2)
    return {"s": dict(seconds=val), "m": dict(minutes=val),
            "h": dict(hours=val), "d": dict(days=val)}[unit]


def _parse_duration_to_seconds(duration: str) -> float:
    """Parse a Polars duration string (e.g. '20s', '5m') into total seconds."""
    m = re.fullmatch(r"(\d+)([smhd])", duration)
    if not m:
        raise ValueError(f"Unsupported duration format: {duration}")
    val, unit = int(m.group(1)), m.group(2)
    return val * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _resolve_transitions(phases, depths, surface_depth=0.0, steal_from_surface=False):
    """Ensure a transition (7) exists between every consecutive pair of primary phases,
    then mark the depth extremum of each ascent<->descent transition as inflection (5).

    Parameters
    ----------
    phases : np.ndarray
        Integer phase array with primary phases (1, 2, 3, 6) assigned and zeros elsewhere.
        Points already marked as transition (7) by the disagreement criterion are preserved
        and incorporated into the transition segments for inflection detection.
    depths : np.ndarray
        Interpolated depth values aligned with phases.
    surface_depth : float, default=0.0
        Depth threshold (m). Transition points at or above this depth are reassigned to
        surfacing (3). Stealing is also suppressed when the preceding segment is surfacing
        (unless steal_from_surface=True).
    steal_from_surface : bool, default=False
        When True, boundary stealing is permitted even when the preceding segment is
        surfacing (3). Use on a second pass after the surfacing merge to enforce strict
        transitions between all primary phases.

    Returns
    -------
    np.ndarray
        Updated phase array with transitions (7) and inflections (5) resolved.
    """
    PRIMARY = {1, 2, 3, 6}
    result = phases.copy()
    n = len(result)

    # Identify contiguous primary-phase segments: (start, end_inclusive, phase)
    segs = []
    i = 0
    while i < n:
        if result[i] in PRIMARY:
            j = i + 1
            while j < n and result[j] == result[i]:
                j += 1
            segs.append((i, j - 1, int(result[i])))
            i = j
        else:
            i += 1

    # Insert transition (7) between every consecutive pair of primary segments.
    # Points already marked 7 by the disagreement criterion are naturally included;
    # adjacent segments without a gap have their boundary stolen instead.
    # Stealing is suppressed when the preceding segment is surfacing (3) unless
    # steal_from_surface=True, which is used on a second pass after surface merging
    # to re-enforce strict transitions between all remaining primary phase pairs.
    for k in range(len(segs) - 1):
        s0, e0, p0 = segs[k]
        s1,  _,  _ = segs[k + 1]
        if s1 > e0 + 1:
            # Natural gap: points between the two segments become transition
            result[e0 + 1 : s1] = 7
        elif p0 != 3 or steal_from_surface:
            # Adjacent segments: steal the last min(3, segment_length) points of the previous
            steal_n = min(3, e0 - s0 + 1)
            result[e0 - steal_n + 1 : e0 + 1] = 7

    # Reassign transition points within surface bounds to surfacing (3)
    result[(result == 7) & (depths <= surface_depth)] = 3

    # Assign inflection (5) at the depth extremum of each transition (7) segment
    # that is bracketed by opposite profiling phases (ascent <-> descent only)
    i = 0
    while i < n:
        if result[i] == 7:
            j = i + 1
            while j < n and result[j] == 7:
                j += 1
            prev_p = next((int(result[p]) for p in range(i - 1, -1, -1) if result[p] in PRIMARY), None)
            next_p = next((int(result[p]) for p in range(j, n) if result[p] in PRIMARY), None)
            if {prev_p, next_p} == {1, 2}:
                seg_d = depths[i:j]
                # descent->ascent: deepest point; ascent->descent: shallowest point
                local_idx = int(np.nanargmax(seg_d)) if prev_p == 2 else int(np.nanargmin(seg_d))
                result[i + local_idx] = 5
            i = j
        else:
            i += 1

    return result


def _assign_phase_sci(
    df, depth_col, gradient_thresholds, cust_gradient_thresholds,
    surface_depth, time_col, propelled_min_duration="1m",
):
    """Assign PHASE_SCI classification codes from smoothed signals.

    When pitch data are available, transitions (7) are first identified as points
    where the velocity and pitch*velocity criteria disagree. The fallback
    _resolve_transitions then inserts transitions between all consecutive primary
    phase pairs and marks inflections (5) at ascent<->descent turning points.

    Phase codes:
        0 unknown    - unclassified
        1 ascent     - platform climbing; both criteria confirm (smooth_grad < neg_grad)
        2 descent    - platform diving;   both criteria confirm (smooth_grad > pos_grad)
        3 surfacing  - quiescent at or above surface_depth; spans data gaps
        4 parking    - assigned downstream
        5 inflection - depth extremum within an ascent<->descent transition
        6 propelled  - quiescent below surface_depth with stable depth; min duration enforced
        7 transition - between any two consecutive primary phases, or where criteria disagree

    Parameters
    ----------
    df : polars.DataFrame
        Dataframe containing 'smooth_grad', 'INTERP_{depth_col}', 'depth_stable',
        'pitch_flat' and 'index' columns (all produced upstream in find_profiles).
    depth_col : str
        Name of the depth column; used to form the 'INTERP_{depth_col}' column name.
    gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold].
    cust_gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold] for the combined
        pitch*velocity criterion. Only active when 'pit_vel' is present on df.
    surface_depth : float
        Depth threshold (m) separating surfacing (3) from propelled (6).
    time_col : str
        Name of the timestamp column; used for propelled duration filtering.
    propelled_min_duration : str, default='1m'
        Minimum duration for a propelled segment to be retained; shorter segments
        revert to unknown (0). Polars duration format (e.g. '30s', '5m').

    Returns
    -------
    polars.DataFrame
        Input dataframe with an additional integer 'PHASE_SCI' column.
    """
    interp_depth = f"INTERP_{depth_col}"
    pos_grad, neg_grad = gradient_thresholds
    pos_cust_grad, neg_cust_grad = cust_gradient_thresholds
    at_surface = pl.col(interp_depth) <= surface_depth

    # Determine motion state from velocity and (optionally) pitch*velocity criteria
    if "pit_vel" in df.columns:
        vel_descent = pl.col("smooth_grad") > pos_grad
        vel_ascent  = pl.col("smooth_grad") < neg_grad
        vel_motion  = vel_descent | vel_ascent
        pit_motion  = pl.col("pit_vel") > pos_cust_grad
        descent  = vel_descent & pit_motion  # both criteria confirm descent
        ascent   = vel_ascent  & pit_motion  # both criteria confirm ascent
        disagree = vel_motion  ^ pit_motion  # one criterion fires, the other does not
    else:
        descent  = pl.col("smooth_grad") > pos_grad
        ascent   = pl.col("smooth_grad") < neg_grad
        disagree = pl.lit(False)

    vertical_motion = descent | ascent

    # Surfacing quiescence spans data gaps: pitch_flat and proximity to surface are sufficient
    # Propelled quiescence requires confirmed depth stability below the surface threshold
    quiescent_surface   = pl.col("depth_stable") & at_surface
    quiescent_propelled = pl.col("depth_stable") & ~vertical_motion & ~at_surface

    # Primary phase assignment: disagreement-based transitions applied first,
    # then directed motion, then quiescent states; unclassified points default to unknown (0)
    df = df.with_columns(
        pl.when(disagree).then(pl.lit(7))
        .when(descent).then(pl.lit(2))
        .when(ascent).then(pl.lit(1))
        .when(quiescent_surface).then(pl.lit(3))
        .when(quiescent_propelled).then(pl.lit(6))
        .otherwise(pl.lit(0))
        .cast(pl.Int64)
        .alias("PHASE_SCI")
    )

    # Fallback: insert transitions between primary phases and mark inflections
    phases = df["PHASE_SCI"].to_numpy().copy()
    depths = df[interp_depth].to_numpy()
    phases = _resolve_transitions(phases, depths, surface_depth)

    # Merge surfacing (3) segments separated only by transitions (7) into continuous surfacing
    n = len(phases)
    i = 0
    while i < n:
        if phases[i] == 7:
            j = i + 1
            while j < n and phases[j] == 7:
                j += 1
            prev_p = next((int(phases[p]) for p in range(i - 1, -1, -1) if phases[p] != 7), None)
            next_p = next((int(phases[p]) for p in range(j, n) if phases[p] != 7), None)
            if prev_p == 3 and next_p == 3:
                phases[i:j] = 3
            i = j
        else:
            i += 1

    # Second pass: re-enforce strict transitions after the surface merge may have
    # exposed surfacing→primary adjacencies suppressed in the first pass
    phases = _resolve_transitions(phases, depths, surface_depth, steal_from_surface=True)
    df = df.with_columns(pl.Series("PHASE_SCI", phases))

    # Enforce minimum propelled duration: short segments revert to unknown (0)
    df = df.with_columns(
        (
            (pl.col("PHASE_SCI") == 6).cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * (pl.col("PHASE_SCI") == 6) - 1
        ).alias("_prop_seg")
    )
    prop_durations = (
        df.filter(pl.col("_prop_seg") >= 0)
          .group_by("_prop_seg")
          .agg((pl.col(time_col).max() - pl.col(time_col).min()).alias("_prop_elapsed"))
    )
    df = (
        df.join(prop_durations, on="_prop_seg", how="left")
          .with_columns(
              pl.when(
                  (pl.col("PHASE_SCI") == 6) &
                  (pl.col("_prop_elapsed") < pl.duration(**_parse_duration_to_kwargs(propelled_min_duration)))
              )
              .then(pl.lit(0))
              .otherwise(pl.col("PHASE_SCI"))
              .cast(pl.Int64)
              .alias("PHASE_SCI")
          ).drop(["_prop_seg", "_prop_elapsed"])
    )

    return df


def _extract_profiles(phases):
    """Derive profile numbers and directions from a PHASE_SCI array.

    A profile spans a core ascent (1) or descent (2) segment and its bounding
    transitions (7). Where an inflection (5) separates consecutive profiles, the
    inflection and any following transitions belong to the forthcoming profile;
    the preceding transitions belong to the prior profile.

    Parameters
    ----------
    phases : np.ndarray
        Integer PHASE_SCI array.

    Returns
    -------
    profile_num : np.ndarray
        Profile index for each point (-1 for non-profile points).
    profile_dir : np.ndarray
        Profile direction: 1 for ascent, -1 for descent, 0 for non-profile.
    """
    CORE = {1, 2}
    n = len(phases)
    profile_num = np.full(n, -1, dtype=float)
    profile_dir = np.zeros(n, dtype=float)

    # Locate contiguous ascent/descent core segments
    cores = []
    i = 0
    while i < n:
        if int(phases[i]) in CORE:
            p = int(phases[i])
            j = i + 1
            while j < n and phases[j] == p:
                j += 1
            cores.append((i, j - 1, p))
            i = j
        else:
            i += 1

    for prof_idx, (core_start, core_end, core_phase) in enumerate(cores):
        direction = 1 if core_phase == 1 else -1

        # Backward extension: include transitions (7); stop at inflection (5, inclusive).
        # Points before the inflection belong to the prior profile's trailing transition.
        seg_start = core_start
        k = core_start - 1
        while k >= 0 and int(phases[k]) in {5, 7}:
            seg_start = k
            if int(phases[k]) == 5:
                break  # Inflection belongs to this profile; points before it do not
            k -= 1

        # Forward extension: include transitions (7) only; stops naturally before any
        # inflection (5) or primary phase, both of which belong to the next profile
        seg_end = core_end
        k = core_end + 1
        while k < n and int(phases[k]) == 7:
            seg_end = k
            k += 1

        profile_num[seg_start : seg_end + 1] = prof_idx
        profile_dir[seg_start : seg_end + 1] = direction

    return profile_num, profile_dir


def find_profiles(
    df,
    gradient_thresholds: list = [0.05, -0.05],
    filter_win_sizes=["20s", "10s"],
    time_col="TIME",
    depth_col="DEPTH",
    transect_duration="10m",
    transect_depth_range=[10.0],
    transect_depth_bottom_limits=None,
    cust_col=None,
    cust_gradient_thresholds=[0.005, -0.005],
    surface_depth=2.5,
    propelled_min_duration="1m",
):
    """
    Identifies vertical profiles in oceanographic or similar data by analyzing depth gradients over time.

    This function processes depth-time data to identify periods where an instrument is performing
    vertical profiling. Phase classification (PHASE_SCI) is computed first from smoothed velocity
    and optional pitch signals; profile numbers and directions are then derived directly from the
    phase sequence. Also detects transect segments where depth is stable and the platform is not
    profiling.

    Phase detection has two avenues, selected by `cust_col`:
    - velocity-only (cust_col=None): ascent/descent assigned when smoothed vertical velocity
      falls outside [neg_grad, pos_grad].
    - combined pitch and velocity (cust_col='pitch'): ascent/descent require both criteria to
      agree. Points where only one criterion fires are classified as transition (7), making
      detection more robust at profile turning points.

    A profile is any contiguous ascent (1) or descent (2) phase with its bounding transitions
    (7) on each side (i.e. a 7,1,7 or 7,2,7 sequence). Inflection points (5) at the turning
    depth between consecutive profiles are assigned to the forthcoming profile.

    Parameters
    ----------
    df : polars.DataFrame
        Input dataframe containing time and depth measurements
    gradient_thresholds : list, default=[0.05, -0.05]
        Two-element list [positive_threshold, negative_threshold] defining the vertical velocity
        range (in meters/second) that is NOT considered part of a profile.
    filter_win_sizes : list, default=['20s', '10s']
        Window sizes for the compound filter applied to gradient calculations, in Polars duration
        format. Index 0 controls the rolling median window size and index 1 controls the rolling
        mean window size.
    time_col : str, default='TIME'
        Name of the column containing timestamp data
    depth_col : str, default='DEPTH'
        Name of the column containing depth measurements
    transect_duration : str, default='10m'
        Minimum duration for a transect to be considered valid, in Polars duration format.
        Also used as the rolling window for depth stability detection.
    transect_depth_range : list, default=[10.0]
        Maximum depth variation (in meters) over the transect window for a segment to be
        considered a transect. If multiple values are given, depth-dependent thresholds are
        applied; transect_depth_bottom_limits must then also be provided.
    transect_depth_bottom_limits : list, default=None
        Depth boundaries (in meters) separating each transect_depth_range band. Required when
        len(transect_depth_range) > 1; must have length len(transect_depth_range) - 1.
    cust_col : str, default=None
        Name of an additional data column (e.g. 'pitch') to be used for phase classification
        and/or diagnostics plotting. When set to 'pitch', phase detection uses the combined
        -(pitch * velocity) criterion alongside velocity.
    cust_gradient_thresholds : list, default=[0.005, -0.005]
        Two-element list [positive_threshold, negative_threshold] for the combined-criterion
        phase detection (only used when cust_col='pitch').
    surface_depth : float, default=2.5
        Depth threshold (m). Quiescent points at or above this depth are classified as
        surfacing (PHASE_SCI=3) rather than propelled (PHASE_SCI=6).
    propelled_min_duration : str, default='1m'
        Minimum duration for a propelled (PHASE_SCI=6) segment to be retained; shorter
        segments revert to unknown (0). Polars duration format (e.g. '30s', '5m').

    Returns
    -------
    polars.DataFrame
        Dataframe with additional columns:
        - 'dt': Time difference between consecutive points (seconds)
        - 'dz': Depth difference between consecutive points (meters)
        - 'grad': Vertical velocity (dz/dt, meters/second)
        - 'smooth_grad': Filtered vertical velocity
        - 'PHASE_SCI': Integer phase code (0–7) classifying each point's platform behaviour
        - 'is_profile': Boolean indicating if a point belongs to a profile
        - 'profile_num': Unique identifier for each identified profile (-1 for non-profile points)
        - 'profile_dir': Profile direction (1=ascent, -1=descent, 0=non-profile)
        - 'is_transect': Boolean indicating if a point belongs to a transect
        - 'transect_num': Unique identifier for each identified transect (-1 for non-transect points)

    Notes
    -----
    - 'depth_col' does not strictly have to be a depth measurement; any variable which follows
      the profile shape (such as pressure) could also be used, though this would change the units
      and interpretation of grad.
    """

    # Get the unedited shape for padding later (to make the input and outputs the same length)
    df_full_len = len(df)

    # Interpolate missing depth values using time as reference
    # Also removes infinite and NaN values before interpolation
    interp_src_cols = [depth_col] + ([cust_col] if cust_col and cust_col in df.columns else [])
    df = (
        df.select(
            pl.col(time_col),
            pl.col(depth_col),
            *([pl.col(cust_col)] if cust_col and cust_col in df.columns else []),
            *(
                pl.col(c).replace([np.inf, -np.inf, np.nan], None)
                          .interpolate_by(time_col)
                          .name.prefix("INTERP_")
                for c in interp_src_cols
            ),
        )
        .with_row_index()
        .drop_nulls(subset=[f"INTERP_{c}" for c in interp_src_cols])
    )

    # Calculate time differences (dt) and depth differences (dz) between consecutive measurements
    df = df.with_columns(
        (pl.col(time_col).diff().cast(pl.Float64) * 1e-9).alias(
            "dt"
        ),  # Convert nanoseconds to seconds
        pl.col(f"INTERP_{depth_col}").diff().alias("dz"),
    )

    # Calculate vertical velocity (gradient) as depth change divided by time change
    df = df.with_columns(
        (pl.col("dz") / pl.col("dt")).alias("grad"),
    ).drop_nulls(subset="grad")

    # Apply a compound filter to smooth the gradient values (rolling median
    # supresses spikes, rolling mean smooths noise)
    # TODO: this window size should be checked against the maximum sample period (dt)
    smooth_cols = ["grad"] + ([f"INTERP_{cust_col}"] if cust_col else [])
    df = df.with_columns(
        pl.col(*smooth_cols)
        .rolling_median_by(time_col, window_size=filter_win_sizes[0])
        .rolling_mean_by(time_col, window_size=filter_win_sizes[1])
        .name.prefix("smooth_"),
    )

    # Centre the smoothed signal: shift each smoothed column backward by half the
    # combined filter window in rows, compensating for the group delay of the chained
    # trailing filters so smoothed[i] reflects data centred on row i (rather than ending
    # at i). Row count is derived from the median sample period.
    median_dt = df.select(pl.col("dt").median()).item()
    shift_rows = int(sum(_parse_duration_to_seconds(w) for w in filter_win_sizes) / (2 * median_dt))
    df = df.with_columns(
        pl.col(*[f"smooth_{c}" for c in smooth_cols]).shift(-shift_rows)
    )

    # Compute pit_vel and pitch_flat for downstream phase classification.
    # pitch_flat is always True when no pitch column is available.
    pos_grad, neg_grad = gradient_thresholds
    if cust_col and re.search('pitch', cust_col, re.IGNORECASE):
        cust_pos, cust_neg = cust_gradient_thresholds
        df = df.with_columns(
            (-pl.col(f"smooth_INTERP_{cust_col}") * pl.col("smooth_grad")).alias("pit_vel")
        )
        df = df.with_columns(
            (pl.col("pit_vel") < cust_pos).fill_null(False).alias("pitch_flat"),
        )
    else:
        df = df.with_columns(pl.lit(True).alias("pitch_flat"))

    # Rolling min and max of depth over the transect window duration for depth stability detection
    df = df.with_columns([
        pl.col(f"INTERP_{depth_col}")
          .rolling_min_by(time_col, window_size=transect_duration)
          .alias("roll_min_depth"),
        pl.col(f"INTERP_{depth_col}")
          .rolling_max_by(time_col, window_size=transect_duration)
          .alias("roll_max_depth"),
    ])

    # Check that depth stays within user-defined transect depth window
    ref_depth_range = transect_depth_range[0] if isinstance(transect_depth_range, (list, tuple)) else transect_depth_range
    df = df.with_columns(
        ((pl.col("roll_max_depth") - pl.col("roll_min_depth")) <= ref_depth_range)
        .alias("depth_stable")
    )

    # Assign science phase classification; profile/transect numbering is derived from this
    df = _assign_phase_sci(
        df, depth_col, gradient_thresholds, cust_gradient_thresholds,
        surface_depth, time_col, propelled_min_duration,
    )

    # Derive profile numbers and directions directly from PHASE_SCI
    profile_num_arr, profile_dir_arr = _extract_profiles(df["PHASE_SCI"].to_numpy())
    df = df.with_columns(
        pl.Series("profile_num", profile_num_arr),
        pl.Series("profile_dir", profile_dir_arr),
    )

    # Derive is_profile from profile_num for convenience
    df = df.with_columns(
        (pl.col("profile_num") >= 0).alias("is_profile"),
    )

    # Transect defined by phase 6 (quiescent below surface): PHASE_SCI is the source of truth
    df = df.with_columns(
        (pl.col("PHASE_SCI") == 6).alias("is_transect")
    )

    # Assign unique transect numbers to consecutive transect points
    df = df.with_columns(
        (
            pl.col("is_transect").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_transect")
            - 1
        ).alias("transect_num")
    )

    # Enforce minimum transect duration and maximum depth range
    # Compute per-transect statistics for filtering
    transect_durations_df = (
        df.filter(pl.col("transect_num") >= 0)
          .group_by("transect_num")
          .agg((pl.col(time_col).max() - pl.col(time_col).min()).alias("elapsed"))
    )
    transect_depths_df = (
        df.filter(pl.col("transect_num") >= 0)
          .group_by("transect_num")
          .agg((pl.col(f"INTERP_{depth_col}").max() - pl.col(f"INTERP_{depth_col}").min()).alias("depth_span"))
    )
    transect_means_df = (
        df.filter(pl.col("transect_num") >= 0)
          .group_by("transect_num")
          .agg(pl.col(f"INTERP_{depth_col}").mean().alias("mean_depth"))
    )
    df = (
        df.join(transect_durations_df, on="transect_num", how="left")
          .join(transect_depths_df, on="transect_num", how="left")
          .join(transect_means_df, on="transect_num", how="left")
    )

    # Build depth-dependent transect window (supports depth-layered thresholds)
    if len(transect_depth_range) > 1:
        if not transect_depth_bottom_limits:
            raise ValueError("transect_depth_bottom_limits must be provided when multiple transect_depth_range values are used.")
        if len(transect_depth_range) - 1 != len(transect_depth_bottom_limits):
            raise ValueError("transect_depth_bottom_limits must have length len(transect_depth_range) - 1")
        expr = None
        for i, rng in enumerate(transect_depth_range):
            if i == 0:
                cond = pl.col("mean_depth") <= transect_depth_bottom_limits[0]
            elif i == len(transect_depth_range) - 1:
                cond = pl.col("mean_depth") > transect_depth_bottom_limits[-1]
            else:
                cond = (
                    (pl.col("mean_depth") > transect_depth_bottom_limits[i - 1]) &
                    (pl.col("mean_depth") <= transect_depth_bottom_limits[i])
                )
            expr = pl.when(cond).then(rng) if expr is None else expr.when(cond).then(rng)
        expr = expr.otherwise(transect_depth_range[-1])
        df = df.with_columns(expr.alias("depth_window"))
    else:
        df = df.with_columns(pl.lit(transect_depth_range[0]).alias("depth_window"))

    df = df.with_columns(
        pl.when(
            (pl.col("elapsed") >= pl.duration(**_parse_duration_to_kwargs(transect_duration))) &
            (pl.col("depth_span") <= pl.col("depth_window"))
        )
        .then(pl.col("is_transect"))
        .otherwise(False)
        .alias("is_transect")
    ).drop(["elapsed", "depth_span", "mean_depth", "depth_window"])

    # Re-number transects after filtering (so IDs match valid segments)
    df = df.with_columns(
        (
            pl.col("is_transect").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_transect")
            - 1
        ).alias("transect_num")
    )

    # Reforming the full length dataframe (This executes faster than polars join or merge methods)
    front_pad = np.full((df["index"].min(), len(df.columns)), np.nan)
    end_pad = np.full((df_full_len - df["index"].max() - 1, len(df.columns)), np.nan)

    data = np.vstack((front_pad, df.to_numpy(), end_pad))
    padded_df = pl.DataFrame(data, schema=df.columns).drop("index")

    return padded_df


@register_step
class FindProfilesStep(BaseStep, QCHandlingMixin):

    step_name = "Find Profiles Pitch"
    required_variables = ["TIME"]
    provided_variables = ["PROFILE_NUMBER", "PROFILE_DIR", "PHASE_SCI"]

    def run(self):
        self.log("Attempting to designate profile numbers")

        self.filter_qc()

        # All parameters fall back to presets so the step can run with no config input
        self.thresholds = self.parameters.get("gradient_thresholds", [0.05, -0.05])
        self.cust_thresholds = self.parameters.get("custom_gradient_thresholds", [0.005, -0.005])
        self.win_sizes = self.parameters.get("filter_window_sizes", ["20s", "10s"])
        self.depth_col = self.parameters.get("depth_column", "PRES")
        self.cust_col = self.parameters.get("custom_column", None)
        self.surface_depth = self.parameters.get("surface_depth", 2.5)
        self.propelled_min_duration = self.parameters.get("propelled_min_duration", "1m")

        # Convert to polars for processing
        cols = ["TIME", self.depth_col] + ([self.cust_col] if self.cust_col else [])
        self._df = pl.from_pandas(
            self.data[cols].to_dataframe(), nan_to_null=False
        )
        self.profile_outputs = find_profiles(
            self._df, self.thresholds, self.win_sizes,
            depth_col=self.depth_col, cust_col=self.cust_col,
            cust_gradient_thresholds=self.cust_thresholds,
            surface_depth=self.surface_depth,
            propelled_min_duration=self.propelled_min_duration,
        )

        if self.diagnostics:
            self.log("Generating diagnostics")
            self.generate_diagnostics()

        profile_numbers = self.profile_outputs["profile_num"].to_numpy()

        # Add profile numbers to data and update context
        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), profile_numbers)
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. #=-1 indicates no profile, #>=0 are profiles.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": -1,
            "valid_max": np.inf,
        }

        # Generate QC for profile numbers
        self.generate_qc(
            {"PROFILE_NUMBER_QC": ["TIME_QC", f"{self.depth_col}_QC"]},
        )

        # Add profile direction to data
        profile_dir = self.profile_outputs["profile_dir"].to_numpy()
        self.data["PROFILE_DIR"] = (("N_MEASUREMENTS",), profile_dir)
        self.data.PROFILE_DIR.attrs = {
            "long_name": "Profile direction. 1=ascent, -1=descent, 0=not a profile.",
            "units": "None",
            "standard_name": "Profile Direction",
            "valid_min": -1,
            "valid_max": 1,
            "flag_values": [-1, 0, 1],
            "flag_meanings": "descent non_profile ascent",
        }

        # Generate QC for profile direction
        self.generate_qc(
            {"PROFILE_DIR_QC": ["TIME_QC", f"{self.depth_col}_QC"]},
        )

        # Add science phase classification to data
        phase_sci = self.profile_outputs["PHASE_SCI"].to_numpy()
        self.data["PHASE_SCI"] = (("N_MEASUREMENTS",), phase_sci)
        self.data.PHASE_SCI.attrs = {
            "long_name": "Science phase classification of platform behaviour.",
            "units": "None",
            "standard_name": "Phase",
            "valid_min": 0,
            "valid_max": 7,
            "flag_values": [0, 1, 2, 3, 4, 5, 6, 7],
            "flag_meanings": "unknown ascent descent surfacing parking inflection propelled transition",
        }

        # Generate QC for science phase
        self.generate_qc(
            {"PHASE_SCI_QC": ["TIME_QC", f"{self.depth_col}_QC"]},
        )

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        """Render a single static diagnostic figure, coloured by science phase.

        Three stacked panels share a time axis:
          0. interpolated depth, plotted as -depth so the ocean surface (0) sits at
             the top and increasing depth runs downward;
          1. vertical velocity with the velocity thresholds and, where pitch is used,
             the combined -(pitch x velocity) signal with its own pitch thresholds;
          2. derived profile number.
        """
        outputs = self.profile_outputs
        depth_col = self.depth_col

        # Padding/rebuild in find_profiles returns TIME as epoch-nanosecond floats;
        # recover datetimes (NaN -> NaT) so the date axis formatter behaves.
        time = outputs["TIME"].to_numpy().astype("datetime64[ns]")
        phase = outputs["PHASE_SCI"].to_numpy()
        neg_depth = -outputs[f"INTERP_{depth_col}"].to_numpy()
        grad = outputs["grad"].to_numpy()
        smooth_grad = outputs["smooth_grad"].to_numpy()
        profile_num = outputs["profile_num"].to_numpy()
        has_pit_vel = "pit_vel" in outputs.columns
        pit_vel = outputs["pit_vel"].to_numpy() if has_pit_vel else None

        fig, axs = plt.subplots(
            3, 1, figsize=(16, 9), height_ratios=[3, 3, 1], sharex=True,
        )

        # Panel 0: depth coloured by phase (surface at top, -depth descending)
        for p_val in range(8):
            mask = phase == p_val
            n_points = int(mask.sum())
            if n_points == 0:
                continue
            axs[0].plot(
                time[mask], neg_depth[mask],
                ls="none", marker=".", markersize=6,
                color=PHASE_COLOURS.get(p_val, "black"),
                label=f"{PHASE_NAMES.get(p_val, f'Phase {p_val}')} (n={n_points})",
                zorder=6 if p_val == 5 else 3,
            )
        axs[0].set_ylabel(f"-{depth_col} (surface at top)")
        axs[0].set_title("Find Profiles (Pitch) | Phase Mapping")
        leg = axs[0].legend(loc="upper right", fontsize=9, markerscale=2.0, ncol=2)
        leg.set_zorder(100)
        axs[0].grid(alpha=0.3)

        # Panel 1: vertical velocity coloured by phase, with velocity thresholds and
        # (when pitch is used) the combined pitch*velocity signal and its thresholds
        axs[1].plot(time, grad, c="k", alpha=0.1, lw=0.8, label="Raw velocity")
        for p_val in range(8):
            mask = phase == p_val
            if int(mask.sum()) == 0:
                continue
            axs[1].plot(
                time[mask], smooth_grad[mask],
                ls="none", marker=".", markersize=4,
                color=PHASE_COLOURS.get(p_val, "black"),
                zorder=6 if p_val == 5 else 3,
            )
        for val, label in zip(self.thresholds, ["Velocity threshold", None]):
            axs[1].axhline(val, ls="--", color="gray", lw=1, label=label)
        if has_pit_vel:
            axs[1].plot(
                time, pit_vel, c="purple", alpha=0.5, lw=0.8,
                label="-(pitch x velocity)",
            )
            for val, label in zip(self.cust_thresholds, ["Pitch threshold", None]):
                axs[1].axhline(val, ls=":", color="purple", lw=1, label=label)
        axs[1].set_ylabel("Vertical velocity (m/s)")
        axs[1].legend(loc="upper right", fontsize=9, ncol=2)
        axs[1].grid(alpha=0.3)

        # Panel 2: derived profile number
        axs[2].plot(
            time, profile_num,
            c="tab:blue", marker=".", markersize=3, ls="", label="Profile number",
        )
        axs[2].set_ylabel("Profile #")
        axs[2].set_xlabel("Time")
        axs[2].legend(loc="upper left")
        axs[2].grid(alpha=0.3)

        axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show(block=True)
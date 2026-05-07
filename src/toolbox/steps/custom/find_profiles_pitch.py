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
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
import numpy as np
import re


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


def _resolve_transitions(phases, depths):
    """Ensure a transition (7) exists between every consecutive pair of primary phases,
    then mark the depth extremum of each ascent<->descent transition as inflection (5).

    Parameters
    ----------
    phases : np.ndarray
        Integer phase array with primary phases (1, 2, 3, 6) assigned and zeros elsewhere.
    depths : np.ndarray
        Interpolated depth values aligned with phases.

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

    # Insert transition (7) between every consecutive pair of primary segments
    for k in range(len(segs) - 1):
        s0, e0, _ = segs[k]
        s1,  _,  _ = segs[k + 1]
        if s1 > e0 + 1:
            # Natural gap: points between the two segments become transition
            result[e0 + 1 : s1] = 7
        else:
            # Adjacent segments: steal the last min(3, segment_length) points of the previous
            steal_n = min(3, e0 - s0 + 1)
            result[e0 - steal_n + 1 : e0 + 1] = 7

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


def _assign_phase_sci(df, depth_col, gradient_thresholds, surface_depth):
    """Assign PHASE_SCI classification codes from smoothed signals.

    Primary phases are assigned first from signal conditions, then transitions (7)
    are inserted between every consecutive pair of primary phases, and inflections (5)
    are marked at the depth extremum of ascent<->descent transitions.

    Phase codes:
        0 unknown    - not between any two primary phases
        1 ascent     - platform climbing (smooth_grad < neg_grad)
        2 descent    - platform diving  (smooth_grad > pos_grad)
        3 surfacing  - stable depth at or above surface_depth
        4 parking    - assigned downstream
        5 inflection - depth extremum within an ascent<->descent transition
        6 propelled  - stable depth below surface_depth
        7 transition - between any two consecutive primary phases

    Parameters
    ----------
    df : polars.DataFrame
        Dataframe containing 'smooth_grad', 'INTERP_{depth_col}', 'depth_stable',
        'pitch_flat' and 'index' columns (all produced upstream in find_profiles).
    depth_col : str
        Name of the depth column; used to form the 'INTERP_{depth_col}' column name.
    gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold].
    surface_depth : float
        Depth threshold (m) separating surfacing (3) from propelled (6).

    Returns
    -------
    polars.DataFrame
        Input dataframe with an additional integer 'PHASE_SCI' column.
    """
    interp_depth = f"INTERP_{depth_col}"
    pos_grad, neg_grad = gradient_thresholds

    # Per-point signal predicates
    descent = pl.col("smooth_grad") > pos_grad
    ascent = pl.col("smooth_grad") < neg_grad
    vertical_motion = descent | ascent
    at_surface = pl.col(interp_depth) <= surface_depth
    quiescent = pl.col("depth_stable") & pl.col("pitch_flat") & ~vertical_motion

    # Assign primary phases; unclassified points default to unknown (0)
    df = df.with_columns(
        pl.when(descent).then(pl.lit(2))
        .when(ascent).then(pl.lit(1))
        .when(quiescent & at_surface).then(pl.lit(3))
        .when(quiescent).then(pl.lit(6))
        .otherwise(pl.lit(0))
        .cast(pl.Int64)
        .alias("PHASE_SCI")
    )

    # Resolve transitions (7) and inflections (5) from the primary phase array
    phases = df["PHASE_SCI"].to_numpy().copy()
    depths = df[interp_depth].to_numpy()
    df = df.with_columns(pl.Series("PHASE_SCI", _resolve_transitions(phases, depths)))

    return df


def find_profiles(
    df,
    gradient_thresholds: list,
    filter_win_sizes=["20s", "10s"],
    time_col="TIME",
    depth_col="DEPTH",
    profile_duration="2m",
    transect_duration="10m",
    transect_depth_range=[10.0],
    transect_depth_bottom_limits=None,
    cust_col=None,
    cust_gradient_thresholds=[0.005, -0.025],
    surface_depth=2.5,
):
    """
    Identifies vertical profiles in oceanographic or similar data by analyzing depth gradients over time.

    This function processes depth-time data to identify periods where an instrument is performing
    vertical profiling based on gradient thresholds. It handles data interpolation, calculates vertical
    velocities, applies median filtering, and assigns unique profile numbers to identified profiles.
    Also detects transect segments where depth is stable and the platform is not profiling, and
    classifies each point's platform behaviour into a science phase code (PHASE_SCI).

    Profile detection has two avenues, selected by `cust_col`:
    - velocity-only (cust_col=None): a point is part of a profile when its smoothed vertical
      velocity falls outside [neg_grad, pos_grad].
    - combined pitch and velocity (cust_col='pitch'): a point is part of a profile when the
      smoothed -(pitch * velocity) product falls outside [cust_neg, cust_pos]. This is more
      robust against missed turns where vertical velocity briefly crosses zero.

    The PHASE_SCI classification is computed independently from profile/transect numbering,
    directly from the smoothed depth, velocity and (optional) pitch signals.

    Parameters
    ----------
    df : polars.DataFrame
        Input dataframe containing time and depth measurements
    gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold] defining the vertical velocity
        range (in meters/second) that is NOT considered part of a profile. Typical values are
        around [0.02, -0.02]
    filter_win_sizes : list, default=['20s', '10s']
        Window sizes for the compound filter applied to gradient calculations, in Polars duration
        format. Index 0 controls the rolling median window size and index 1 controls the rolling
        mean window size.
    time_col : str, default='TIME'
        Name of the column containing timestamp data
    depth_col : str, default='DEPTH'
        Name of the column containing depth measurements
    profile_duration : str, default='2m'
        Minimum duration for a profile to be considered valid, in Polars duration format
        (e.g. '30s', '5m', '1h').
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
        Name of an additional data column (e.g. 'pitch') to be used for profile detection
        and/or diagnostics plotting. When set to 'pitch', profile detection uses the combined
        -(pitch * velocity) criterion instead of velocity alone.
    cust_gradient_thresholds : list, default=[0.005, -0.025]
        Two-element list [positive_threshold, negative_threshold] for the combined-criterion
        profile detection (only used when cust_col='pitch').
    surface_depth : float, default=2.5
        Depth threshold (m). Stable-depth points at or above this depth are classified as
        surfacing (PHASE_SCI=3) rather than propelled (PHASE_SCI=6).

    Returns
    -------
    polars.DataFrame
        Dataframe with additional columns:
        - 'dt': Time difference between consecutive points (seconds)
        - 'dz': Depth difference between consecutive points (meters)
        - 'grad': Vertical velocity (dz/dt, meters/second)
        - 'smooth_grad': Filtered vertical velocity
        - 'is_profile': Boolean indicating if a point belongs to a profile
        - 'profile_num': Unique identifier for each identified profile (-1 for non-profile points)
        - 'is_transect': Boolean indicating if a point belongs to a transect
        - 'transect_num': Unique identifier for each identified transect (-1 for non-transect points)
        - 'PHASE_SCI': Integer phase code (0–7) classifying each point's platform behaviour.

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
    interp_src_cols = [depth_col] + ([cust_col] if cust_col in df.columns else [])
    df = (
        df.select(
            pl.col(time_col),
            pl.col(depth_col),
            *([pl.col(cust_col)] if cust_col in df.columns else []),
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

    # Profile detection: combined pitch/velocity criterion when pitch is available, velocity-only
    # otherwise. pitch_flat is materialised here so it can be reused for transect detection and
    # phase classification (always True when no pitch is available).
    pos_grad, neg_grad = gradient_thresholds
    if re.search('pitch', cust_col, re.IGNORECASE):
        cust_pos, cust_neg = cust_gradient_thresholds
        df = df.with_columns(
            (-pl.col(f"smooth_INTERP_{cust_col}") * pl.col("smooth_grad")).alias("pit_vel")
        )
        df = df.with_columns(
            pl.col("pit_vel").is_between(cust_neg, cust_pos).not_().fill_null(False).alias("is_profile"),
            (pl.col("pit_vel") < cust_pos).fill_null(False).alias("pitch_flat"),
        )
    else:
        df = df.with_columns(
            pl.col("smooth_grad").is_between(neg_grad, pos_grad).not_().fill_null(False).alias("is_profile"),
            pl.lit(True).alias("pitch_flat"),
        )

    # Assign unique profile numbers to consecutive points identified as profiles
    # This converts the boolean 'is_profile' column into numbered profile segments
    df = df.with_columns(
        (
            pl.col("is_profile").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_profile")
            - 1
        ).alias("profile_num")
    )

    # Enforce minimum profile duration
    # Compute elapsed time per profile segment and mask out short profiles
    profile_durations = (
        df.filter(pl.col("profile_num") >= 0)
          .group_by("profile_num")
          .agg((pl.col(time_col).max() - pl.col(time_col).min()).alias("elapsed"))
    )
    df = (
        df.join(profile_durations, on="profile_num", how="left")
          .with_columns(
              pl.when(pl.col("elapsed") >= pl.duration(**_parse_duration_to_kwargs(profile_duration)))
              .then(pl.col("is_profile"))
              .otherwise(False)
              .alias("is_profile")
          ).drop("elapsed")
    )

    # Re-number profiles after duration filter (so IDs match valid segments)
    df = df.with_columns(
        (
            pl.col("is_profile").cast(pl.Int64).diff().replace(-1, 0).cum_sum()
            * pl.col("is_profile")
            - 1
        ).alias("profile_num")
    )

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

    # Transect defined as stable depth, flat pitch, and not a profile
    df = df.with_columns(
        (pl.col("depth_stable") & pl.col("pitch_flat") & ~pl.col("is_profile"))
        .alias("is_transect")
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

    # Assign science phase classification (independent of profile/transect numbering)
    df = _assign_phase_sci(df, depth_col, gradient_thresholds, surface_depth)

    # Reforming the full length dataframe (This executes faster than polars join or merge methods)
    front_pad = np.full((df["index"].min(), len(df.columns)), np.nan)
    end_pad = np.full((df_full_len - df["index"].max() - 1, len(df.columns)), np.nan)

    data = np.vstack((front_pad, df.to_numpy(), end_pad))
    padded_df = pl.DataFrame(data, schema=df.columns).drop("index")

    return padded_df


@register_step
class FindProfilesStep(BaseStep, QCHandlingMixin):

    step_name = "Find Profiles"
    required_variables = ["TIME"]
    provided_variables = ["PROFILE_NUMBER", "PHASE_SCI"]

    def run(self):
        self.log("Attempting to designate profile numbers")

        self.filter_qc()

        self.thresholds = self.parameters["gradient_thresholds"]
        self.win_sizes = self.parameters["filter_window_sizes"]
        self.depth_col = self.parameters["depth_column"]
        self.cust_col = self.parameters.get("custom_column", None)
        self.surface_depth = self.parameters.get("surface_depth", 2.5)

        if self.diagnostics:
            self.log("Generating diagnostics")
            root = self.generate_diagnostics()
            root.mainloop()

        # Convert to polars for processing
        cols = ["TIME", self.depth_col] + ([self.cust_col] if self.cust_col else [])
        self._df = pl.from_pandas(
            self.data[cols].to_dataframe(), nan_to_null=False
        )
        self.profile_outputs = find_profiles(
            self._df, self.thresholds, self.win_sizes,
            depth_col=self.depth_col, cust_col=self.cust_col,
            surface_depth=self.surface_depth,
        )
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

        def generate_plot():
            # Allows interactive plots
            mpl.use("TkAgg")

            # Update data for plot
            cols = ["TIME", self.depth_col] + ([self.cust_col] if self.cust_col else [])
            self._df = pl.from_pandas(
                self.data[cols].to_dataframe(), nan_to_null=False
            )
            self.profile_outputs = find_profiles(
                self._df, self.thresholds, self.win_sizes,
                depth_col=self.depth_col, cust_col=self.cust_col,
                surface_depth=self.surface_depth,
            )

            # Split data into profile and non-profile points for plotting
            profiles = self.profile_outputs.filter(
                pl.col("is_profile").cast(pl.Boolean)
            )
            not_profiles = self.profile_outputs.filter(
                pl.col("is_profile").cast(pl.Boolean).not_()
            )

            n_rows = 4 if self.cust_col else 3
            height_ratios = [3, 3, 1, 2] if self.cust_col else [3, 3, 1]
            fig, axs = plt.subplots(
                n_rows, 1, figsize=(18, 8 + (2 if self.cust_col else 0)),
                height_ratios=height_ratios, sharex=True,
            )
            axs[0].set(xlabel="Time", ylabel="Interpolated Depth")
            axs[1].set(xlabel="Time", ylabel="Vertical Velocity")
            axs[2].set(xlabel="Time", ylabel="Profile Number")
            if self.cust_col:
                axs[3].set(xlabel="Time", ylabel=self.cust_col)
            fig.tight_layout()

            # Plot depth vs time, highlighting profile and non-profile points
            for data, col, label in zip(
                [profiles, not_profiles],
                ["tab:blue", "tab:red"],
                ["Profile", "Not Profile"],
            ):
                axs[0].plot(
                    data["TIME"],
                    -data[f"INTERP_{self.depth_col}"],
                    marker=".",
                    markersize=1,
                    ls="",
                    c=col,
                    label=label,
                )
                axs[1].plot(
                    data["TIME"],
                    data["smooth_grad"],
                    marker=".",
                    markersize=1,
                    ls="",
                    c=col,
                    label=label,
                )

            # Plot raw and smoothed gradients with threshold lines
            axs[1].plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["grad"],
                c="k",
                alpha=0.1,
                label="Raw Velocity",
            )
            for val, label in zip(self.thresholds, ["Gradient Thresholds", None]):
                axs[1].axhline(val, ls="--", color="gray", label=label)

            # Plot profile numbers
            axs[2].plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["profile_num"],
                c="gray",
            )

            # Plot custom column if provided
            if self.cust_col and f"INTERP_{self.cust_col}" in self.profile_outputs.columns:
                axs[3].plot(
                    self.profile_outputs["TIME"],
                    self.profile_outputs[f"INTERP_{self.cust_col}"],
                    c="purple",
                    marker=".",
                    markersize=1,
                    ls="",
                    label=self.cust_col,
                )
                axs[3].legend(loc="upper right")

            for ax in axs[:2]:
                ax.legend(loc="upper right")
            plt.show(block=False)

        root = tk.Tk()
        root.title("Parameter Adjustment")
        root.geometry(f"380x{50*len(self.parameters)}")
        entries = {}

        # Gradient thresholds
        row = 0
        values = self.thresholds
        tk.Label(root, text=f"Gradient Thresholds:").grid(row=row, column=0)
        for i, label, value in zip(range(2), ["+ve", "-ve"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        # Filter window sizes
        row = 2
        values = self.win_sizes
        tk.Label(root, text=f"Filter Window Sizes:").grid(
            row=row, column=0, pady=(20, 0)
        )
        for i, label, value in zip(range(2), ["Median Filter", "Mean Filter"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        # Depth column name
        row = 4
        value = self.depth_col
        tk.Label(root, text=f"Depth column name:").grid(row=row, column=0, pady=(20, 0))
        entry = tk.Entry(root, textvariable="depth_column")
        entry.insert(0, value)
        entry.grid(row=row, column=1, pady=(20, 0))
        entries["depth_column"] = entry

        def on_cancel():
            plt.close('all')
            root.quit()  # Stops the mainloop
            root.destroy()  # Destroys the window

        def on_regenerate():
            # Update parameter attributes
            self.thresholds = [float(entries["+ve"].get()), float(entries["-ve"].get())]
            self.win_sizes = [
                entries["Median Filter"].get(),
                entries["Mean Filter"].get(),
            ]
            self.depth_col = entries["depth_column"].get()

            # Regenerate data and plot it
            plt.close('all')
            generate_plot()

        def on_save():
            self.log(
                f"continuing with parameters: \n"
                f"  Gradient Thresholds: {self.thresholds}\n"
                f"  Filter Window Sizes: {self.win_sizes}\n"
                f"  Depth column: {self.depth_col}\n"
            )
            plt.close('all')
            root.quit()  # Stops the mainloop
            root.destroy()  # Destroys the window

        tk.Button(root, text="Regenerate", command=on_regenerate).grid(
            row=row + 1, column=0, pady=(20, 0)
        )
        tk.Button(root, text="Save", command=on_save).grid(
            row=row + 1, column=1, pady=(20, 0)
        )
        tk.Button(root, text="Cancel", command=on_cancel).grid(
            row=row + 1, column=2, pady=(20, 0)
        )

        generate_plot()
        return root
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

"""Class definition for finding vertical and horizontal profiles in depth data."""

from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE_COLOURS = {
    0: "#9ca3af",  
    1: "#22c55e",  
    2: "#3b82f6",  
    3: "#f97316",  
    4: "#a855f7",  
    5: "#06b6d4",  
    6: "#ef4444",  
    7: "#eab308",  
    8: "#ec4899",  
    9: "#84cc16",  
}

PHASE_NAMES = {
    0: "0 Unknown", 
    1: "1 Ascent", 
    2: "2 Descent", 
    3: "3 Surfacing",
    4: "4 Parking",
    5: "5 Inflection",
    6: "6 Propelled",
    7: "7 Transition"
}

# ---------------------------------------------------------------------------
# Core Processing Logic
# ---------------------------------------------------------------------------

def find_profiles(df_raw, depth_col, time_window_seconds, target_transect_phase, 
                  velocity_threshold, acceleration_threshold, transition_buffer_seconds, 
                  min_duration_minutes, peak_prominence, min_samples_between_peaks, 
                  gap_threshold_minutes, surface_depth, surfacing_threshold,
                  parking_gradient_threshold):
    """
    Identifies and classifies vertical and horizontal profiles from depth-time data.
    Maps scientific phases and derives continuous cycle numbers.
    """
    
    df = df_raw.dropna(subset=["TIME", depth_col]).sort_values("TIME")
    df = df[df[depth_col] != 0].copy()
    df = df.drop_duplicates(subset=["TIME"]).reset_index(drop=True)

    if df.empty:
        df_raw["PROFILE_NUMBER"] = np.nan
        df_raw["PROFILE_DIRECTION"] = np.nan
        df_raw["GRADIENT"] = np.nan
        df_raw["CYCLE"] = 1
        df_raw["SCI_PHASE"] = 0
        return df_raw

    time_window_str = f"{time_window_seconds}s"

    df.set_index("TIME", inplace=True)
    df = df.resample(time_window_str).mean().dropna(subset=[depth_col])
    df.reset_index(inplace=True)

    if df.empty:
        df_raw["PROFILE_NUMBER"] = np.nan
        df_raw["PROFILE_DIRECTION"] = np.nan
        df_raw["GRADIENT"] = np.nan
        df_raw["CYCLE"] = 1
        df_raw["SCI_PHASE"] = 0
        return df_raw

    time_seconds = df["TIME"].to_numpy().astype("int64") / 1e9
    depth = df[depth_col].values
    
    df.set_index("TIME", inplace=True)
    
    smoothed_depth = df[depth_col].rolling(time_window_str, center=True, min_periods=1).mean().values
    raw_velocity = np.gradient(smoothed_depth, time_seconds)
    
    despiked_velocity = pd.Series(raw_velocity, index=df.index).rolling(time_window_str, center=True, min_periods=1).median()
    smoothed_velocity = despiked_velocity.rolling("15s", center=True, min_periods=1).mean().values
    
    raw_acceleration = np.gradient(smoothed_velocity, time_seconds)
    despiked_acceleration = pd.Series(raw_acceleration, index=df.index).rolling(time_window_str, center=True, min_periods=1).median()
    smoothed_acceleration = despiked_acceleration.rolling("15s", center=True, min_periods=1).mean().values
    
    df.reset_index(inplace=True)

    raw_phases = np.zeros(len(df), dtype=int)
    raw_phases[smoothed_velocity > velocity_threshold] = 2
    raw_phases[smoothed_velocity < -velocity_threshold] = 1
    
    transect_mask = (raw_phases == 0) & (np.abs(smoothed_acceleration) <= acceleration_threshold)
    raw_phases[transect_mask] = target_transect_phase

    phases = np.zeros(len(df), dtype=int)
    
    for p_val in [1, 2, target_transect_phase]:
        mask = (raw_phases == p_val)
        padded = np.concatenate(([False], mask, [False]))
        starts = np.where(padded[1:] & ~padded[:-1])[0]
        ends = np.where(~padded[1:] & padded[:-1])[0]
        
        for s, e in zip(starts, ends):
            start_time = time_seconds[s]
            end_time = time_seconds[e - 1]
            block_duration = end_time - start_time
            
            if block_duration < (min_duration_minutes * 60):
                continue
            
            actual_trim = min(transition_buffer_seconds, block_duration / 3)
            
            trim_s = s
            while trim_s < e and (time_seconds[trim_s] - start_time) <= actual_trim:
                trim_s += 1
                
            trim_e = e - 1
            while trim_e >= s and (end_time - time_seconds[trim_e]) <= actual_trim:
                trim_e -= 1
                
            if trim_s <= trim_e:
                phases[trim_s:trim_e + 1] = p_val

    deep_peaks, _ = find_peaks(
        depth, 
        prominence=peak_prominence, 
        distance=min_samples_between_peaks
    )

    shallow_peaks, _ = find_peaks(
        -depth, 
        prominence=peak_prominence, 
        distance=min_samples_between_peaks
    )
    
    gap_mask = df["TIME"].diff() > pd.Timedelta(minutes=gap_threshold_minutes)
    chunk_ids = gap_mask.cumsum()
    
    extra_peaks = []
    for _, chunk in df.groupby(chunk_ids):
        if chunk.empty: 
            continue
            
        min_idx = chunk[depth_col].idxmin()
        if chunk.loc[min_idx, depth_col] <= surface_depth:
            extra_peaks.append(min_idx)
            
        max_idx = chunk[depth_col].idxmax()
        if chunk.loc[max_idx, depth_col] > surface_depth:
            extra_peaks.append(max_idx)
    
    all_peaks = np.unique(np.concatenate((deep_peaks, shallow_peaks, extra_peaks))).astype(int)
    valid_peaks = [p for p in all_peaks if phases[p] != target_transect_phase]
    
    transect_inflections = []
    padded_t = np.concatenate(([False], phases == target_transect_phase, [False]))
    t_starts = np.where(padded_t[1:] & ~padded_t[:-1])[0]
    t_ends = np.where(~padded_t[1:] & padded_t[:-1])[0] - 1
    
    for s, e in zip(t_starts, t_ends):
        idx = s - 1
        while idx >= 0 and phases[idx] not in [1, 2]:
            idx -= 1
            
        if idx >= 0:
            gap = depth[idx:s+1]
            infl_idx = idx + (np.argmax(gap) if phases[idx] == 2 else np.argmin(gap))
            transect_inflections.append(infl_idx)
            
        idx = e + 1
        while idx < len(phases) and phases[idx] not in [1, 2]:
            idx += 1
            
        if idx < len(phases):
            gap = depth[e:idx+1]
            infl_idx = e + (np.argmax(gap) if phases[idx] == 1 else np.argmin(gap))
            transect_inflections.append(infl_idx)

    all_inflections = np.unique(np.concatenate((valid_peaks, transect_inflections))).astype(int)
    phases[all_inflections] = 5

    shallow_mask = (depth <= surfacing_threshold) & (np.isin(phases, [5, target_transect_phase]))
    phases[shallow_mask] = 3

    padded_zeros = np.concatenate(([False], phases == 0, [False]))
    zero_starts = np.where(padded_zeros[1:] & ~padded_zeros[:-1])[0]
    zero_ends = np.where(~padded_zeros[1:] & padded_zeros[:-1])[0] - 1

    for s, e in zip(zero_starts, zero_ends):
        left_val = phases[s - 1] if s > 0 else None
        right_val = phases[e + 1] if e < len(phases) - 1 else None
        
        if left_val is not None and right_val is not None:
            if left_val == right_val:
                phases[s:e+1] = left_val
            else:
                phases[s:e+1] = 7
        else:
            phases[s:e+1] = 7

    # --- Convert drifting parking regions to ascent/descent ---
    parking_mask = np.isin(phases, [4, 6])
    padded_parking = np.concatenate(([False], parking_mask, [False]))
    p_starts = np.where(padded_parking[1:] & ~padded_parking[:-1])[0]
    p_ends = np.where(~padded_parking[1:] & padded_parking[:-1])[0]

    for s, e in zip(p_starts, p_ends):
        if (e - s) < 2:
            continue
            
        t_sec = time_seconds[s:e]
        z = depth[s:e]
        
        # Fit a linear trend to check the gradient over the parking block
        m, _ = np.polyfit(t_sec - t_sec[0], z, 1)
        
        if abs(m) > parking_gradient_threshold:
            # Overwrite the phase: 2 (Descent) if gradient is positive (going down), 1 (Ascent) if negative
            phases[s:e] = 2 if m > 0 else 1

    df["PHASE"] = phases

    df_merge = df[["TIME", "PHASE"]].copy()
    df_merge["BIN_TIME"] = df_merge["TIME"]

    mapped_df = pd.merge_asof(
        df_raw.sort_values("TIME"),
        df_merge.sort_values("TIME"),
        on="TIME",
        direction="nearest"
    )
    
    mapped_df["PHASE"] = mapped_df["PHASE"].fillna(7).astype(int)

    inflection_times = df.loc[df["PHASE"] == 5, "TIME"]
    mapped_df.loc[mapped_df["PHASE"] == 5, "PHASE"] = 7 
    
    for t in inflection_times:
        mapped_mask = mapped_df["BIN_TIME"] == t
        if not mapped_mask.any():
            continue
            
        idx = df.index[df["TIME"] == t][0]
        curr_d = df.loc[idx, depth_col]
        
        d_prev = df.loc[idx - 1, depth_col] if idx > 0 else curr_d
        d_next = df.loc[idx + 1, depth_col] if idx < len(df) - 1 else curr_d
        
        raw_subset = mapped_df[mapped_mask]
        
        if curr_d >= (d_prev + d_next) / 2:
            extreme_idx = raw_subset[depth_col].idxmax()
        else:
            extreme_idx = raw_subset[depth_col].idxmin()
            
        mapped_df.loc[extreme_idx, "PHASE"] = 5

    mapped_df.drop(columns=["BIN_TIME"], inplace=True)
    mapped_df["SCI_PHASE"] = mapped_df["PHASE"]

    # --- Generate Profile Number, Direction, Cycle, and Gradient ---

    phases_arr = mapped_df["SCI_PHASE"].to_numpy()
    n = len(phases_arr)

    # Direction: -1 ascent, 1 descent, 0 transect-like, NaN otherwise
    direction = np.full(n, np.nan)
    direction[phases_arr == 1] = -1
    direction[phases_arr == 2] = 1
    direction[np.isin(phases_arr, [3, 4, 6])] = 0
    mapped_df["PROFILE_DIRECTION"] = direction

    # Profile number: each ascent/descent core block defines a profile, extended
    # to include adjacent transects up to inflection/surfacing boundaries (or
    # the midpoint of the inter-core span when no boundary marker is present).
    core_mask = np.isin(phases_arr, [1, 2])
    padded_core = np.concatenate(([False], core_mask, [False]))
    c_starts = np.where(padded_core[1:] & ~padded_core[:-1])[0]
    c_ends = np.where(~padded_core[1:] & padded_core[:-1])[0]  # exclusive
    core_blocks = list(zip(c_starts, c_ends))

    profile_num = np.full(n, np.nan)

    if core_blocks:
        boundaries = []  # inclusive last-index of each profile (except last)
        for i in range(len(core_blocks) - 1):
            end_i = core_blocks[i][1]
            start_next = core_blocks[i + 1][0]
            if start_next <= end_i:
                boundaries.append(end_i - 1)
                continue

            region = np.arange(end_i, start_next)
            region_phases = phases_arr[region]
            infl = region[region_phases == 5]
            surf = region[region_phases == 3]
            if len(infl) > 0:
                split = int(infl[-1])
            elif len(surf) > 0:
                split = int(surf[-1])
            else:
                split = (end_i + start_next - 1) // 2
            boundaries.append(split)

        prev_end = 0
        for k in range(len(core_blocks)):
            this_end = boundaries[k] + 1 if k < len(boundaries) else n
            profile_num[prev_end:this_end] = k + 1
            prev_end = this_end

    # Surfacing rows are part of the cycle but not part of any profile.
    profile_num[phases_arr == 3] = np.nan
    mapped_df["PROFILE_NUMBER"] = profile_num

    surf_mask = mapped_df["SCI_PHASE"] == 3
    down_mask = mapped_df["SCI_PHASE"] == 2
    state_subset = mapped_df.loc[surf_mask | down_mask]
    is_new_cycle = (state_subset["SCI_PHASE"] == 2) & (state_subset["SCI_PHASE"].shift(1) == 3)

    cycle_trigger = pd.Series(0, index=mapped_df.index)
    cycle_trigger.loc[state_subset[is_new_cycle].index] = 1
    mapped_df["CYCLE"] = cycle_trigger.cumsum() + 1

    # Gradient: per-profile linear fit of depth vs time over the ascent/descent
    # core rows only (transects would dilute the slope).
    mapped_df["GRADIENT"] = np.nan
    core_series = pd.Series(core_mask, index=mapped_df.index)
    core_rows = mapped_df[core_series & mapped_df["PROFILE_NUMBER"].notna()]
    for _, group in core_rows.groupby("PROFILE_NUMBER"):
        x = (group["TIME"] - group["TIME"].iloc[0]).dt.total_seconds().values
        y = group[depth_col].values
        if len(x) > 1:
            m, _ = np.polyfit(x, y, 1)
            pnum = group["PROFILE_NUMBER"].iloc[0]
            mapped_df.loc[mapped_df["PROFILE_NUMBER"] == pnum, "GRADIENT"] = m

    mapped_df = mapped_df.sort_values("N_MEASUREMENTS").reset_index(drop=True)
    return mapped_df


@register_step
class FindProfilesStep(BaseStep, QCHandlingMixin):
    """
    Identifies and classifies vertical and horizontal profiles from depth-time data.
    Derives continuous cycle numbers and assigns scientific phase flags.
    """
    
    step_name = "Find Profiles"
    required_variables = ["TIME"]
    provided_variables = ["PROFILE_NUMBER", "PROFILE_DIRECTION", "PROFILE_GRADIENT", "CYCLE", "SCI_PHASE"]

    parameter_schema = {
        "depth_column": {
            "type": str,
            "default": "PRES",
            "description": "Depth or pressure column name. Defaults to PRES."
        },
        "time_window_seconds": {
            "type": int,
            "default": 30,
            "description": "Time window in seconds for smoothing and binning."
        },
        "is_propelled": {
            "type": [str, bool],
            "default": "auto",
            "description": "Whether to map transects to Propelled (True), Parking (False), or check attributes ('auto')."
        },
        "velocity_threshold": {
            "type": float,
            "default": 0.033,
            "description": "Velocity threshold (m/s) to trigger ascent/descent mapping."
        },
        "acceleration_threshold": {
            "type": float,
            "default": 0.0005,
            "description": "Acceleration threshold to define stable horizontal phases."
        },
        "transition_buffer_seconds": {
            "type": int,
            "default": 30,
            "description": "Buffer trim applied to ends of continuous phases."
        },
        "min_duration_minutes": {
            "type": int,
            "default": 5,
            "description": "Minimum minutes required for a phase block to be valid."
        },
        "peak_prominence": {
            "type": float,
            "default": 20,
            "description": "Topographical prominence required to flag an inflection."
        },
        "min_samples_between_peaks": {
            "type": int,
            "default": 20,
            "description": "Minimum bins required between peak detections."
        },
        "gap_threshold_minutes": {
            "type": int,
            "default": 5,
            "description": "Data gap length indicating disconnected chunks for inflection checking."
        },
        "surface_depth": {
            "type": float,
            "default": 20,
            "description": "Depth boundary defining general surface operation."
        },
        "surfacing_threshold": {
            "type": float,
            "default": 5,
            "description": "Strict threshold to assign the surfacing (Phase 3) state."
        },
        "parking_gradient_threshold": {
            "type": float,
            "default": 0.005,
            "description": "Absolute gradient threshold (m/s) above which a drifting parking phase is converted back to ascent/descent."
        }
    }

    def run(self):
        self.log("Attempting to designate profile numbers, cycles, directions, and phases")
        self.check_data()
        self.filter_qc()

        depth_col = getattr(self, "depth_column", "PRES")
        if depth_col not in self.data.variables:
            raise ValueError(f"Specified depth column '{depth_col}' not found in the dataset.")

        time_window_seconds = getattr(self, "time_window_seconds", 30)
        is_propelled = getattr(self, "is_propelled", "auto")
        velocity_threshold = getattr(self, "velocity_threshold", 0.033)
        acceleration_threshold = getattr(self, "acceleration_threshold", 0.0005)
        transition_buffer_seconds = getattr(self, "transition_buffer_seconds", 30)
        min_duration_minutes = getattr(self, "min_duration_minutes", 5)
        peak_prominence = getattr(self, "peak_prominence", 20.0)
        min_samples_between_peaks = getattr(self, "min_samples_between_peaks", 20)
        gap_threshold_minutes = getattr(self, "gap_threshold_minutes", 5)
        surface_depth = getattr(self, "surface_depth", 20.0)
        surfacing_threshold = getattr(self, "surfacing_threshold", 5.0)
        parking_gradient_threshold = getattr(self, "parking_gradient_threshold", 0.005)

        if is_propelled == "auto":
            attr_id = str(self.data.attrs.get("id", ""))
            target_transect_phase = 6 if "ALR" in attr_id else 4
        elif is_propelled is True:
            target_transect_phase = 6
        else:
            target_transect_phase = 4

        cols_to_extract = ["TIME", depth_col]
        df_raw = self.data[cols_to_extract].to_dataframe().reset_index()

        df_final = find_profiles(
            df_raw, depth_col, time_window_seconds, target_transect_phase, 
            velocity_threshold, acceleration_threshold, transition_buffer_seconds, 
            min_duration_minutes, peak_prominence, min_samples_between_peaks, 
            gap_threshold_minutes, surface_depth, surfacing_threshold,
            parking_gradient_threshold
        )

        if self.diagnostics:
            self.generate_diagnostics(df_final, depth_col)

        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), df_final["PROFILE_NUMBER"].to_numpy())
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. NaN indicates no profile.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": 1,
            "valid_max": np.inf,
        }

        self.data["PROFILE_DIRECTION"] = (("N_MEASUREMENTS",), df_final["PROFILE_DIRECTION"].to_numpy())
        self.data.PROFILE_DIRECTION.attrs = {
            "long_name": "Profile direction: -1 ascent, 1 descent, 0 transect (surfacing/parking/propelled), NaN otherwise.",
            "units": "None",
            "standard_name": "Profile Direction",
            "valid_min": -1,
            "valid_max": 1,
        }

        self.data["PROFILE_GRADIENT"] = (("N_MEASUREMENTS",), df_final["GRADIENT"].to_numpy())
        self.data.PROFILE_GRADIENT.attrs = {
            "long_name": "Profile Vertical Gradient",
            "units": "m/s",
        }

        self.data["CYCLE"] = (("N_MEASUREMENTS",), df_final["CYCLE"].to_numpy())
        self.data.CYCLE.attrs = {
            "long_name": "Continuous cycle number derived from surfacing points",
            "units": "None",
            "standard_name": "Cycle Number",
            "valid_min": 1,
            "valid_max": np.inf,
        }

        self.data["SCI_PHASE"] = (("N_MEASUREMENTS",), df_final["SCI_PHASE"].to_numpy())
        self.data.SCI_PHASE.attrs = {
            "long_name": "Scientific Phase Classification",
            "units": "None",
            "valid_min": 0,
            "valid_max": 7,
            "flag_values": "0, 1, 2, 3, 4, 5, 6, 7",
            "flag_meanings": "unknown ascent descent surfacing parking inflection propelled transition"
        }

        self.generate_qc({
            "PROFILE_NUMBER_QC": ["TIME_QC", f"{depth_col}_QC"],
            "PROFILE_DIRECTION_QC": ["TIME_QC", f"{depth_col}_QC"],
            "PROFILE_GRADIENT_QC": ["TIME_QC", f"{depth_col}_QC"],
            "CYCLE_QC": ["TIME_QC", f"{depth_col}_QC"],
            "SCI_PHASE_QC": ["TIME_QC", f"{depth_col}_QC"]
        })

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self, mapped_df, depth_col):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Panel 1: Phase mapping
        for p_val in range(8):
            mask = mapped_df["SCI_PHASE"] == p_val
            n_points = mask.sum()
            
            if n_points > 0:
                t_data = mapped_df["TIME"][mask]
                depth_data = mapped_df[depth_col][mask]
            else:
                t_data = []
                depth_data = []
                
            lbl = f"{PHASE_NAMES.get(p_val, f'Phase {p_val}')} (n={n_points})"
            z_ord = 6 if p_val == 5 else 3
                
            ax1.plot(t_data, depth_data, linestyle="none", marker=".", markersize=8, 
                    color=PHASE_COLOURS.get(p_val, "black"), label=lbl, zorder=z_ord)

        ax1.invert_yaxis()
        ax1.set_ylabel("Pressure/Depth")
        ax1.set_title("Diagnostics | High Resolution Phase Mapping")
        leg = ax1.legend(loc="upper right", fontsize=10, markerscale=2.0)
        leg.set_zorder(100) 
        ax1.grid(alpha=0.3)

        # Panel 2: Profile ID & Cycle
        ax2.plot(mapped_df["TIME"], mapped_df["PROFILE_NUMBER"], color="tab:blue", marker=".", ls="", ms=4, label="Profile Number")
        ax2.plot(mapped_df["TIME"], mapped_df["CYCLE"], color="tab:red", marker=".", ls="", ms=4, label="Cycle Number")
        ax2.set_ylabel("ID / Cycle")
        ax2.set_xlabel("Time")
        ax2.legend(loc="upper left")
        ax2.grid(alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        
        plt.show(block=True)
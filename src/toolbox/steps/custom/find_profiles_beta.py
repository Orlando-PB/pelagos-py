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

"""Class definition for finding profiles and their direction."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
import numpy as np


# --- Configurable Variables ---
PLOT_FIGSIZE = (18, 10)
PLOT_MARKER_SIZE = 1
COLOUR_ASCENDING = "tab:blue"
COLOUR_DESCENDING = "teal"
COLOUR_NOT_PROFILE = "tab:red"
COLOUR_RAW_VELOCITY = "k"
COLOUR_RAW_ALPHA = 0.1
COLOUR_THRESHOLDS = "gray"
TURN_BUFFER_SIZE = 5
# ------------------------------


def find_profiles(
    df,
    gradient_thresholds: list,
    filter_win_sizes=["20s", "10s"],
    time_col="TIME",
    depth_col="DEPTH",
):
    """
    Identifies vertical profiles and their direction by analysing depth gradients over time.

    This function processes depth-time data to identify periods where an instrument is performing
    vertical profiling based on gradient thresholds. It handles data interpolation, calculates vertical
    velocities, applies median filtering, and assigns unique profile numbers and directions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing time and depth measurements.
    gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold] defining the vertical velocity
        range (in metres/second) that is NOT considered part of a profile. Typical values are around [0.02, -0.02].
    filter_win_sizes : list, default=['20s', '10s']
        Window sizes for the compound filter applied to gradient calculations, in Pandas duration format.
        Index 0 controls the rolling median window size and index 1 controls the rolling mean window size.
    time_col : str, default='TIME'
        Name of the column containing timestamp data.
    depth_col : str, default='DEPTH'
        Name of the column containing depth measurements.

    Returns
    -------
    pandas.DataFrame
        Dataframe with additional columns:
        - 'dt': Time difference between consecutive points (seconds)
        - 'dz': Depth difference between consecutive points (metres)
        - 'grad': Vertical velocity (dz/dt, metres/second)
        - 'smooth_grad': Median-filtered vertical velocity
        - 'is_profile': Boolean indicating if a point belongs to a profile
        - 'profile_num': Unique identifier for each identified profile (NaN for non-profile points)
        - 'profile_dir': 1 for Ascending, -1 for Descending (NaN for non-profile points)

    Notes
    -----
    - The function considers a point part of a profile when its smoothed vertical velocity
      falls outside the range specified by gradient_thresholds, utilising hysteresis to prevent
      fragmentation from noise.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    df_indexed = df.set_index(time_col).copy()
    
    df_indexed[f"INTERP_{depth_col}"] = df_indexed[depth_col].replace([np.inf, -np.inf], np.nan)
    df_indexed[f"INTERP_{depth_col}"] = df_indexed[f"INTERP_{depth_col}"].interpolate(method='time')
    
    df_valid = df_indexed.dropna(subset=[f"INTERP_{depth_col}"]).copy()
    
    dt = df_valid.index.to_series().diff().dt.total_seconds()
    dz = df_valid[f"INTERP_{depth_col}"].diff()
    
    df_valid["dt"] = dt
    df_valid["dz"] = dz
    df_valid["grad"] = df_valid["dz"] / df_valid["dt"]
    
    df_grad = df_valid.dropna(subset=["grad"]).copy()
    
    df_grad["smooth_grad"] = (
        df_grad["grad"]
        .rolling(window=filter_win_sizes[0], center=True).median()
        .rolling(window=filter_win_sizes[1], center=True).mean()
    )
    
    pos_grad, neg_grad = gradient_thresholds
    
    is_outside = ~df_grad["smooth_grad"].between(neg_grad, pos_grad)
    
    direction = pd.Series(np.nan, index=df_grad.index)
    direction.loc[df_grad["smooth_grad"] > pos_grad] = -1  # Descending (depth increasing)
    direction.loc[df_grad["smooth_grad"] < neg_grad] = 1   # Ascending (depth decreasing)
    direction = direction.ffill().fillna(0)
    
    direction_change = direction != direction.shift(1)
    direction_change.iloc[0] = False 
    
    turn_regions = ~is_outside
    turn_regions = turn_regions | direction_change | direction_change.shift(-1, fill_value=False)
    turn_regions = turn_regions.rolling(window=TURN_BUFFER_SIZE, center=True, min_periods=1).max().astype(bool)
    
    is_profile = ~turn_regions
    
    new_profile_starts = is_profile & ~is_profile.shift(1, fill_value=False)
    profile_blocks = new_profile_starts.cumsum()
    
    df_grad["is_profile"] = is_profile
    df_grad["profile_num"] = profile_blocks.where(is_profile, np.nan)
    df_grad["profile_dir"] = direction.where(is_profile, np.nan)
    
    result_df = df.set_index(time_col).join(
        df_grad[["dt", "dz", "grad", "smooth_grad", "is_profile", "profile_num", "profile_dir"]],
        how="left"
    ).reset_index()
    
    result_df[f"INTERP_{depth_col}"] = df_indexed[f"INTERP_{depth_col}"].values
    
    return result_df


@register_step
class FindProfilesStep(BaseStep, QCHandlingMixin):
    """
    Determine profile numbers and whether water profiles are ascending or descending.

    This step processes depth data to segment the dataset into continuous vertical 
    profiles, calculating both the unique profile number and the direction of travel 
    (1 for Ascending, -1 for Descending).
    """

    step_name = "Find Profiles beta"

    def run(self):
        self.log("Attempting to designate profile numbers and directions")

        self.filter_qc()

        self.thresholds = self.parameters["gradient_thresholds"]
        self.win_sizes = self.parameters["filter_window_sizes"]
        self.depth_col = self.parameters["depth_column"]

        if self.diagnostics:
            self.log("Generating diagnostics")
            root = self.generate_diagnostics()
            root.mainloop()

        self._df = self.data[["TIME", self.depth_col]].to_dataframe()
        if "TIME" not in self._df.columns:
            self._df = self._df.reset_index()

        self.profile_outputs = find_profiles(
            self._df, self.thresholds, self.win_sizes, depth_col=self.depth_col
        )
        
        profile_numbers = self.profile_outputs["profile_num"].to_numpy()
        profile_directions = self.profile_outputs["profile_dir"].to_numpy()

        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), profile_numbers)
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. #=-1 indicates no profile, #>=0 are profiles.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": -1,
            "valid_max": np.inf,
        }

        self.data["PROFILE_DIRECTION"] = (("N_MEASUREMENTS",), profile_directions)
        self.data.PROFILE_DIRECTION.attrs = {
            "long_name": "Profile direction. 1=Ascending, -1=Descending",
            "units": "None",
            "standard_name": "Profile Direction",
            "valid_min": -1,
            "valid_max": 1,
        }

        self.reconstruct_data()
        self.update_qc()

        self.generate_qc(
            {
                "PROFILE_NUMBER_QC": ["TIME_QC", f"{self.depth_col}_QC"],
                "PROFILE_DIRECTION_QC": ["PROFILE_NUMBER_QC", "TIME_QC", f"{self.depth_col}_QC"],
            }
        )

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):

        def generate_plot():
            mpl.use("TkAgg")

            self._df = self.data[["TIME", self.depth_col]].to_dataframe()
            if "TIME" not in self._df.columns:
                self._df = self._df.reset_index()

            self.profile_outputs = find_profiles(
                self._df, self.thresholds, self.win_sizes, depth_col=self.depth_col
            )

            ascending = self.profile_outputs[self.profile_outputs["profile_dir"] == 1]
            descending = self.profile_outputs[self.profile_outputs["profile_dir"] == -1]
            not_profiles = self.profile_outputs[self.profile_outputs["is_profile"] != True]

            fig = plt.figure(figsize=PLOT_FIGSIZE)
            gs = fig.add_gridspec(3, 4, wspace=0.3, hspace=0.3)
            
            ax_depth = fig.add_subplot(gs[0, :3])
            ax_vel = fig.add_subplot(gs[1, :3], sharex=ax_depth)
            ax_prof = fig.add_subplot(gs[2, :3], sharex=ax_depth)

            ax_depth.set_ylabel("Interpolated Depth")
            ax_vel.set_ylabel("Vertical Velocity")
            ax_prof.set_ylabel("Profile Number")
            ax_prof.set_xlabel("Time")
            
            ax_depth.tick_params(labelbottom=False)
            ax_vel.tick_params(labelbottom=False)

            for data, col, label in zip(
                [ascending, descending, not_profiles],
                [COLOUR_ASCENDING, COLOUR_DESCENDING, COLOUR_NOT_PROFILE],
                ["Ascending", "Descending", "Not Profile"],
            ):
                ax_depth.plot(
                    data["TIME"],
                    -data[f"INTERP_{self.depth_col}"],
                    marker=".",
                    markersize=PLOT_MARKER_SIZE,
                    ls="",
                    c=col,
                    label=label,
                )
                ax_vel.plot(
                    data["TIME"],
                    data["smooth_grad"],
                    marker=".",
                    markersize=PLOT_MARKER_SIZE,
                    ls="",
                    c=col,
                    label=label,
                )

            ax_vel.plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["grad"],
                c=COLOUR_RAW_VELOCITY,
                alpha=COLOUR_RAW_ALPHA,
                label="Raw Velocity",
            )
            for val, label in zip(self.thresholds, ["Gradient Thresholds", None]):
                ax_vel.axhline(val, ls="--", color=COLOUR_THRESHOLDS, label=label)

            ax_prof.plot(
                self.profile_outputs["TIME"],
                self.profile_outputs["profile_num"],
                c=COLOUR_THRESHOLDS,
            )

            ax_depth.legend(loc="upper right")
            ax_vel.legend(loc="upper right")

            valid_profiles = self.profile_outputs["profile_num"].dropna().unique()
            if len(valid_profiles) > 0:
                indices = np.linspace(0, len(valid_profiles) - 1, min(3, len(valid_profiles))).astype(int)
                sample_profs = valid_profiles[indices]
                
                zoom_axs = [fig.add_subplot(gs[i, 3]) for i in range(len(sample_profs))]
                
                for ax, p_num in zip(zoom_axs, sample_profs):
                    p_data = self.profile_outputs[self.profile_outputs["profile_num"] == p_num]
                    
                    if p_data["profile_dir"].iloc[0] == 1:
                        p_col = COLOUR_ASCENDING
                    else:
                        p_col = COLOUR_DESCENDING
                        
                    ax.plot(
                        p_data["TIME"], 
                        -p_data[f"INTERP_{self.depth_col}"], 
                        color=p_col, 
                        marker=".", 
                        markersize=3
                    )
                    ax.set_title(f"Profile {int(p_num)}")
                    ax.tick_params(axis='x', rotation=45, labelsize=8)

            plt.show(block=False)

        root = tk.Tk()
        root.title("Parameter Adjustment")
        root.geometry(f"380x{50*len(self.parameters)}")
        entries = {}

        row = 0
        values = self.thresholds
        tk.Label(root, text="Gradient Thresholds:").grid(row=row, column=0)
        for i, label, value in zip(range(2), ["+ve", "-ve"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        row = 2
        values = self.win_sizes
        tk.Label(root, text="Filter Window Sizes:").grid(row=row, column=0, pady=(20, 0))
        for i, label, value in zip(range(2), ["Median Filter", "Mean Filter"], values):
            tk.Label(root, text=f"{label}:").grid(row=row + 1, column=2 * i)
            entry = tk.Entry(root, textvariable=label, width=10)
            entry.insert(0, value)
            entry.grid(row=row + 1, column=2 * i + 1)
            entries[label] = entry

        row = 4
        value = self.depth_col
        tk.Label(root, text="Depth column name:").grid(row=row, column=0, pady=(20, 0))
        entry = tk.Entry(root, textvariable="depth_column")
        entry.insert(0, value)
        entry.grid(row=row, column=1, pady=(20, 0))
        entries["depth_column"] = entry

        def on_cancel():
            plt.close('all')
            root.quit()
            root.destroy()

        def on_regenerate():
            self.thresholds = [float(entries["+ve"].get()), float(entries["-ve"].get())]
            self.win_sizes = [
                entries["Median Filter"].get(),
                entries["Mean Filter"].get(),
            ]
            self.depth_col = entries["depth_column"].get()

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
            root.quit()
            root.destroy()

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
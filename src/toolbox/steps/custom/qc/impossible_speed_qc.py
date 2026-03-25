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

"""QC test to identify impossible speeds in glider data."""

#### Mandatory imports ####
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib


@register_qc
class impossible_speed_qc(BaseQC):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: TIME, LATITUDE, LONGITUDE
    Checks that the the gliders horizontal speed stays below 3m/s
    """

    qc_name = "impossible speed qc"
    expected_parameters = {}
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Get time difference in seconds safely
        dt = (self.data["TIME"] - self.data["TIME"].shift(N_MEASUREMENTS=1)) / np.timedelta64(1, "s")

        # Interpolate missing or infinite coordinates
        lat = self.data["LATITUDE"].where(np.isfinite(self.data["LATITUDE"])).interpolate_na(dim="N_MEASUREMENTS")
        lon = self.data["LONGITUDE"].where(np.isfinite(self.data["LONGITUDE"])).interpolate_na(dim="N_MEASUREMENTS")

        # Convert coordinates to radians for Haversine calculation
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Shift to get previous coordinates
        prev_lat_rad = lat_rad.shift(N_MEASUREMENTS=1)
        prev_lon_rad = lon_rad.shift(N_MEASUREMENTS=1)
        
        # Haversine formula
        a = (
            np.sin((lat_rad - prev_lat_rad) / 2)**2 +
            np.cos(prev_lat_rad) * np.cos(lat_rad) *
            np.sin((lon_rad - prev_lon_rad) / 2)**2
        )
        
        # Radius of Earth is approx 6371000 metres
        distance_m = 6371000.0 * 2 * np.arcsin(np.sqrt(a))

        # Calculate raw absolute speed arriving at each point
        self.absolute_speed = distance_m / dt
        self.absolute_speed = self.absolute_speed.fillna(0.0)

        # Calculate speed leaving each point
        speed_leaving = self.absolute_speed.shift(N_MEASUREMENTS=-1)

        # Spike Detection: isolated bad fixes have impossible speeds arriving AND leaving
        is_spike = (self.absolute_speed > 3.0) & (speed_leaving > 3.0)

        # Ensure the final point is flagged if it arrives too fast 
        is_spike = xr.where(speed_leaving.isnull() & (self.absolute_speed > 3.0), True, is_spike)

        speed_is_valid = ~is_spike & self.absolute_speed.notnull() & np.isfinite(self.absolute_speed)

        flag_values = xr.where(speed_is_valid, 1, 4)

        self.flags = xr.Dataset(
            data_vars={
                col + "_QC": ("N_MEASUREMENTS", flag_values.values) for col in self.required_variables
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        
        fig, axs = plt.subplots(nrows=3, figsize=(7, 6), sharex=True, dpi=200)

        time_vals = self.data["TIME"].values
        abs_speed_vals = self.absolute_speed.values
        lat_vals = self.data["LATITUDE"].values
        lon_vals = self.data["LONGITUDE"].values

        mask_flag1 = (self.flags["LATITUDE_QC"] == 1).values
        mask_flag4 = (self.flags["LATITUDE_QC"] == 4).values
        
        count_flag1 = np.sum(mask_flag1)
        count_flag4 = np.sum(mask_flag4)

        # Plot 1: Absolute Speed
        axs[0].plot(time_vals, abs_speed_vals, c="teal", lw=0.5, zorder=1, label="Raw Speed (m/s)")
        max_speed = np.nanmax(abs_speed_vals)
        y_max = 20.0 if max_speed > 20.0 else max(4.0, max_speed * 1.1)
        axs[0].set(
            title="Impossible Speed Test Diagnostics",
            ylabel="Speed (m/s)",
            ylim=(0, y_max),
        )
        axs[0].axhline(3, ls="--", c="magenta", lw=1, label="Threshold (3m/s)")
        axs[0].legend(loc="upper right", fontsize="small")

        # Plot 2: Latitude over time
        axs[1].plot(
            time_vals[mask_flag1], lat_vals[mask_flag1], 
            c="teal", ls="", marker="o", markersize=1.5, label=f"Flag 1 (N={count_flag1})"
        )
        if count_flag4 > 0:
            axs[1].plot(
                time_vals[mask_flag4], lat_vals[mask_flag4], 
                c="magenta", ls="", marker="o", markersize=1.5, label=f"Flag 4 (N={count_flag4})"
            )
        axs[1].set(ylabel="Latitude")
        axs[1].legend(loc="upper right", fontsize="small")

        # Plot 3: Longitude over time
        axs[2].plot(
            time_vals[mask_flag1], lon_vals[mask_flag1], 
            c="teal", ls="", marker="o", markersize=1.5, label=f"Flag 1 (N={count_flag1})"
        )
        if count_flag4 > 0:
            axs[2].plot(
                time_vals[mask_flag4], lon_vals[mask_flag4], 
                c="magenta", ls="", marker="o", markersize=1.5, label=f"Flag 4 (N={count_flag4})"
            )
        axs[2].set(
            xlabel="Time (s)",
            ylabel="Longitude",
        )
        axs[2].legend(loc="upper right", fontsize="small")

        fig.tight_layout()
        plt.show(block=True)
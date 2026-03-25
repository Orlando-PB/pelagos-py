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

        # Calculate absolute speed in metres per second
        self.absolute_speed = distance_m / dt

        # First row will be NaN because there is no previous point to calculate speed.
        # We fill the first NaN with 0.0 so it passes the valid speed check.
        self.absolute_speed = self.absolute_speed.fillna(0.0)

        # TODO: Does this need a flag for potentially bad data for cases where speed is inf?
        speed_is_valid = (
            (self.absolute_speed < 3.0)  #  Speed threshold
            & self.absolute_speed.notnull()
            & np.isfinite(self.absolute_speed)
        )

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
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        time_vals = self.data["TIME"].values
        abs_speed_vals = self.absolute_speed.values

        for i in range(10):
            # Plot by flag number
            mask = self.flags["LATITUDE_QC"] == i
            if not mask.any():
                continue

            # Plot the data
            ax.plot(
                time_vals[mask.values],
                abs_speed_vals[mask.values],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )

        ax.set(
            title="Impossible Speed Test",
            xlabel="Time (s)",
            ylabel="Absolute Horizontal Speed (m/s)",
            ylim=(0, 4),
        )
        ax.axhline(3, ls="--", c="k")
        ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)
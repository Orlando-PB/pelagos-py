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
from toolbox.steps.base_qc import BaseTest, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import polars as pl
import xarray as xr
import numpy as np
import matplotlib

@register_qc
class impossible_speed_qc(BaseTest):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: TIME, LATITUDE, LONGITUDE
    Checks that the the gliders horizontal speed stays below 3m/s
    """

    test_name = "impossible speed test"
    
    parameter_schema = {
        "max_speed": {
            "type": float,
            "default": 3.0,
            "description": "Maximum allowable horizontal speed in m/s"
        }
    }
    
    test_name = "impossible speed qc"
    expected_parameters = {}
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        max_speed = getattr(self, "max_speed", 3.0)

        self.df = self.df.with_columns(
            (pl.col("TIME").diff().cast(pl.Float64) * 1e-9).alias("dt")
        )
        for label in ["LATITUDE", "LONGITUDE"]:
            self.df = self.df.with_columns(
                pl.col(label)
                .replace([np.inf, -np.inf, np.nan], None)
                .interpolate_by("TIME")
                .diff()
                .alias(f"delta_{label}")
            )
            self.df = self.df.with_columns(
                (pl.col(f"delta_{label}") / pl.col("dt")).alias(f"{label}_speed")
            )
            
        self.df = self.df.with_columns(
            (
                (pl.col("LATITUDE_speed") ** 2 + pl.col("LONGITUDE_speed") ** 2) ** 0.5
            ).alias("absolute_speed")
        )

        self.df = self.df.with_columns(
            (
                (pl.col("absolute_speed") < max_speed)  
                & pl.col("absolute_speed").is_not_null()
                & pl.col("absolute_speed").is_finite()
            ).alias("speed_is_valid")
        )

        for label in ["LATITUDE", "LONGITUDE", "TIME"]:
            self.df = self.df.with_columns(
                pl.when(pl.col("speed_is_valid"))
                .then(1)
                .otherwise(4)
                .alias(f"{label}_QC")
            )

        flags = self.df.select(pl.col("^.*_QC$"))
        self.flags = xr.Dataset(
            data_vars={
                col: ("N_MEASUREMENTS", flags[col].to_numpy()) for col in flags.columns
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def create_diagnostic_plot(self):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        max_speed = getattr(self, "max_speed", 3.0)

        for i in range(10):
            plot_data = self.df.filter(pl.col("LATITUDE_QC") == i)
            if len(plot_data) == 0:
                continue

            ax.plot(
                plot_data["TIME"],
                plot_data["absolute_speed"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )

        ax.set(
            title="Impossible Speed Test",
            xlabel="Time (s)",
            ylabel="Absolute Horizontal Speed (m/s)",
            ylim=(0, max_speed + 1),
        )
        ax.axhline(max_speed, ls="--", c="k")
        ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        return fig

    def plot_diagnostics(self):
        if self.is_web_mode():
            self.web_diagnostic_loop()
        else:
            matplotlib.use("tkagg")
            fig = self.create_diagnostic_plot()
            plt.show(block=True)
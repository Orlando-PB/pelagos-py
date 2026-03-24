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

"""QC test to identify impossible dates in TIME variable."""

#### Mandatory imports ####
from toolbox.steps.base_qc import BaseTest, register_qc, flag_cols

#### Custom imports ####
import polars as pl
import xarray as xr
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

@register_qc
class impossible_date_qc(BaseTest):
    """
    Target Variable: TIME
    Flag Number: 4 (bad data)
    Variables Flagged: TIME
    Checks that the datetime of each point is valid.
    """

    test_name = "impossible date qc"
    
    parameter_schema = {
        "start_year": {
            "type": int,
            "default": 1985,
            "description": "The earliest logical year for the data."
        }
    }
    
    required_variables = ["TIME"]
    qc_outputs = ["TIME_QC"]

    def return_qc(self):
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        start_yr = getattr(self, "start_year", 1985)

        self.df = self.df.with_columns(
            pl.when(pl.col("TIME").is_null())
            .then(9)
            .when(
                (
                    (pl.col("TIME") > datetime(start_yr, 1, 1))
                    & (pl.col("TIME") < datetime.now())
                )
            )
            .then(1)
            .otherwise(4)
            .alias("TIME_QC")
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
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        for i in range(10):
            plot_data = self.df.with_row_index().filter(pl.col("TIME_QC") == i)
            if len(plot_data) == 0:
                continue

            ax.plot(
                plot_data["index"],
                plot_data["TIME"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )
        ax.set(
            title="Impossible Date Test",
            xlabel="Index",
            ylabel="TIME",
        )
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
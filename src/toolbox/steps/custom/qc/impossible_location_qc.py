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

"""QC test to identify impossible locations in LATITUDE and LONGITUDE variables."""

#### Mandatory imports ####
from toolbox.steps.base_qc import BaseTest, register_qc, flag_cols

#### Custom imports ####
import polars as pl
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

@register_qc
class impossible_location_qc(BaseTest):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the latitude and longitude are valid.
    """

    test_name = "impossible location qc"
    expected_parameters = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        for label, bounds in zip(["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]):
            self.df = self.df.with_columns(
                pl.when(pl.col(label).is_nan())
                .then(9)
                .when((pl.col(label) > bounds[0]) & (pl.col(label) < bounds[1]))
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
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, dpi=200)

        for ax, var, bounds in zip(
            axs, ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            for i in range(10):
                plot_data = self.df.with_row_index().filter(pl.col(f"{var}_QC") == i)
                if len(plot_data) == 0:
                    continue

                ax.plot(
                    plot_data["index"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )
            ax.set(
                xlabel="Index",
                ylabel=var,
            )
            ax.legend(title="Flags", loc="upper right")
            for bound in bounds:
                ax.axhline(bound, ls="--", c="k")

        fig.suptitle("Impossible Location Test")
        fig.tight_layout()
        return fig

    def plot_diagnostics(self):
        if self.is_web_mode():
            self.web_diagnostic_loop()
        else:
            matplotlib.use("tkagg")
            fig = self.create_diagnostic_plot()
            plt.show(block=True)
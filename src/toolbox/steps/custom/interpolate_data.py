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

"""Class definition for deriving CTD variables."""

from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

import polars as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


@register_step
class InterpolateVariables(BaseStep, QCHandlingMixin):
    """
    A processing step for interpolating data.
    """

    step_name = "Interpolate Data"
    required_variables = ["TIME"]
    provided_variables = []

    parameter_schema = {
        "qc_handling_settings": {
            "type": dict,
            "default": {
                "flag_filter_settings": {
                    "PRES": [3, 4, 9],
                    "LATITUDE": [3, 4, 9],
                    "LONGITUDE": [3, 4, 9]
                },
                "reconstruction_behaviour": "replace",
                "flag_mapping": {3: 8, 4: 8, 9: 8}
            },
            "description": "Dictionary defining QC handling and flag mapping"
        }
    }

    def run(self):
        self.log("Interpolating variables...")

        self.filter_qc()

        self.df = pl.from_pandas(self.data[list(self.filter_settings.keys() | {"TIME"})].to_dataframe(), nan_to_null=False)
        self.unprocessed_df = self.df.clone()

        self.df = self.df.with_columns(
            pl.col(var).replace({np.nan: None}).interpolate_by("TIME").replace({None: np.nan})
            for var in self.filter_settings.keys()
        )

        for var in self.filter_settings.keys():
            self.data[var][:] = self.df[var].to_numpy()

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            if self.is_web_mode():
                self.web_diagnostic_loop()
            else:
                self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def create_diagnostic_plot(self):
        """
        Creates and returns the matplotlib figure for web or native display.
        """
        fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 8), dpi=200)

        plot_var = list(self.filter_settings.keys())[0]
        for ax, data in zip(axs.flatten(), [self.unprocessed_df, self.df]):
            ax.plot(data[plot_var])

        fig.tight_layout()
        return fig

    def generate_diagnostics(self):
        """
        Generate diagnostic plots comparing original and interpolated data natively.
        """
        matplotlib.use("tkagg")
        fig = self.create_diagnostic_plot()
        plt.show(block=True)
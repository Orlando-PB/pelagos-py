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

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin

#### Custom imports ####
import polars as pl
import numpy as np
import gsw
import matplotlib.pyplot as plt
import matplotlib

# Plotting Configuration for easy tweaking
PLOT_DPI = 150
MAX_POINTS = 100000
PLOT_COLOUR = "tab:blue"

@register_step
class DeriveCTDVariables(BaseStep, QCHandlingMixin):
    """
    A processing step class for deriving oceanographic variables from CTD data.
    """

    step_name = "Derive CTD"
    required_variables = ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]
    provided_variables = ["DEPTH", "PRAC_SALINITY", "ABS_SALINITY", "CONS_TEMP", "DENSITY"]

    parameter_schema = {
        "to_derive": {
            "type": list,
            "default": ["DEPTH", "PRAC_SALINITY", "ABS_SALINITY", "CONS_TEMP", "DENSITY"],
            "description": "List of oceanographic variables to calculate."
        },
        "qc_handling_settings": {
            "type": dict,
            "default": {
                "flag_filter_settings": {
                    "PRES": [3, 4, 9],
                    "TEMP": [3, 4, 9],
                    "CNDC": [3, 4, 9]
                },
                "reconstruction_behaviour": "replace",
                "flag_mapping": {3: 8, 4: 8, 9: 8}
            },
            "description": "Rules for handling existing QC flags during calculation."
        }
    }

    def run(self):
        self.log(f"Processing CTD...")

        # Pre-filter data based on QC flags defined in schema
        self.filter_qc()

        # Extract only variables needed for GSW to Polars
        df = pl.from_pandas(
            self.data[
                ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]
            ].to_dataframe(),
            nan_to_null=False,
        )

        gsw_function_calls = (
            ("DEPTH", gsw.z_from_p, ["PRES", "LATITUDE"]),
            ("PRAC_SALINITY", gsw.SP_from_C, ["CNDC", "TEMP", "PRES"]),
            (
                "ABS_SALINITY",
                gsw.SA_from_SP,
                ["PRAC_SALINITY", "PRES", "LONGITUDE", "LATITUDE"],
            ),
            ("CONS_TEMP", gsw.CT_from_t, ["ABS_SALINITY", "TEMP", "PRES"]),
            ("DENSITY", gsw.rho, ["ABS_SALINITY", "CONS_TEMP", "PRES"]),
        )

        variable_metadata = {
            "DEPTH": {"long_name": "Depth", "units": "m", "standard_name": "DEPTH"},
            "PRAC_SALINITY": {"long_name": "Practical salinity", "units": "1", "standard_name": "PRAC_SALINITY"},
            "ABS_SALINITY": {"long_name": "Absolute salinity", "units": "g/kg", "standard_name": "ABS_SALINITY"},
            "CONS_TEMP": {"long_name": "Conservative temperature", "units": "°C", "standard_name": "CONS_TEMP"},
            "DENSITY": {"long_name": "Density", "units": "kg/m3", "standard_name": "DENSITY"},
        }

        for var_name, func, args in gsw_function_calls:
            if var_name not in self.to_derive:
                continue

            self.log(f"Deriving {var_name}...")

            df = df.with_columns(
                pl.struct(args)
                .map_batches(lambda x: func(*(x.struct.field(arg) for arg in args)))
                .alias(var_name)
            )

            self.data[var_name] = (("N_MEASUREMENTS",), df[var_name].to_numpy())
            self.data[var_name].attrs = variable_metadata[var_name]

            # Propagate QC flags from parents to derived child
            self.generate_qc({f"{var_name}_QC": [f"{arg}_QC" for arg in args]})

        # Put filtered bad data back if requested
        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            if self.is_web_mode():
                self.web_diagnostic_loop()
            else:
                matplotlib.use("tkagg")
                fig = self.create_diagnostic_plot()
                if fig:
                    plt.show()

        self.context["data"] = self.data
        return self.context

    def create_diagnostic_plot(self):
        """Builds subplots for derived variables vs Time, excluding bad data."""
        vars_to_plot = [v for v in self.to_derive if v in self.data.variables]
        if not vars_to_plot:
            return None

        plot_df = pl.from_pandas(
            self.data[vars_to_plot + ["TIME"] + [f"{v}_QC" for v in vars_to_plot]].to_dataframe(),
            nan_to_null=False
        )

        # Downsample logic to maintain performance
        nth = max(1, len(plot_df) // MAX_POINTS)
        plot_df = plot_df.gather_every(nth)

        fig, axs = plt.subplots(
            nrows=len(vars_to_plot), 
            ncols=1, 
            figsize=(10, 3 * len(vars_to_plot)), 
            sharex=True, 
            dpi=PLOT_DPI
        )
        
        if len(vars_to_plot) == 1:
            axs = [axs]

        for ax, var in zip(axs, vars_to_plot):
            # Hide flags 3 (Prob Bad), 4 (Bad), and 9 (Missing)
            clean_df = plot_df.filter(~pl.col(f"{var}_QC").is_in([3, 4, 9]))
            
            if len(clean_df) > 0:
                ax.plot(clean_df["TIME"], clean_df[var], color=PLOT_COLOUR, lw=1, marker='.', ms=1)
            
            ax.set_ylabel(f"{var}\n({self.data[var].attrs.get('units', '')})")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Derived {var} (Filtered: Flags 3, 4, 9 hidden)", loc='left', fontsize=10)

        axs[-1].set_xlabel("Time")
        fig.tight_layout()
        return fig
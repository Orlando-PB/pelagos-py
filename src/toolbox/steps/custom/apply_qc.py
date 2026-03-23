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

"""Class definition for quality control steps."""

from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
from toolbox.steps import QC_CLASSES

import polars as pl
import xarray as xr
import numpy as np
import json


@register_step
class ApplyQC(BaseStep):
    """
    Step to apply quality control tests to the dataset.
    """

    step_name = "Apply QC"

    # Updated schema to include your foundational QC tests by default
    parameter_schema = {
        "qc_settings": {
            "type": dict,
            "default": {
                "impossible date test": {},
                "impossible location test": {},
                "position on land test": {},
                "impossible speed test": {
                    "max_speed": 3.0
                }
            },
            "description": "Dictionary of QC tests to apply and their parameters."
        }
    }

    def organise_flags(self, new_flags):
        """
        Method for taking in new flags (new_flags) and cross checking against existing flags.
        """
        # Define combinatrix for handling flag upgrade behaviour
        qc_combinatrix = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 1, 2, 3, 4, 5, 1, 1, 8, 9],
                [2, 2, 2, 3, 4, 5, 2, 2, 8, 9],
                [3, 3, 3, 3, 4, 3, 3, 3, 3, 9],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 9],
                [5, 5, 5, 3, 4, 5, 5, 5, 8, 9],
                [6, 1, 2, 3, 4, 5, 6, 6, 8, 9],
                [7, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [8, 8, 8, 3, 4, 8, 8, 8, 8, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            ]
        )

        flag_columns_to_update = set(new_flags.data_vars) & set(
            self.flag_store.data_vars
        )
        for column_name in flag_columns_to_update:
            self.flag_store[column_name][:] = qc_combinatrix[
                self.flag_store[column_name], new_flags[column_name]
            ]

        flag_columns_to_add = set(new_flags.data_vars) - set(self.flag_store.data_vars)
        if len(flag_columns_to_add) > 0:
            for column_name in flag_columns_to_add:
                self.flag_store[column_name] = new_flags[column_name]

    def run(self):
        if not self.qc_settings or len(self.qc_settings.keys()) == 0:
            self.log("No QC operations were specified. Skipping.")
            return self.context
            
        invalid_requests = set(self.qc_settings.keys()) - set(QC_CLASSES.keys())
        if invalid_requests:
            raise KeyError(
                f"[Apply QC] The following requested QC tests could not be found: {invalid_requests}"
            )
            
        queued_qc = [QC_CLASSES.get(key) for key in self.qc_settings.keys()]

        self.check_data()
        data = self.context["data"].copy(deep=True)

        qc_history = self.context.setdefault("qc_history", {})

        all_required_variables = set({})
        test_qc_outputs_cols = set({})
        for test in queued_qc:
            if hasattr(test, "dynamic"):
                test_instance = test(None, **self.qc_settings[test.test_name])
                all_required_variables.update(test_instance.required_variables)
                test_qc_outputs_cols.update(test_instance.qc_outputs)
                del test_instance
            else:
                all_required_variables.update(test.required_variables)
                test_qc_outputs_cols.update(test.qc_outputs)
                
            if not set(all_required_variables).issubset(set(data.keys())):
                raise KeyError(
                    f"[Apply QC] The data is missing variables: ({set(all_required_variables) - set(data.keys())}) which are required for running QC '{test.test_name}'."
                )

        existing_flags = [
            flag_col for flag_col in data.data_vars if flag_col in test_qc_outputs_cols
        ]
        
        self.flag_store = xr.Dataset(coords={"N_MEASUREMENTS": data["N_MEASUREMENTS"]})
        if len(existing_flags) > 0:
            self.log(f"Found existing flags columns {set(existing_flags)} in data.")
            self.flag_store = data[existing_flags].fillna(9).astype(int)
        
        other_existing_qc = set([var for var in data.data_vars if var.endswith("_QC")]) - set(test_qc_outputs_cols)
        if any(other_existing_qc):
            self.log(f"Found QC columns for untested values: {other_existing_qc}")
            self.log("These columns will not be modified and are not subject to this step.")

        mia_qc = test_qc_outputs_cols - set(data.data_vars)
        base = [var[:-3] for var in mia_qc]
        if not set(base).issubset(set(data.keys())): 
            raise KeyError(
                f"[Apply QC] The data is missing: ({set(base) - set(data.keys())}), which is/are defined in the config to be flagged."
            )
            
        data_subset = data[base]
        masks = xr.where(data_subset.isnull(), 9, 0).astype(int)
        masks = masks.rename({var: f"{var}_QC" for var in base})
        self.flag_store.update(masks)

        for qc_test_name, qc_test_params in self.qc_settings.items():
            self.log(f"Applying: {qc_test_name}") 
            qc_test_instance = QC_CLASSES[qc_test_name](data, **qc_test_params)
            returned_flags = qc_test_instance.return_qc()  
            self.organise_flags(returned_flags)

            for flagged_var in returned_flags.data_vars:
                var_flags = returned_flags[flagged_var]
                percent_flagged = (
                    var_flags.to_numpy() != 0
                ).sum() / len(var_flags)
                if percent_flagged == 0:
                    self.log_warn(f"All flags for {flagged_var} remain 0 after {qc_test_name}")
                qc_history.setdefault(flagged_var, []).append(
                    (qc_test_name, percent_flagged)
                )

                parent_attrs = data[flagged_var[:-3]].attrs                
                attrs = self.flag_store[flagged_var].attrs
                attrs["quality_control_conventions"] = "Argo standard flags"
                attrs["valid_min"] = 0
                attrs["valid_max"] = 9
                attrs["flag_values"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                attrs["flag_meanings"] = "NO_QC, GOOD, PROB_GOOD, PROB_BAD, BAD, VALUE_CHANGED, NOT_USED, NOT_USED, ESTIMATED, MISSING"
                attrs["long_name"] = f"{parent_attrs['long_name']} quality flag"
                attrs["standard_name"] = f"{parent_attrs['standard_name']}_flag"
                attr_test = qc_test_name.replace(" ", "_").lower()
                attrs[f"{attr_test}_flag_cts"] = json.dumps({i: int(np.sum(var_flags.to_numpy() == i)) for i in range(10)})
                attrs[f"{attr_test}_stats"] = json.dumps(var_flags.to_series().describe().round(5).to_dict())
                attrs[f"{attr_test}_params"] = json.dumps(qc_test_params)

            if self.diagnostics and not self.is_web_mode():
                qc_test_instance.plot_diagnostics()

            del qc_test_instance

        for flag_column in self.flag_store.data_vars:
            if (self.flag_store[flag_column] == 0).all():
                self.log_warn(f"{flag_column} is all 0 after running all QC steps. Check intended QC variables and test requirements.")
            elif (self.flag_store[flag_column] == 0).any():
                n_zero = int((self.flag_store[flag_column] == 0).sum())
                self.log_warn(f"{flag_column} (length={len(self.flag_store[flag_column])}) has {n_zero} zero QC values following all QC steps.")

            data[flag_column] = (
                ("N_MEASUREMENTS",),
                self.flag_store[flag_column].to_numpy(),
            )
            data[flag_column].attrs = self.flag_store[flag_column].attrs.copy()
            
        self.context["data"] = data
        self.context["qc_history"] = qc_history

        return self.context
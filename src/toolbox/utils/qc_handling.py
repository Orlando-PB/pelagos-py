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

"""Class definition to handle quality control bulk operations."""

import numpy as np
import xarray as xr

class QCHandlingMixin:
    def __init__(self):
        # Fetch user inputs
        qc_settings = self.parameters.get("qc_handling_settings") or {}
        self.filter_settings = qc_settings.get("flag_filter_settings") or {}
        self.behaviour = qc_settings.get("reconstruction_behaviour") or "replace"

        self.flag_mapping = {flag: flag for flag in list(range(10))}
        if user_mappings := qc_settings.get("flag_mapping"):
            # Ensure keys and values are integers
            int_mappings = {int(k): int(v) for k, v in user_mappings.items()}
            self.flag_mapping.update(int_mappings)

        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        
        self.data = self.context["data"].copy(deep=True)
        self.data_copy = self.data.copy(deep=True)

        missing_variables = []
        for var in self.filter_settings:
            if var not in self.data or f"{var}_QC" not in self.data:
                self.log(f"One or both of {var}/{var}_QC are missing. Skipping filter.")
                missing_variables.append(var)
        for missing in missing_variables:
            self.filter_settings.pop(missing)

        super().__init__()

    def filter_qc(self):
        """NaN-out data based on bad QC flags."""
        for var, flags_to_nan in self.filter_settings.items():
            # Ensure the QC column is treated as integers for the .isin check
            qc_col = f"{var}_QC"
            clean_qc = self.data[qc_col].fillna(9).astype(int)
            mask = ~clean_qc.isin(flags_to_nan)
            self.data[var] = self.data[var].where(mask, np.nan)

    def reconstruct_data(self):
        """Reconstruct data by replacing flagged values with original values."""
        if self.behaviour == "replace":
            pass
        elif self.behaviour == "reinsert":
            for var, flags_to_nan in self.filter_settings.items():
                qc_col = f"{var}_QC"
                clean_qc = self.data[qc_col].fillna(9).astype(int)
                mask = clean_qc.isin(flags_to_nan)
                self.data[var] = xr.where(mask, self.data_copy[var], self.data[var])
        else:
            raise KeyError(f"Behaviour '{self.behaviour}' is not recognised.")

    def update_qc(self):
        """Update QC flags based on changes in data values."""
        for var in self.filter_settings.keys():
            qc_col = f"{var}_QC"
            
            # --- THE FIX: Handle NaNs in the QC column before mapping ---
            # We fill NaNs with 9 (Missing) and force to int so the mapper doesn't crash
            self.data[qc_col] = self.data[qc_col].fillna(9).astype(int)
            
            is_same = self.data[var] == self.data_copy[var]
            both_nan = np.logical_and(self.data[var].isnull(), self.data_copy[var].isnull())
            mask = is_same | both_nan

            updated_flags = xr.apply_ufunc(
                lambda x: self.flag_mapping.get(int(x), int(x)),
                self.data[qc_col],
                vectorize=True,
            )

            self.data[qc_col] = xr.where(mask, self.data_copy[qc_col], updated_flags)

    def generate_qc(self, qc_constituents: dict):
        """Generate QC flags for child variables based on parent flags."""
        qc_combinatrix = np.array([
            [0, 0, 0, 3, 4, 0, 0, 0, 0, 9],
            [0, 1, 2, 3, 4, 5, 1, 1, 8, 9],
            [0, 2, 2, 3, 4, 5, 2, 2, 8, 9],
            [3, 3, 3, 3, 4, 3, 3, 3, 3, 9],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 9],
            [0, 5, 5, 3, 4, 5, 5, 5, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 6, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 8, 8, 3, 4, 8, 8, 8, 8, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ])

        for qc_child, qc_parents in qc_constituents.items():
            if qc_child[:-3] not in self.data:
                continue

            if not set(qc_parents).issubset(set(self.data.data_vars)):
                continue

            # Ensure parents are clean integers (no NaNs)
            self.data[qc_child] = self.data[qc_parents[0]].fillna(9).astype(int)

            if len(qc_parents) > 1:
                for qc_parent in qc_parents[1:]:
                    c_vals = self.data[qc_child].values.astype(int)
                    p_vals = self.data[qc_parent].fillna(9).values.astype(int)
                    self.data[qc_child][:] = qc_combinatrix[c_vals, p_vals]

            is_nan = np.isnan(self.data[f"{qc_child[:-3]}"])
            self.data[qc_child] = xr.where(is_nan, 9, self.data[qc_child])

        # Cleanup: Assign unchecked QC to any new columns missing it
        all_vars = {v for v in self.data.data_vars if v.isupper() and "_QC" not in v and v not in self.data.dims}
        has_qc = {v[:-3] for v in self.data.data_vars if "_QC" in v}
        missing = list(all_vars - has_qc)

        if missing:
            subset = self.data[missing]
            flags = xr.where(subset.isnull(), 9, 0).astype(int).rename({v: f"{v}_QC" for v in missing})
            self.data.update(flags)
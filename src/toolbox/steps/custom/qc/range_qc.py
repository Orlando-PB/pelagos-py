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

"""Unified range QC step combining gross-range, impossible-range, and exact-value flagging."""

#### Mandatory imports ####
import numpy as np
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols
from toolbox.utils.qc_handling import merge_flags

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib


@register_qc
class range_qc(BaseQC):
    """
    Unified range test. For each variable / flag, the rule can be:

    - ``[low, high]``                 -> flag values OUTSIDE the range (gross range, default)
    - ``{outside: [low, high]}``      -> flag values outside the range (explicit)
    - ``{inside:  [low, high]}``      -> flag values inside the range (impossible range)
    - a single scalar (e.g. ``0.0``)  -> flag values exactly equal to it
    - ``{equals: [v1, v2, ...]}``     -> flag values exactly equal to any listed value

    Bounds are exclusive (``< low`` / ``> high`` for outside, ``> low`` & ``< high`` for inside),
    matching the behaviour of ``gross_range_qc`` and ``impossible_range_qc``.

    Flags are applied from most severe to least, and any point left unflagged is set to 1 (good).

    Target Variable: Any
    Flag Number: Any
    Variables Flagged: Any

    EXAMPLE
    -------
    - name: "Apply QC"
      parameters:
        qc_settings: {
            "range qc": {
              "variable_ranges": {
                "TEMP": {3: [0, 30], 4: [-2.5, 40]},                  # outside (gross range)
                "PRES": {4: {"inside": [-999, -2]}, 3: {"inside": [-2, 0]}},
                "DOXY": {4: 0.0, 3: {"equals": [-999.0, 99999.0]}},   # exact-value flagging
              },
              "also_flag": {"PRES": ["CNDC", "TEMP"]},
              "plot": ["TEMP", "PRES"]
            }
        }
      diagnostics: true
    """

    qc_name = "range qc"
    dynamic = True

    expected_parameters = {
        "variable_ranges": {},
        "also_flag": {},
        "plot": [],
    }

    parameter_schema = {
        "variable_ranges": {
            "type": dict,
            "default": None,
            "description": (
                "Per-variable flag rules. Each entry maps a variable to "
                "{flag: spec}, where spec is [low, high] (outside, default), "
                "{'outside': [low, high]}, {'inside': [low, high]}, a scalar "
                "(equals), or {'equals': [v1, v2, ...]}."
            ),
        },
        "also_flag": {
            "type": dict,
            "default": None,
            "description": (
                "Optional. Propagate a tested variable's flags onto other "
                "variables, e.g. {'PRES': ['CNDC', 'TEMP']}. Less-severe "
                "propagated flags will never overwrite worse existing flags."
            ),
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

        if not self.variable_ranges:
            raise KeyError(f"'variable_ranges' is required but missing from {self.qc_name} settings")

        self.tested_variables = list(self.variable_ranges.keys())
        self.required_variables = self.tested_variables.copy()

        self.qc_outputs = list(
            set(f"{var}_QC" for var in self.tested_variables)
            | set(f"{var}_QC" for var in sum(self.also_flag.values(), []))
        )

        self._rules = {var: self._parse_rules(var, meta) for var, meta in self.variable_ranges.items()}
        self.flags = None

    @staticmethod
    def _parse_rules(var, meta):
        """Normalise each (flag -> spec) entry into ('mode', payload)."""
        parsed = {}
        for flag, spec in meta.items():
            if isinstance(spec, dict):
                if len(spec) != 1:
                    raise ValueError(
                        f"{var} flag {flag}: expected a single key (outside/inside/equals), got {list(spec)}"
                    )
                mode, payload = next(iter(spec.items()))
                mode = mode.lower()
                if mode in ("outside", "inside"):
                    if not (isinstance(payload, (list, tuple)) and len(payload) == 2):
                        raise ValueError(f"{var} flag {flag}: '{mode}' needs [low, high]")
                    parsed[flag] = (mode, (float(payload[0]), float(payload[1])))
                elif mode == "equals":
                    values = payload if isinstance(payload, (list, tuple)) else [payload]
                    parsed[flag] = ("equals", [float(v) for v in values])
                else:
                    raise ValueError(f"{var} flag {flag}: unknown mode '{mode}'")
            elif isinstance(spec, (list, tuple)):
                if len(spec) != 2:
                    raise ValueError(f"{var} flag {flag}: list form must be [low, high]")
                parsed[flag] = ("outside", (float(spec[0]), float(spec[1])))
            elif isinstance(spec, (int, float)):
                parsed[flag] = ("equals", [float(spec)])
            else:
                raise ValueError(f"{var} flag {flag}: unsupported spec {spec!r}")
        return parsed

    def return_qc(self):
        self.data = self.data[self.required_variables]

        # Accumulate flags per output column so also_flag merges via the ARGO
        # combinatrix instead of clobbering worse flags with less-bad ones.
        out_qc: dict[str, xr.DataArray] = {}

        def _merge_in(col: str, new_qc: xr.DataArray):
            new_qc = new_qc.astype(np.int8)
            if col in out_qc:
                merged = merge_flags(out_qc[col].values, new_qc.values)
                out_qc[col] = xr.DataArray(merged, coords=new_qc.coords, dims=new_qc.dims)
            else:
                out_qc[col] = new_qc

        for var in self.tested_variables:
            qc = xr.zeros_like(self.data[var], dtype=np.int8)

            # Apply flags from most severe to least so worse flags win ties
            for flag in sorted(self._rules[var], reverse=True):
                mode, payload = self._rules[var][flag]
                values = self.data[var]

                if mode == "outside":
                    low, high = payload
                    hit = (values < low) | (values > high)
                elif mode == "inside":
                    low, high = payload
                    hit = (values > low) & (values < high)
                else:  # equals
                    hit = xr.zeros_like(values, dtype=bool)
                    for v in payload:
                        hit = hit | (values == v)

                qc = xr.where((qc == 0) & hit, flag, qc)

            # Anything left untouched is good
            qc = xr.where(qc == 0, 1, qc)

            _merge_in(f"{var}_QC", qc)

            for extra_var in self.also_flag.get(var, []):
                _merge_in(f"{extra_var}_QC", qc)

        for col, qc in out_qc.items():
            self.data[col] = qc

        self.flags = self.data[list(out_qc.keys())]
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        if len(self.plot) == 0:
            self.log_warn("Diagnostics were called but no plots were specified in the config.")
            return

        fig, axs = plt.subplots(nrows=len(self.plot), figsize=(8, 6), dpi=200)
        if len(self.plot) == 1:
            axs = [axs]

        for ax, var in zip(axs, self.plot):
            if f"{var}_QC" not in self.qc_outputs:
                self.log_warn(f"Cannot plot {var}_QC as it was not included in this test.")
                continue

            for i in range(10):
                plot_data = self.data[[var, "N_MEASUREMENTS"]].where(
                    self.data[f"{var}_QC"] == i, drop=True
                )
                if len(plot_data[var]) == 0:
                    continue
                ax.plot(
                    plot_data["N_MEASUREMENTS"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )

            for mode, payload in self._rules.get(var, {}).values():
                if mode in ("outside", "inside"):
                    for bound in payload:
                        ax.axhline(bound, ls="--", c="k")
                else:  # equals
                    for v in payload:
                        ax.axhline(v, ls=":", c="k")

            ax.set(xlabel="Index", ylabel=var, title=f"{var} Range Test")
            ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)

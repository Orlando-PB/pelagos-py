# This file is part of pelagos_py.
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

"""Unified range QC test.

Replaces the old ``gross range qc`` (flag values *outside* a good band) and
``impossible range qc`` (flag values *inside* an impossible band) with a single
test that does both. Which behaviour applies is read from the *order* of each
range's bounds: an ascending ``[low, high]`` is a good band (flag data outside it),
a descending ``[high, low]`` is an impossible band (flag data inside it). The test
can also propagate a variable's flags onto companion variables (e.g. flag PRES and
TEMP bad whenever CNDC is bad), limit the checks to a DEPTH window, and plot every
flagged variable when diagnostics are enabled.
"""

#### Mandatory imports ####
import numpy as np
from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr


# Argo flag-merge matrix used when propagating an "also_flag" flag onto a companion:
# the result of merging an existing flag (row) with a new one (column) is
# QC_COMBINATRIX[existing, new]. This is the same logic Apply QC uses to merge each
# test's flags into the dataset, copied here so a single test's own cross-flagging is
# consistent with it (e.g. a companion already flagged bad (4) is never downgraded to
# probably-bad (3) by propagation). See ApplyQC.organise_flags; this could later move
# to a shared utility.
QC_COMBINATRIX = np.array(
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

FLAG_MEANINGS = {
    0: "no QC",
    1: "good",
    2: "prob good",
    3: "prob bad",
    4: "bad",
    5: "changed",
    8: "interp",
    9: "missing",
}


@register_qc
class range_qc(BaseQC):
    """
    Flag measurements by value range. A single, configurable replacement for the
    old separate "gross range" and "impossible range" tests.

    How each ``{flag: bounds}`` entry is read depends on the shape of ``bounds``, so
    no extra parameter is needed:

    - **Ascending** ``[low, high]`` — a *good* band; data ``< low`` or ``> high``
      gets the flag. (The old "gross range" behaviour.)
    - **Descending** ``[high, low]`` — an *impossible* band; data strictly between
      the two bounds gets the flag. (The old "impossible range" behaviour.)
    - **A single scalar** ``value`` — flags exact matches (``data == value``). Handy
      for fill/filler values, e.g. ``4: 0.0`` to flag a pressure of exactly ``0`` as
      bad.

    Within a variable, entries are applied most-severe-flag-first, so on overlap the
    worse flag wins. Anything checked but not flagged is marked good (1).

    Target Variable: Any
    Flag Number: Any (user-defined)
    Variables Flagged: Any (the tested variables, plus any ``also_flag`` companions)

    EXAMPLE
    -------
    ::

        - name: "Apply QC"
          parameters:
            qc_settings:
              range qc:
                variable_ranges:
                  PRES:                 # descending -> impossible band: flag data INSIDE it
                    3: [-2.4, -5]
                    4: [-5, -.inf]
                    9: 0.0              # single scalar -> flag the exact fill value 0.0
                  TEMP:                 # ascending -> good band: flag data OUTSIDE it
                    3: [0, 30]
                    4: [-2.5, 40]
                  CNDC:
                    3: [5, 42]
                    4: [2, 45]
                also_flag:
                  CNDC: [PRES, TEMP]    # CNDC's flags propagate onto PRES & TEMP (worst wins)
                test_depth_range: [-100, 0]   # OPTIONAL: only check this DEPTH window
          diagnostics: true             # plots every flagged variable, coloured by flag
    """

    qc_name = "range qc"

    # Target variables are user-defined, so __init__ is redefined to resolve the
    # test's required/provided variables from the parameters.
    dynamic = True

    parameter_schema = {
        "variable_ranges": {
            "type": dict,
            "required": True,
            "description": "Per-variable {flag: [low, high]} ranges. An ascending pair is a good "
                           "band (data outside it is flagged); a descending pair is an impossible "
                           "band (data within it is flagged).",
        },
        "also_flag": {
            "type": dict,
            "default": {},
            "description": "Propagate a variable's flags onto companion variables, e.g. "
                           "{'CNDC': ['PRES', 'TEMP']}. Merged with the Argo matrix so the worst "
                           "flag wins.",
        },
        "test_depth_range": {
            "type": list,
            "default": None,
            "description": "Optional [min, max] DEPTH window; checks apply only to samples within it.",
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

        if self.also_flag is None:
            self.also_flag = {}

        self.tested_variables = list(self.variable_ranges.keys())

        self.required_variables = self.tested_variables.copy()
        if self.test_depth_range is not None:
            self.required_variables.append("DEPTH")

        # Outputs are the tested variables plus any companions they propagate onto.
        self.qc_outputs = list(
            {f"{var}_QC" for var in self.tested_variables}
            | {f"{var}_QC" for var in sum(self.also_flag.values(), [])}
        )

    @staticmethod
    def _band_hit(vals, bounds):
        """Return a boolean mask of the values a configured ``bounds`` entry flags.

        - A single scalar flags exact matches (e.g. a fill value such as ``0``).
        - An ascending ``[low, high]`` pair is a good band: values below ``low`` or
          above ``high`` are flagged.
        - A descending ``[high, low]`` pair is an impossible band: values strictly
          between the two bounds are flagged.

        NaNs compare ``False`` throughout, so missing values are never flagged here.
        """
        if not isinstance(bounds, (list, tuple)):
            return vals == bounds  # exact-match a single value
        a, b = bounds
        if a <= b:
            return (vals < a) | (vals > b)  # good band -> flag outside
        return (vals > b) & (vals < a)  # impossible band -> flag inside

    def return_qc(self):
        n = len(self.data["N_MEASUREMENTS"])

        # Restrict checks to a DEPTH window if requested; otherwise check everything.
        if self.test_depth_range is not None:
            depth = self.data["DEPTH"].values
            low, high = self.test_depth_range
            depth_mask = (depth >= low) & (depth <= high)
        else:
            depth_mask = np.ones(n, dtype=bool)

        qc_arrays = {}
        for var in self.tested_variables:
            vals = self.data[var].values
            qc = np.zeros(n, dtype=int)

            # Most-severe flag first so it wins where ranges overlap.
            for flag in sorted(self.variable_ranges[var], reverse=True):
                hit = self._band_hit(vals, self.variable_ranges[var][flag])
                qc[hit & depth_mask & (qc == 0)] = flag

            # Anything checked but unflagged is good.
            qc[(qc == 0) & depth_mask] = 1
            qc_arrays[var] = qc

        # Propagate flags onto companions using the Argo merge matrix, so a companion
        # keeps its own flag wherever merging the propagated one does not upgrade it
        # (e.g. an existing bad (4) is never downgraded by a propagated probably-bad (3)).
        # Companions that are not themselves tested start from "no QC" (0), so they
        # mirror the source's flags; Apply QC then merges the result with their existing
        # flags.
        for var, companions in self.also_flag.items():
            src = qc_arrays.get(var)
            if src is None:
                continue
            for companion in companions:
                base = qc_arrays.get(companion, np.zeros(n, dtype=int))
                qc_arrays[companion] = QC_COMBINATRIX[base, src]

        self.flags = xr.Dataset(coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]})
        for var, qc in qc_arrays.items():
            self.flags[f"{var}_QC"] = (("N_MEASUREMENTS",), qc)

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        # Auto-plot every variable this test flagged: the tested variables first
        # (they get range lines), then any companions it propagated onto.
        plot_order = list(self.tested_variables)
        for companion in sum(self.also_flag.values(), []):
            if companion not in plot_order:
                plot_order.append(companion)
        plot_vars = [
            var for var in plot_order
            if var in self.data and f"{var}_QC" in self.flags
        ]
        if not plot_vars:
            return

        # Use TIME on the x-axis when available, otherwise the measurement index.
        if "TIME" in self.data:
            x = self.data["TIME"].values
            xlabel = "Time"
        else:
            x = self.data["N_MEASUREMENTS"].values
            xlabel = "Index"

        n_vars = len(plot_vars)
        fig, axes = plt.subplots(
            n_vars, 1, sharex=True, figsize=(10, 2.6 * n_vars + 1.5), dpi=150
        )
        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, plot_vars):
            vals = self.data[var].values
            qc = self.flags[f"{var}_QC"].values

            # Plot points coloured by flag (drawn low-to-high so worse flags sit on top).
            for flag in range(10):
                mask = qc == flag
                if not np.any(mask):
                    continue
                ax.plot(
                    x[mask], vals[mask], ls="", marker="o", markersize=2.5,
                    alpha=0.8, color=flag_cols[flag],
                    label=f"{flag} ({FLAG_MEANINGS.get(flag, 'n/a')})", zorder=flag,
                )

            # Range boundaries for variables that define their own ranges (a single
            # scalar is drawn as one line).
            if var in self.variable_ranges:
                for flag, bounds in self.variable_ranges[var].items():
                    bound_list = bounds if isinstance(bounds, (list, tuple)) else [bounds]
                    for bound in bound_list:
                        if np.isfinite(bound):
                            ax.axhline(
                                bound, ls="--", lw=1, alpha=0.6,
                                color=flag_cols.get(flag, "k"),
                            )

            ax.set_ylabel(var, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=8)
            if var == "PRES":
                ax.invert_yaxis()
            ax.legend(
                title="Flag", loc="center left", bbox_to_anchor=(1.01, 0.5),
                fontsize=7, framealpha=0.9, fancybox=True,
            )

        axes[-1].set_xlabel(xlabel, fontsize=8)
        fig.suptitle("Range QC", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.83, 1])
        plt.show(block=True)

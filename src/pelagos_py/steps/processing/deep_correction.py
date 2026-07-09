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

"""Generic deep-value (dark offset) correction for profile data.

Estimates a constant baseline (``dark_value``) from the deepest measurements of
the first few deep profiles and subtracts it from the whole record. Written for
chlorophyll fluorescence but works for any variable with a deep signal-free
region (the offset and thresholds are all parameters)."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin

#### Custom imports ####
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#: Shallowest depth still treated as a plausible "deep", signal-free region; a
#: ``depth_threshold`` below this warns the user (see ``compute_dark_value``).
MIN_DEEP_THRESHOLD = 300


@register_step
class deep_correction(BaseStep, QCHandlingMixin):
    """
    Subtract a deep dark offset from a profile variable.

    Many sensors read a small, roughly constant baseline where the true signal
    should be zero — for example chlorophyll fluorescence in deep, dark water.
    This step estimates that baseline (the *dark value*) from the deepest
    measurements and subtracts it from the whole record, writing the result to
    an ``_ADJUSTED`` companion variable. It is written for chlorophyll but works
    for any variable with a deep, signal-free region.

    The dark value is estimated as follows:

    1. Find profiles whose ``depth_var`` reaches past ``depth_threshold``
       (deeper = larger, e.g. ``PRES > 950``) and take the first ``n_profiles``.
    2. Smooth each profile with a ``smoothing_window``-point rolling median.
    3. Keep the deep points (below the threshold) that are finite and below
       ``max_valid_value`` (guards against non-dark readings).
    4. Take each profile's minimum deep value, provided it has at least
       ``min_valid_points`` valid deep points.
    5. Use the median of those per-profile minima as the dark value.

    A ``dark_value`` supplied via config skips this estimation and is subtracted
    directly.

    Parameters
    ----------
    name : str
        Name identifier for this step instance.
    parameters : dict, optional
        Configuration parameters for the correction (see ``parameter_schema``).
    diagnostics : bool, optional
        Whether to generate diagnostic visualizations. Default is False.
    context : dict, optional
        Processing context dictionary.

    Attributes
    ----------
    step_name : str
        Identifier for this processing step. Set to "Deep Correction".

    Examples
    --------
    A minimal config only needs the variable to correct; every other parameter
    has a sensible default::

        - name: "Deep Correction"
          parameters:
            apply_to: "CHLA"
          diagnostics: true
    """

    step_name = "Deep Correction"
    required_variables = ["PROFILE_NUMBER"]
    provided_variables = []

    parameter_schema = {
        "apply_to": {
            "type": str,
            "default": "CHLA",
            "description": "Name of the variable to apply the correction to.",
        },
        "dark_value": {
            "type": float,
            "default": None,
            "description": "Dark offset to subtract; if null it is computed from the data.",
        },
        "depth_var": {
            "type": str,
            "default": "PRES",
            "description": "Vertical-coordinate variable used to find deep data (deeper = larger).",
        },
        "depth_threshold": {
            "type": float,
            "default": 950,
            "description": "Only data where depth_var exceeds this is used to compute the dark value.",
            "unit": "dbar",
        },
        "n_profiles": {
            "type": int,
            "default": 5,
            "description": "Number of deep profiles (in order of occurrence) to use.",
        },
        "smoothing_window": {
            "type": int,
            "default": 5,
            "description": "Rolling-median window (points) applied along each profile before sampling.",
        },
        "max_valid_value": {
            "type": float,
            "default": 5,
            "description": "Deep values at or above this are ignored (guards against non-dark readings).",
        },
        "min_valid_points": {
            "type": int,
            "default": 3,
            "description": "A profile must have at least this many valid deep points to contribute.",
        },
    }

    def run(self):
        self.filter_qc()

        # Resolve the input/output variable names (OG1 _ADJUSTED convention).
        self.apply_to, self.output_as = self.resolve_variables()

        self.compute_dark_value()
        self.apply_dark_correction()

        self.reconstruct_data()
        self.update_qc()

        # Generate new QC if a non-adjusted variable was used in processing (this
        # causes an _ADJUSTED variable to be added).
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def resolve_variables(self):
        # OG1 convention: processing a raw variable produces an _ADJUSTED
        # companion, and an existing _ADJUSTED variable is preferred as the
        # input and edited in place.
        apply_to = self.apply_to
        if apply_to not in self.data.data_vars:
            raise KeyError(
                f"[{self.step_name}] The variable {apply_to} does not exist in the data."
            )

        if not apply_to.endswith("_ADJUSTED") and f"{apply_to}_ADJUSTED" in self.data.data_vars:
            self.log(
                f"User requested processing on {apply_to} but {apply_to}_ADJUSTED "
                f"already exists. Using {apply_to}_ADJUSTED..."
            )
            apply_to = f"{apply_to}_ADJUSTED"

        output_as = apply_to if apply_to.endswith("_ADJUSTED") else f"{apply_to}_ADJUSTED"

        self.log(f"Processing {apply_to}...")
        return apply_to, output_as

    def compute_dark_value(self):
        # Per-profile diagnostics for plotting; left empty when dark_value is
        # supplied via config (no estimation needed).
        self._profile_diagnostics = {}

        # A dark value supplied via config short-circuits the computation.
        if self.dark_value is not None:
            self.log(f"Using dark value from config: {self.dark_value}")
            return

        var, depth = self.apply_to, self.depth_var
        self.log(
            f"Computing dark value from the first {self.n_profiles} profiles "
            f"reaching {depth} > {self.depth_threshold}."
        )

        if self.depth_threshold < MIN_DEEP_THRESHOLD:
            self.log_warn(
                f"depth_threshold ({self.depth_threshold}) is shallower than "
                f"{MIN_DEEP_THRESHOLD} {depth}; this may not be deep enough for a "
                "signal-free dark region and could bias the dark value."
            )

        missing_vars = {"PROFILE_NUMBER", depth, var} - set(self.data.data_vars)
        if missing_vars:
            raise KeyError(
                f"[{self.step_name}] {missing_vars} could not be found in the data."
            )

        df = self.data[["PROFILE_NUMBER", depth, var]].to_pandas()
        df = df.dropna(subset=["PROFILE_NUMBER", depth])

        # Profiles reaching past the threshold, kept in order of occurrence.
        deep_reach = df.groupby("PROFILE_NUMBER")[depth].max()
        deep_profiles = deep_reach[deep_reach > self.depth_threshold].index.to_numpy()
        if len(deep_profiles) == 0:
            raise ValueError(
                f"[{self.step_name}] No profiles reach past the depth threshold. "
                "Try adjusting 'depth_threshold' or 'depth_var'."
            )
        selected = deep_profiles[: self.n_profiles]

        minima = []
        for profile_number in selected:
            profile = df[df["PROFILE_NUMBER"] == profile_number].sort_values(depth)

            # Smooth along the full profile, then sample the deep region.
            smoothed = (
                profile[var]
                .rolling(self.smoothing_window, min_periods=1, center=True)
                .median()
            )
            deep_mask = (
                (profile[depth] > self.depth_threshold)
                & np.isfinite(smoothed)
                & (smoothed < self.max_valid_value)
            )

            record = {
                "depth": profile[depth].to_numpy(),
                "raw": profile[var].to_numpy(),
                "smoothed": smoothed.to_numpy(),
                "deep_mask": deep_mask.to_numpy(),
            }
            self._profile_diagnostics[profile_number] = record

            deep_vals = smoothed[deep_mask]
            if int(deep_vals.notnull().sum()) >= self.min_valid_points:
                idxmin = deep_vals.idxmin()
                record["min_value"] = float(deep_vals.loc[idxmin])
                record["min_depth"] = float(profile.loc[idxmin, depth])
                minima.append(record["min_value"])

        if len(minima) == 0:
            raise ValueError(
                f"[{self.step_name}] No profiles had at least {self.min_valid_points} "
                "valid deep points. Try adjusting the parameters."
            )

        self.dark_value = float(np.nanmedian(minima))
        self.log(
            f"\nComputed dark value: {self.dark_value:.6f} "
            f"(median of {len(minima)} profile minimums)\n"
            f"Min values range: {np.min(minima):.6f} to {np.max(minima):.6f}"
        )

    def apply_dark_correction(self):
        # Subtract the dark value to create the corrected variable.
        self.data[self.output_as] = xr.DataArray(
            self.data[self.apply_to] - self.dark_value,
            dims=self.data[self.apply_to].dims,
            coords=self.data[self.apply_to].coords,
        )

        # Copy and update attributes
        if hasattr(self.data[self.apply_to], "attrs"):
            self.data[self.output_as].attrs = self.data[self.apply_to].attrs.copy()
        self.data[self.output_as].attrs[
            "comment"
        ] = f"{self.apply_to} with dark value correction (dark_value={self.dark_value:.6f})"
        self.data[self.output_as].attrs["dark_value"] = self.dark_value

    def generate_diagnostics(self):
        """
        Plot the deep correction as a two-panel figure.

        **Left** — the deep profiles used for the estimate, each shown as its
        raw (faint) and rolling-median-smoothed (bold) trace against depth, with
        the sampled per-profile minimum marked. The depth threshold and the
        resulting dark value are drawn as reference lines, so a biofouled or
        otherwise invalid profile is easy to spot.

        **Right** — the deep values before (faint) and after subtracting the
        dark value; a valid estimate leaves the corrected values straddling
        zero.
        """
        mpl.use("tkagg")

        if not self._profile_diagnostics:
            self.log(
                "No profile diagnostics to plot (dark value was supplied via config)."
            )
            return

        fig, (ax_prof, ax_corr) = plt.subplots(
            1, 2, figsize=(12, 8), dpi=200, sharey=True
        )

        cmap = mpl.cm.viridis
        n = len(self._profile_diagnostics)
        colours = [cmap(i / max(n - 1, 1)) for i in range(n)]

        # --- Left: the deep profiles used, so biofouled/invalid ones stand out.
        for colour, (profile_number, rec) in zip(
            colours, self._profile_diagnostics.items()
        ):
            ax_prof.plot(rec["raw"], rec["depth"], c=colour, alpha=0.25, lw=0.8)
            ax_prof.plot(
                rec["smoothed"],
                rec["depth"],
                c=colour,
                lw=1.6,
                label=f"Prof. {int(profile_number)}",
            )
            if "min_value" in rec:
                ax_prof.scatter(
                    rec["min_value"],
                    rec["min_depth"],
                    c=[colour],
                    edgecolors="k",
                    zorder=5,
                    s=45,
                )

        ax_prof.axhline(
            self.depth_threshold, ls="--", c="grey", label="Depth threshold"
        )
        ax_prof.axvline(
            self.dark_value, ls="--", c="r", label=f"Dark value ({self.dark_value:.4g})"
        )
        ax_prof.invert_yaxis()  # deeper (larger depth_var) at the bottom
        ax_prof.set(
            xlabel=self.apply_to,
            ylabel=self.depth_var,
            title="Deep profiles used for dark estimate",
        )
        ax_prof.legend(fontsize=8, loc="upper right")

        # --- Right: deep values before/after correction (should straddle zero).
        for colour, rec in zip(colours, self._profile_diagnostics.values()):
            mask = rec["deep_mask"]
            deep_depth = rec["depth"][mask]
            deep_vals = rec["smoothed"][mask]
            ax_corr.plot(
                deep_vals, deep_depth, c=colour, alpha=0.3, marker="o", ls="", ms=3
            )
            ax_corr.plot(
                deep_vals - self.dark_value,
                deep_depth,
                c=colour,
                marker="o",
                ls="",
                ms=3,
            )

        ax_corr.axvline(self.dark_value, ls="--", c="r", label="Dark value (raw)")
        ax_corr.axvline(0, ls="--", c="k", label="Corrected baseline")
        ax_corr.set(
            xlabel=self.output_as,
            title="Deep values before (faint) / after correction",
        )
        ax_corr.legend(fontsize=8, loc="upper right")

        fig.suptitle(f"Deep Correction — {self.apply_to}")
        fig.tight_layout()
        plt.show(block=True)

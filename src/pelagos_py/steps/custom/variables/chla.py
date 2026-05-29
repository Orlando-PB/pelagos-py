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

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm


def check_chl_variables(self, allowed_requests):
    user_request = self.apply_to
    if user_request not in self.data.data_vars:
        raise KeyError(f"The variable {user_request} does not exist in the data.")
    if user_request not in allowed_requests:
        raise KeyError(
            f"The variable {user_request} is not permitted for [{self.step_name}]"
        )

    if f"{user_request}_ADJUSTED" in self.data.data_vars:
        self.log(
            f"User requested processing on {user_request} but {user_request}_ADJUSTED already exists. Using {user_request}_ADJUSTED..."
        )
        user_request = f"{user_request}_ADJUSTED"

    output_as = user_request + ("_ADJUSTED" if "_ADJUSTED" not in user_request else "")

    self.log(f"Processing {user_request}...")
    return user_request, output_as


@register_step
class chla_deep_correction(BaseStep, QCHandlingMixin):

    step_name = "Chla Deep Correction"
    required_variables = ["TIME", "PROFILE_NUMBER", "DEPTH"]
    provided_variables = []

    def run(self):
        """
        Example
        -------

        - name: "Chla Deep Correction"
          parameters:
            apply_to: "CHLA"
            dark_value: null
            depth_threshold: 200
        diagnostics: true
        """
        self.filter_qc()

        # Save a copy of the pre-corrected data for the diagnostics plot
        self.data_copy = self.data.copy(deep=True)

        # Check this step is being applied to a valid variable
        self.apply_to, self.output_as = check_chl_variables(
            self,
            ["CHLA", "CHLA_ADJUSTED" "CHLA_FLUORESCENCE", "CHLA_FLUORESCENCE_ADJUSTED"],
        )

        self.compute_dark_value()
        if self.dark_value is None:
            self.reconstruct_data()
            self.context["data"] = self.data
            return self.context
        self.apply_dark_correction()

        self.reconstruct_data()
        self.update_qc()

        # Generate new QC if a non-adjusted variable was used in processing
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def compute_dark_value(self):
        # Check config file for existing dark value
        if getattr(self, "dark_value", None) is not None:
            self.log(f"Using dark value from config: {self.dark_value}")
            return self.dark_value
        self.log(
            f"Computing dark value from profiles reaching >= {self.depth_threshold}m"
        )

        # Get DEPTH and CHLA data TODO: Refactor below for user input variables
        missing_vars = {"TIME", "PROFILE_NUMBER", "DEPTH", self.apply_to} - set(
            self.data.data_vars
        )
        if missing_vars:
            raise KeyError(
                f"[Chla Deep Correction] {missing_vars} could not be found in the data."
            )

        # Convert to pandas dataframe and interpolate the DEPTH data
        interp_data = self.data[
            ["TIME", "PROFILE_NUMBER", "DEPTH", self.apply_to]
        ].to_pandas()
        interp_data["DEPTH"] = (
            interp_data.set_index("TIME")["DEPTH"].interpolate().reset_index(drop=True)
        )
        interp_data = interp_data.dropna(subset=[self.apply_to, "PROFILE_NUMBER"])

        # Subset the data to only deep measurements
        interp_data = interp_data[interp_data["DEPTH"] < self.depth_threshold]

        # Remove profiles that do not have CHLA data below the threshold depth
        deep_profiles = (
            interp_data.groupby("PROFILE_NUMBER")
            .agg({self.apply_to: "count"})
            .reset_index()
        )
        deep_profiles = deep_profiles[deep_profiles[self.apply_to] > 0][
            "PROFILE_NUMBER"
        ].to_numpy()
        if len(deep_profiles) == 0:
            self.log(
                f"WARNING: [Chla Deep Correction] No profiles found below {self.depth_threshold}m. "
                f"Skipping dark correction."
            )
            self.dark_value = None
            return

        interp_data = interp_data[interp_data["PROFILE_NUMBER"].isin(deep_profiles)]

        self.chla_deep_minima = interp_data.loc[
            interp_data.groupby("PROFILE_NUMBER")[self.apply_to].idxmin(),
            ["TIME", "PROFILE_NUMBER", "DEPTH", self.apply_to],
        ]

        self.dark_value = np.nanmedian(self.chla_deep_minima[self.apply_to])
        self.log(
            f"\nComputed dark value: {self.dark_value:.6f} "
            f"(median of {len(self.chla_deep_minima)} profile minimums)\n"
            f"Min values range: {np.min(self.chla_deep_minima[self.apply_to]):.6f} "
            f"to {np.max(self.chla_deep_minima[self.apply_to]):.6f}"
        )

    def apply_dark_correction(self):
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
        mpl.use("tkagg")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        ax.plot(
            self.chla_deep_minima[self.apply_to],
            self.chla_deep_minima["DEPTH"],
            ls="",
            marker="o",
            c="b",
        )

        ax.axhline(self.depth_threshold, ls="--", c="k", label="Depth Threshold")
        ax.axvline(self.dark_value, ls="--", c="r", label="Dark Value")
        ax.legend(loc="upper right")

        ax.set(
            xlabel=f"{self.apply_to}",
            ylabel="DEPTH",
            title="Deep Minima Values",
        )

        fig.tight_layout()
        plt.show(block=True)


@register_step
class chla_quenching_correction(BaseStep, QCHandlingMixin):

    step_name = "Chla Quenching Correction"
    required_variables = ["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]
    provided_variables = []

    def run(self):
        """
        Example
        -------

        - name: "Chla Quenching Correction"
          parameters:
            method: "Argo"
            apply_to: "CHLA"
            mld_settings: {
              "threshold_on": "TEMP",
              "reference_depth": 10,
              "threshold": 0.2
              }
            plot_profiles: []
          diagnostics: true
        """
        self.pre_qc_data = self.data.copy(deep=True)
        self.filter_qc()
        self.pre_correction_data = self.data.copy(deep=True)

        self.apply_to, self.output_as = check_chl_variables(
            self,
            ["CHLA", "CHLA_ADJUSTED" "CHLA_FLUORESCENCE", "CHLA_FLUORESCENCE_ADJUSTED"],
        )
        
        if self.apply_to != self.output_as:
            self.data[self.output_as] = self.data[self.apply_to]

        # Get the function call for the specified method
        methods = {"argo": self.apply_xing2012_quenching_correction}
        if self.method.lower() not in methods.keys():
            raise KeyError(f"Method {self.method} is not supported")
        method_function = methods[self.method.lower()]

        if self.method.lower() in ["argo"]:
            self.sun_args = (
                self.data[["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]]
                .to_pandas()
                .dropna()
            )

            # only look at the values nearest the surface and find when and where they were taken
            self.sun_args = (
                self.sun_args.groupby(["PROFILE_NUMBER"])
                .apply(lambda x: x.nlargest(50, "DEPTH"))
                .reset_index(drop=True)
                .groupby(["PROFILE_NUMBER"])
                .agg({var: "median" for var in ["TIME", "LATITUDE", "LONGITUDE"]})
            )

        method_variable_requirements = {
            "argo": {
                "PROFILE_NUMBER",
                "DEPTH",
                self.apply_to,
                self.mld_settings["threshold_on"],
            }
        }
        data_subset = self.data[list(method_variable_requirements[self.method.lower()])]

        # Apply the checks across individual profiles
        profile_numbers = np.unique(
            data_subset["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS")
        )
        for profile_number in tqdm(
            profile_numbers, colour="green", desc="\033[97mProgress\033[0m", unit="prof"
        ):

            # Subset the data
            profile = data_subset.where(
                data_subset["PROFILE_NUMBER"] == profile_number, drop=True
            )

            corrected_chla, meta = method_function(profile)
            
            self.diagnostic_meta[profile_number] = meta
            self.stats[meta["status"]] += 1

            profile_indices = np.where(self.data["PROFILE_NUMBER"] == profile_number)
            self.data[self.output_as][profile_indices] = corrected_chla

        self.reconstruct_data()
        self.update_qc()

        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        self.log("\n--- Quenching Correction Summary ---")
        self.log(f"Total profiles evaluated: {self.stats['total']}")
        self.log(f"Successfully corrected:   {self.stats['corrected']}")
        self.log(f"Skipped (Night time):     {self.stats['skipped_night']}")
        self.log(f"Skipped (No valid MLD):   {self.stats['skipped_no_mld']}")
        self.log(f"Skipped (No CHLA data):   {self.stats['skipped_no_data']}")
        self.log(f"Skipped (Missing GPS):    {self.stats['skipped_no_gps']}\n")

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def calculate_mld(self, profile):
        for k, v in self.mld_settings.items():
            setattr(self, k, v)

        profile_subset = profile.where(
            profile["DEPTH"] <= self.reference_depth, drop=True
        ).dropna(dim="N_MEASUREMENTS", subset=["DEPTH", self.threshold_on])

        if len(profile_subset["DEPTH"]) == 0:
            return np.nan

        reference_point = profile_subset.isel(
            {"N_MEASUREMENTS": np.nanargmax(profile_subset["DEPTH"])},
        )
        
        if reference_point["DEPTH"] < 2 * self.reference_depth:
            return np.nan

        reference_value = reference_point[self.threshold_on]
        profile_subset["delta"] = profile_subset[self.threshold_on] - reference_value

        profile_subset = profile_subset.where(
            np.abs(profile_subset["delta"]) >= np.abs(self.threshold), drop=True
        )

        mld_value = np.nan
        if len(profile_subset["DEPTH"]) != 0:
            mld_value = float(profile_subset.isel({"N_MEASUREMENTS": 0})["DEPTH"])
        return mld_value

    def apply_xing2012_quenching_correction(self, profile):
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        N = len(chlf)

        meta = {
            "status": "skipped_no_data",
            "mld": np.nan,
            "sun_angle": np.nan,
            "z_qd": np.nan
        }

        if len(profile["PROFILE_NUMBER"]) == 0 or np.isnan(profile["PROFILE_NUMBER"].values[0]):
            return chlf, meta
            
        profile_number = int(profile["PROFILE_NUMBER"].values[0])

        if profile_number not in self.sun_args.index:
            meta["status"] = "skipped_no_gps"
            return chlf, meta

        time, lat, long = self.sun_args.loc[profile_number].to_numpy()

        if pd.isna(time) or pd.isna(lat) or pd.isna(long):
            meta["status"] = "skipped_no_gps"
            return chlf, meta

        time_utc = pd.to_datetime(time)
        if time_utc.tzinfo is None:
            time_utc = time_utc.tz_localize("UTC")

        solar_position = pvlib.solarposition.get_solarposition(time_utc, lat, long)
        sun_angle = solar_position["elevation"].values
        if (
            sun_angle <= 0
            or N == 0
            or len(depth) != N
            or not np.isfinite(mld)
            or mld >= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        # --- Identify max F_Chl within MLD
        # TODO: -DEPTH
        within_mld = depth >= mld
        if not np.any(within_mld):
            return chlf, meta

        chlf_mld = np.where(within_mld, chlf, np.nan)

        if np.all(np.isnan(chlf_mld)):
            return chlf, meta

        idx_max, chlf_max = np.nanargmax(chlf_mld), np.nanmax(chlf_mld)
        chlf_max_depth = float(depth[idx_max])

        # --- Apply correction: flatten shallower than z_qd
        # TODO: -DEPTH
        chl_corr = np.copy(chlf)
        chl_corr[(depth >= chlf_max_depth) & (~np.isnan(chlf))] = chlf_max

        return chl_corr, meta

    def generate_diagnostics(self):
        # --- Configurable Plot Variables ---
        MIN_POINTS_TO_PLOT = 100
        COLOR_SUN = "orange"
        COLOR_RAW = "lightgrey"
        COLOR_UNCORRECTED = "indianred"
        COLOR_CORRECTED = "steelblue"
        PLOT_SIZE_OVERVIEW = (14, 8)
        PLOT_SIZE_PROFILES = (12, 5)

        mpl.use("tkagg")

        fig_overview, (ax_sun, ax_chla) = plt.subplots(
            2, 1, figsize=PLOT_SIZE_OVERVIEW, sharex=True, gridspec_kw={'height_ratios': [1, 3]}, dpi=150
        )
        
        time_vals = self.data["TIME"].values
        
        # Sort values chronologically to prevent zigzag plotting artifacts
        sun_args_sorted = self.sun_args.sort_values(by="TIME")
        sun_times = pd.to_datetime(sun_args_sorted["TIME"].values)
        
        sun_angles = []
        for t, lat, lon in zip(sun_times, sun_args_sorted["LATITUDE"], sun_args_sorted["LONGITUDE"]):
            t_utc = t.tz_localize("UTC") if t.tzinfo is None else t
            sun_angles.append(pvlib.solarposition.get_solarposition(t_utc, lat, lon)["elevation"].values[0])
            
        ax_sun.plot(sun_times, sun_angles, color=COLOR_SUN, lw=1.5)
        ax_sun.axhline(0, color="black", ls="--", lw=1)
        ax_sun.set_ylabel("Sun Elevation (deg)")
        ax_sun.set_title("Deployment Overview: Sun Elevation and CHLA Adjustments")
        ax_sun.grid(True, alpha=0.3)
        ax_sun.fill_between(sun_times, sun_angles, 0, where=(np.array(sun_angles) > 0), color="yellow", alpha=0.2)
        ax_sun.fill_between(sun_times, sun_angles, 0, where=(np.array(sun_angles) <= 0), color="grey", alpha=0.2)

        ax_chla.scatter(
            self.pre_qc_data["TIME"].values, 
            self.pre_qc_data[self.apply_to].values, 
            c=COLOR_RAW, s=5, alpha=0.5, label="Raw (Failed QC)"
        )
        ax_chla.scatter(
            time_vals, 
            self.pre_correction_data[self.apply_to].values, 
            c=COLOR_UNCORRECTED, s=5, alpha=0.7, label="Uncorrected"
        )
        ax_chla.scatter(
            time_vals, 
            self.data[self.output_as].values, 
            c=COLOR_CORRECTED, s=5, alpha=0.7, label="Corrected"
        )
        
        ax_chla.set_ylabel(self.apply_to)
        ax_chla.set_xlabel("Time")
        ax_chla.legend(loc="upper right")
        ax_chla.grid(True, alpha=0.3)
        
        fig_overview.tight_layout()
        fig_overview.show()

        if len(self.plot_profiles) == 0:
            return

        nrows = int(np.ceil(len(self.plot_profiles) / 3))
        fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(12, nrows * 6), dpi=200)

        for profile_number, ax in zip(self.plot_profiles, axs.flatten()):

            for data, var, col, label in zip(
                [self.data_copy, self.data],
                [self.apply_to, self.output_as],
                ["r", "b"],
                ["Uncorrected", "Corrected"],
            ):
                # Select the raw profile data
                profile = data.where(
                    data["PROFILE_NUMBER"] == profile_number, drop=True
                ).dropna(dim="N_MEASUREMENTS", subset=[var, "DEPTH"])

                if len(profile[var]) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"Missing Data\n--Prof. {profile_number}--",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                ax.plot(
                    profile[var],
                    profile["DEPTH"],
                    c=col,
                    ls="",
                    marker="o",
                    label=label,
                )

                ax.set(
                    xlabel=self.apply_to,
                    ylabel="DEPTH",
                )

                ax.legend(title=f"Prof. {profile_number}", loc="lower right")

        fig_profs.suptitle("Quenching Correction: Profile Level Diagnostics")
        fig_profs.tight_layout()
        plt.show(block=True)

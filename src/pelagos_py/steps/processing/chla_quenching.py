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

"""Pipeline step for correcting chlorophyll-a fluorescence for non-photochemical quenching."""

# Quenching correction methods. When implementing one, add the method key to
# ``methods_requiring_sun`` if it needs the solar elevation angle.
#   [x] xing2012      (Xing et al. 2012)
#   [x] biermann2015  (Biermann et al. 2015)
#   [x] hemsley2015   (Hemsley et al. 2015)
#   [x] xing2018      (Xing et al. 2018)
#   [x] terrats2020   (Terrats et al. 2020)
#   [x] thomalla2018  (Thomalla et al. 2018)
#   [ ] swart2015     (Swart et al. 2015)
#   [ ] sackmann2008  (Sackmann et al. 2008)

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
from scipy.stats import linregress
from tqdm import tqdm

#: Diagnostics tuning for the method-comparison panels.
DAY_MIN_ELEVATION = 5.0  #: solar elevation (deg) above which a profile is "day".
NIGHT_MAX_ELEVATION = -5.0  #: solar elevation (deg) below which a profile is "night".
COMPARE_BIN_METRES = 5.0  #: depth bin (m) for pairing day/night median fluorescence.
COMPARE_SURFACE_LIMIT_METRES = 50.0  #: only bins within this depth of the surface are scored (the quenching layer, where methods differ).
MAX_COMPARE_PROFILES = 200  #: cap on day profiles run through every method.
TIMESERIES_DEPTH_LIMIT = 180.0  #: only the top this-many metres are shown in the timeseries.
SECTION_MARKER_SIZE = 1.5  #: scatter marker size (points^2) for the section plots.

#: Night-reference tuning for the 'hemsley2015'/'thomalla2018' methods.
NIGHT_REF_BIN_METRES = 1.0  #: depth bin (m) for averaging nighttime profiles into a reference.
HEMSLEY_REGRESSION_DEPTH = 60.0  #: top-of-water depth (m) over which the Hemsley regression is fit.


def estimate_euphotic_depth(par, depth):
    """Estimate the euphotic depth (Zeu) from a downwelling PAR profile.

    Fits ``ln(PAR)`` against depth over the profile (Beer-Lambert exponential
    attenuation) and takes Zeu as the 1% light level, ``Zeu = ln(100) / Kd``.

    Parameters
    ----------
    par : array-like
        Downwelling PAR profile (positive values only are used).
    depth : array-like
        Depth (metres, positive down) at each PAR value.

    Returns
    -------
    float
        Euphotic depth (metres, positive down), or ``NaN`` when the fit is
        invalid (fewer than 4 valid points, non-physical slope, or Zeu beyond
        the ~186 m clear-water limit of Morel & Maritorena 2001).
    """
    par = np.asarray(par, dtype=float)
    depth = np.asarray(depth, dtype=float)

    # Only finite, positive PAR at finite depths can be log-fitted.
    mask = np.isfinite(par) & (par > 0) & np.isfinite(depth)
    if np.sum(mask) < 4:
        return np.nan

    z = depth[mask]
    y = np.log(par[mask])
    if z[0] > z[-1]:  # regression wants increasing depth
        z = z[::-1]
        y = y[::-1]

    slope = linregress(z, y).slope
    # Reject non-physical attenuation (too clear or too turbid).
    if not np.isfinite(slope) or slope >= -0.005 or slope <= -1.0:
        return np.nan

    zeu = 4.605 / (-slope)
    return zeu if zeu <= 186 else np.nan


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
class chla_quenching_correction(BaseStep, QCHandlingMixin):

    step_name = "CHLA Quenching"
    # MLD, backscatter and PAR are only needed by some methods, so they are
    # checked at run time against the selected method rather than required here.
    required_variables = ["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]
    provided_variables = []

    #: Method keys whose correction needs the solar-elevation angle, so the
    #: per-profile sun inputs are only computed when one of them is selected.
    methods_requiring_sun = {
        "xing2012",
        "biermann2015",
        "xing2018",
        "terrats2020",
        "hemsley2015",
        "thomalla2018",
    }

    parameter_schema = {
        "method": {
            "type": str,
            "default": "xing2012",
            "options": [
                "xing2012",
                "biermann2015",
                "hemsley2015",
                "xing2018",
                "terrats2020",
                "thomalla2018",
                "swart2015",
                "sackmann2008",
            ],
            "description": (
                "Quenching correction method. Implemented: 'xing2012' (MLD-based), "
                "'biermann2015' (euphotic-depth-based, needs PAR), 'xing2018' and "
                "'terrats2020' (backscatter-based, need BBP + PAR + MLD), "
                "'hemsley2015' (global night fluorescence-bbp regression, needs BBP + PAR) "
                "and 'thomalla2018' (per-night fl:bbp ratio profile, needs BBP + PAR). "
                "The remaining options are placeholders."
            ),
        },
        "apply_to": {
            "type": str,
            "default": "CHLA",
            "description": "Name of the variable to apply the correction to.",
        },
        "bbp_var": {
            "type": str,
            "default": "BBP700",
            "description": "Backscatter variable used by 'xing2018'/'terrats2020'.",
        },
        "par_var": {
            "type": str,
            "default": "DOWNWELLING_PAR",
            "description": (
                "Downwelling PAR variable used to derive the euphotic depth "
                "('biermann2015') and the iPAR=15 depth ('xing2018'/'terrats2020')."
            ),
        },
        "plot_profiles": {
            "type": list,
            "default": [],
            "description": "Profile numbers to plot in diagnostics.",
        },
    }

    def run(self):
        """
        Example
        -------
        ::

            - name: "CHLA Quenching"
              parameters:
                method: "xing2012"
                apply_to: "CHLA"
                plot_profiles: []
              diagnostics: true

        The mixed layer depth is read from the ``MLD`` variable, which must be
        produced by a preceding Mixed Layer Depth step.
        """

        self.filter_qc()

        # required for plotting the unprocessed data later
        self.data_copy = self.data.copy(deep=True)

        # Check this step is being applied to a valid variable.
        self.apply_to, self.output_as = check_chl_variables(
            self,
            ["CHLA", "CHLA_ADJUSTED" "CHLA_FLUORESCENCE", "CHLA_FLUORESCENCE_ADJUSTED"],
        )
        # If a new "_ADJUSTED" variable will be needed, create it
        if self.apply_to != self.output_as:
            self.data[self.output_as] = self.data[self.apply_to]

        # Get the function call for the specified method
        method_key = self.method.lower()
        methods = {
            "xing2012": self.apply_xing2012_quenching_correction,
            "biermann2015": self.apply_biermann2015_quenching_correction,
            "hemsley2015": self.apply_hemsley2015_quenching_correction,
            "xing2018": self.apply_xing2018_quenching_correction,
            "terrats2020": self.apply_terrats2020_quenching_correction,
            "thomalla2018": self.apply_thomalla2018_quenching_correction,
            "swart2015": self.apply_swart2015_quenching_correction,
            "sackmann2008": self.apply_sackmann2008_quenching_correction,
        }
        if method_key not in methods:
            raise KeyError(
                f"Method '{self.method}' is not supported. "
                f"Choose from: {', '.join(methods)}"
            )
        method_function = methods[method_key]
        # Kept so the diagnostics can re-run the configured method over a single
        # profile to capture its decision internals (see ``_explain_profile``).
        self._method_function = method_function

        # Methods differ in which auxiliary variables they need; check the ones
        # the chosen method relies on are present before doing any work.
        needs_mld = method_key in ("xing2012", "xing2018", "terrats2020")
        needs_par = method_key in (
            "biermann2015", "xing2018", "terrats2020", "hemsley2015", "thomalla2018"
        )
        needs_bbp = method_key in (
            "xing2018", "terrats2020", "hemsley2015", "thomalla2018"
        )
        missing = []
        if needs_mld and "MLD" not in self.data.data_vars:
            missing.append("MLD (add a Mixed Layer Depth step beforehand)")
        if needs_par and self.par_var not in self.data.data_vars:
            missing.append(f"{self.par_var} (PAR; set 'par_var' or add a step providing it)")
        if needs_bbp and self.bbp_var not in self.data.data_vars:
            missing.append(f"{self.bbp_var} (backscatter; set 'bbp_var' or add a BBP step beforehand)")
        if missing:
            self.halt(f"Method '{self.method}' requires: " + "; ".join(missing) + ".")

        # if the method requires sunlight angle, find the inputs for the sun angle calculation
        if method_key in self.methods_requiring_sun:
            self.sun_args = (
                self.data[["PROFILE_NUMBER", "TIME", "DEPTH", "LATITUDE", "LONGITUDE"]]
                .to_pandas()
                .dropna()
            )

            # only look at the values nearest the surface and find when and where they were taken
            self.sun_args = (
                self.sun_args.groupby("PROFILE_NUMBER", group_keys=True)
                .apply(lambda x: x.nlargest(50, "DEPTH"), include_groups=False)
                .groupby(level="PROFILE_NUMBER")
                .agg({var: "median" for var in ["TIME", "LATITUDE", "LONGITUDE"]})
            )

        # Methods that correct day profiles against nighttime references build
        # those references once, up front, from the whole (uncorrected) dataset
        # (the per-profile loop below only sees one profile at a time). With
        # diagnostics on, build both night-based methods' references (when BBP +
        # PAR are present) so they can be scored in the comparison panel too,
        # not just when one of them is the configured method.
        build_refs = {method_key} & {"hemsley2015", "thomalla2018"}
        if (
            self.diagnostics
            and hasattr(self, "sun_args")
            and self.bbp_var in self.data.data_vars
            and self.par_var in self.data.data_vars
        ):
            build_refs |= {"hemsley2015", "thomalla2018"}
        for ref_method in build_refs:
            self._build_night_references(ref_method)

        # Subset the data to just the variables the chosen method needs.
        subset_vars = ["PROFILE_NUMBER", "DEPTH", self.apply_to]
        if needs_mld:
            subset_vars.append("MLD")
        if needs_bbp:
            subset_vars.append(self.bbp_var)
        if needs_par:
            subset_vars.append(self.par_var)
        data_subset = self.data[subset_vars]

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

            corrected_chla = method_function(profile)

            # Stitch back into the full data
            profile_indices = np.where(self.data["PROFILE_NUMBER"] == profile_number)
            self.data[self.output_as][profile_indices] = corrected_chla

        self.reconstruct_data()
        self.update_qc()

        # Generate new QC if a non-adjusted variable was used in processing (this causes an _ADJUSTED variable to be added)"
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def apply_xing2012_quenching_correction(self, profile):
        """
        Apply non-photochemical quenching (NPQ) correction following
        Xing et al. (2012, *JGR–Oceans*, 117:C01019).

        The maximum fluorescence within the mixed-layer depth (MLD)
        is taken as the non-quenched reference. All shallower
        (PRES < z_qd) values are adjusted upward to that maximum.
        Correction is only applied when solar elevation > 0°.

        Parameters
        ----------
        chlf : array-like of shape (N,)
            Uncorrected chlorophyll fluorescence profile F_Chl(PRES).
        pres : array-like of shape (N,)
            Pressure (dbar), increasing with depth.
        mld : float
            Mixed-layer depth (m or dbar).
        sun_angle : float
            Solar elevation angle (degrees). NPQ correction is applied
            only if `sun_angle > 0`.

        Returns
        -------
        chl_corr : ndarray of shape (N,)
            NPQ-corrected fluorescence profile.
        npq : ndarray of shape (N,)
            NPQ index = (chl_corr − chlf) / chlf.
        z_qd : float
            Quenching depth (dbar): pressure of maximum fluorescence
            within the MLD. NaN if not computable or if night-time.

        Notes
        -----
        • No correction is applied if solar elevation ≤ 0° (nighttime).
        • Shallower than z_qd → fluorescence set to Fmax (non-quenched reference).
        • Below MLD → unchanged.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        N = len(chlf)

        # --- Read the MLD for this profile (a per-profile scalar broadcast across
        # its measurements by the Mixed Layer Depth step)
        mld_values = np.asarray(profile["MLD"].values, dtype=float)
        finite_mld = mld_values[np.isfinite(mld_values)]
        mld = float(finite_mld[0]) if finite_mld.size else np.nan

        # --- Night-time or invalid inputs: skip correction
        profile_number = int(profile["PROFILE_NUMBER"][0])
        time, lat, long = self.sun_args.loc[profile_number].to_numpy()
        time_utc = pd.to_datetime(time).tz_localize("UTC")
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
            return chlf

        chlf_mld = np.where(within_mld, chlf, np.nan)
        idx_max, chlf_max = np.nanargmax(chlf_mld), np.nanmax(chlf_mld)
        chlf_max_depth = float(depth[idx_max])

        # --- Apply correction: flatten shallower than z_qd
        # TODO: -DEPTH
        chl_corr = np.copy(chlf)
        chl_corr[(depth >= chlf_max_depth) & (~np.isnan(chlf))] = chlf_max

        self._explain(
            profile,
            depth=depth,
            chlf=chlf,
            mld=mld,
            z_ref=chlf_max_depth,
            f_ref=chlf_max,
        )
        return chl_corr

    # ------------------------------------------------------------------
    # Placeholder methods for further quenching corrections.
    # Each takes a single-profile dataset (same interface as
    # ``apply_xing2012_quenching_correction``) and should return the
    # NPQ-corrected fluorescence array for that profile. When implementing
    # one, add its key to ``methods_requiring_sun`` if it needs the solar
    # elevation angle.
    # ------------------------------------------------------------------
    def _sun_elevation(self, profile):
        """Solar elevation (degrees) for a profile from its median surface fix.

        Uses the per-profile median TIME/LATITUDE/LONGITUDE gathered in ``run``
        (``self.sun_args``); a value > 0 means daytime.
        """
        return self._sun_elevation_for(int(profile["PROFILE_NUMBER"][0]))

    def _sun_elevation_for(self, profile_number):
        """Cached solar elevation (degrees) for a profile number.

        The diagnostics run every method over many profiles, so the pvlib
        solar-position lookup is memoised per profile.
        """
        cache = getattr(self, "_sun_cache", None)
        if cache is None:
            cache = self._sun_cache = {}
        if profile_number not in cache:
            time, lat, long = self.sun_args.loc[profile_number].to_numpy()
            time_utc = pd.to_datetime(time).tz_localize("UTC")
            solar_position = pvlib.solarposition.get_solarposition(time_utc, lat, long)
            cache[profile_number] = float(solar_position["elevation"].values[0])
        return cache[profile_number]

    def apply_biermann2015_quenching_correction(self, profile):
        """
        Apply non-photochemical quenching (NPQ) correction following
        Biermann et al. (2015, *Ocean Science*, 11:83-91).

        The maximum fluorescence below the euphotic depth (Zeu) is taken as the
        non-quenched reference; all shallower values are lifted to it. Zeu is
        derived per profile from the PAR profile (1% light level). Correction is
        applied only in daytime (solar elevation > 0).

        ``DEPTH`` is negative-down, so it is converted to positive-down metres
        (``z = -DEPTH``) to match the published formulation and the PAR fit.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        par = np.asarray(profile[self.par_var].values, dtype=float)
        z = -depth  # positive down (metres below surface)
        N = len(chlf)

        sun_angle = self._sun_elevation(profile)
        zeu = estimate_euphotic_depth(par, z)
        if (
            sun_angle <= 0
            or N == 0
            or len(depth) != N
            or not np.isfinite(zeu)
            or zeu <= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        # Reference = max F_Chl below the euphotic depth.
        below_zeu = z >= zeu
        chlf_below = np.where(below_zeu, chlf, np.nan)
        if np.all(np.isnan(chlf_below)):
            return chlf

        idx_max = np.nanargmax(chlf_below)
        z_qd = z[idx_max]
        f_max = chlf[idx_max]

        # Lift everything shallower than the quenching depth to the reference.
        chl_corr = np.copy(chlf)
        chl_corr[(z <= z_qd) & (~np.isnan(chlf))] = f_max

        self._explain(
            profile,
            depth=depth,
            chlf=chlf,
            zeu=-zeu,
            z_ref=-z_qd,
            f_ref=f_max,
        )
        return chl_corr

    def apply_hemsley2015_quenching_correction(self, profile):
        """
        Apply the Hemsley et al. (2015, *Biogeosciences*, 12:7093) NPQ
        correction as applied by Thomalla et al. (2018).

        A single global regression of nighttime fluorescence against
        backscatter over the top ``HEMSLEY_REGRESSION_DEPTH`` metres,
        ``Chl_NT = m*bbp_NT + c``, is fit once for the whole deployment (see
        :meth:`_build_night_references`). For each daytime profile the fitted
        slope and intercept are then applied over the euphotic zone,
        ``Chl_DT(z) = m*bbp_DT(z) + c`` for ``0 <= z <= Zeu``. Zeu is the 1%
        light level derived from the PAR profile.

        Following Thomalla et al. (2018), the regression is applied to *all*
        daytime profiles (their study did not skip profiles with a subsurface
        maximum). ``DEPTH`` is negative-down and is converted to positive-down
        metres to match the published formulation and the PAR fit.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        par = np.asarray(profile[self.par_var].values, dtype=float)
        z = -depth  # positive down (metres below surface)
        N = len(chlf)

        regression = getattr(self, "_hemsley_regression", None)
        sun_angle = self._sun_elevation(profile)
        zeu = estimate_euphotic_depth(par, z)
        if (
            sun_angle <= 0
            or regression is None
            or N == 0
            or len(bbp) != N
            or not np.isfinite(zeu)
            or zeu <= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        m, c = regression["slope"], regression["intercept"]
        chl_corr = np.copy(chlf)
        # Replace fluorescence over the euphotic zone with the bbp-based estimate.
        fill = (z >= 0) & (z <= zeu) & np.isfinite(bbp) & (~np.isnan(chlf))
        chl_corr[fill] = m * bbp[fill] + c

        self._explain(profile, depth=depth, chlf=chlf, zeu=-zeu, z_ref=-zeu)
        return chl_corr

    def _build_night_references(self, method_key):
        """Build the nighttime references used by 'hemsley2015'/'thomalla2018'.

        Runs once in :meth:`run` before the per-profile loop. Profiles are
        classified day/night by solar elevation (``> 0`` day, ``< 0`` night)
        from the per-profile surface fix in ``self.sun_args``.

        - ``hemsley2015`` fits one global fluorescence-vs-backscatter regression
          over the top ``HEMSLEY_REGRESSION_DEPTH`` m of all nighttime data,
          stored on ``self._hemsley_regression``.
        - ``thomalla2018`` groups consecutive nighttime profiles into "nights",
          builds a depth-binned mean fluorescence / mean bbp / fl:bbp ratio
          profile per night (``self._night_refs``), and maps each daytime
          profile to its most recent *preceding* night
          (``self._thomalla_day_night``). The earliest daytime profiles, which
          have no preceding night, fall back to the nearest *following* night.
        """
        pns = [int(p) for p in self.sun_args.index]
        elev = {pn: self._sun_elevation_for(pn) for pn in pns}
        times = {pn: pd.to_datetime(self.sun_args.loc[pn, "TIME"]).value for pn in pns}
        pns_time = sorted(pns, key=lambda p: times[p])
        is_night = {pn: elev[pn] < 0 for pn in pns}

        pnum = self.data["PROFILE_NUMBER"].values
        z_all = -np.asarray(self.data["DEPTH"].values, dtype=float)
        fl_all = np.asarray(self.data[self.apply_to].values, dtype=float)
        bbp_all = np.asarray(self.data[self.bbp_var].values, dtype=float)

        if method_key == "hemsley2015":
            night_pns = [pn for pn in pns if is_night[pn]]
            mask = np.isin(pnum, night_pns)
            z, f, b = z_all[mask], fl_all[mask], bbp_all[mask]
            sel = (
                np.isfinite(z)
                & (z >= 0)
                & (z <= HEMSLEY_REGRESSION_DEPTH)
                & np.isfinite(f)
                & np.isfinite(b)
            )
            if np.sum(sel) < 5 or np.ptp(b[sel]) == 0:
                self._hemsley_regression = None
                self.log(
                    "Hemsley 2015: too few valid nighttime fluorescence/backscatter "
                    "points to fit a regression; day profiles will be left uncorrected."
                )
                return
            fit = linregress(b[sel], f[sel])
            self._hemsley_regression = {
                "slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "r2": float(fit.rvalue ** 2),
                "n": int(np.sum(sel)),
                # Raw fitted points, kept so the diagnostics can scatter them.
                "bbp": b[sel],
                "fl": f[sel],
            }
            self.log(
                f"Hemsley 2015: night regression Chl = {fit.slope:.4g}*bbp "
                f"+ {fit.intercept:.4g} (r2={fit.rvalue ** 2:.2f}, n={int(np.sum(sel))})."
            )
            return

        # thomalla2018: group consecutive nighttime profiles into nights.
        nights_members, current = [], []
        for pn in pns_time:
            if is_night[pn]:
                current.append(pn)
            elif current:
                nights_members.append(current)
                current = []
        if current:
            nights_members.append(current)

        night_refs = []
        for members in nights_members:
            mask = np.isin(pnum, members)
            ref = self._bin_night(z_all[mask], fl_all[mask], bbp_all[mask])
            if ref is None:
                continue
            ref["time"] = float(np.median([times[pn] for pn in members]))
            night_refs.append(ref)

        day_night = {}
        if night_refs:
            night_times = [ref["time"] for ref in night_refs]
            for pn in (p for p in pns if elev[p] > 0):
                dt = times[pn]
                preceding = [i for i, nt in enumerate(night_times) if nt <= dt]
                if preceding:
                    day_night[pn] = max(preceding, key=lambda i: night_times[i])
                else:  # earliest day profiles: no preceding night -> use the next one
                    day_night[pn] = min(
                        range(len(night_times)), key=lambda i: night_times[i]
                    )

        self._night_refs = night_refs
        self._thomalla_day_night = day_night
        self.log(
            f"Thomalla 2018: built {len(night_refs)} nighttime fl:bbp reference "
            f"profile(s) covering {len(day_night)} day profile(s)."
        )

    @staticmethod
    def _bin_night(z, fl, bbp):
        """Depth-binned mean fluorescence / bbp / fl:bbp ratio for a night.

        Returns a dict with ascending bin-centre depths ``z`` (positive down),
        mean fluorescence ``fl``, and the ``ratio`` (mean fl / mean bbp), or
        ``None`` if no bin has both a finite mean fluorescence and a positive
        mean backscatter.
        """
        z = np.asarray(z, dtype=float)
        fl = np.asarray(fl, dtype=float)
        bbp = np.asarray(bbp, dtype=float)
        valid = np.isfinite(z) & (z >= 0)
        if not np.any(valid):
            return None
        keys = np.floor(z[valid] / NIGHT_REF_BIN_METRES).astype(int)
        fl_v, bbp_v = fl[valid], bbp[valid]

        centres, mean_fl, ratio = [], [], []
        for k in np.unique(keys):
            in_bin = keys == k
            f = np.nanmean(fl_v[in_bin]) if np.any(np.isfinite(fl_v[in_bin])) else np.nan
            b = np.nanmean(bbp_v[in_bin]) if np.any(np.isfinite(bbp_v[in_bin])) else np.nan
            if not (np.isfinite(f) and np.isfinite(b) and b > 0):
                continue
            centres.append((k + 0.5) * NIGHT_REF_BIN_METRES)
            mean_fl.append(f)
            ratio.append(f / b)
        if not centres:
            return None
        centres = np.asarray(centres, dtype=float)
        order = np.argsort(centres)  # np.interp needs increasing x
        return {
            "z": centres[order],
            "fl": np.asarray(mean_fl, dtype=float)[order],
            "ratio": np.asarray(ratio, dtype=float)[order],
        }

    def apply_xing2018_quenching_correction(self, profile):
        """
        Apply the Xing et al. (2018, *Optics Express*, 26:24734) S08+ NPQ
        correction (deep-mixing regime only).

        Within the NPQ layer (above the shallower of the MLD and the iPAR=15
        depth) the fluorescence-to-backscatter ratio ``F_Chl / b_bp`` is
        maximised, and fluorescence is reset to ``b_bp x R_max`` across that
        layer. See :meth:`_apply_xing_terrats`.
        """
        return self._apply_xing_terrats(profile, hybrid=False)

    def apply_terrats2020_quenching_correction(self, profile):
        """
        Apply the Xing (2018) / Terrats et al. (2020, *GRL*, e2020GL089059)
        hybrid NPQ correction.

        Deep mixing (iPAR=15 depth <= MLD) uses the Xing (2018) S08+ method;
        shallow mixing (iPAR=15 depth > MLD) uses the Terrats (2020) X18_S08
        hybrid, applying the XB18 sigmoid below the MLD and a uniform
        ``b_bp x R_MLD`` above it. See :meth:`_apply_xing_terrats`.
        """
        return self._apply_xing_terrats(profile, hybrid=True)

    def _apply_xing_terrats(self, profile, hybrid):
        """Backscatter-based NPQ correction shared by 'xing2018'/'terrats2020'.

        With ``hybrid=False`` the S08+ deep-mixing branch is always used
        (Xing 2018). With ``hybrid=True`` the shallow-mixing branch (Terrats
        2020) is used when the iPAR=15 depth is deeper than the MLD.

        ``DEPTH`` and ``MLD`` are negative-down; both are converted to
        positive-down metres to match the published formulations. On any
        condition that prevents a correction (night, missing inputs, degenerate
        profile) the uncorrected fluorescence is returned unchanged.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        ipar = np.asarray(profile[self.par_var].values, dtype=float)
        z = -depth  # positive down (metres below surface)
        N = len(chlf)

        sun_angle = self._sun_elevation(profile)
        if (
            sun_angle <= 0
            or N == 0
            or len(bbp) != N
            or len(ipar) != N
            or np.all(np.isnan(chlf))
            or np.all(np.isnan(bbp))
            or np.all(np.isnan(ipar))
        ):
            return chlf

        # MLD as positive-down metres (stored negative-down, one value per profile).
        finite_mld = np.asarray(profile["MLD"].values, dtype=float)
        finite_mld = finite_mld[np.isfinite(finite_mld)]
        mld = -float(finite_mld[0]) if finite_mld.size else np.nan

        # Depth at which iPAR crosses 15 umol m-2 s-1, on the irregular grid.
        zi_par15 = self._depth_of_ipar15(z, ipar)

        # Shallow mixing: light penetrates below the mixed layer.
        shallow = hybrid and np.isfinite(zi_par15) and np.isfinite(mld) and zi_par15 > mld

        if not shallow:
            # --- Deep-mixing S08+ (Xing 2018) --------------------------------
            if not (np.isfinite(mld) and np.isfinite(zi_par15)):
                return chlf
            # NPQ layer: shallower than the shallower of MLD and the iPAR=15 depth.
            z_ref = min(mld, zi_par15)
            npq_layer = (z <= z_ref) & np.isfinite(z)
            fratio = np.divide(
                chlf, bbp, out=np.full_like(chlf, np.nan), where=(bbp != 0)
            )
            fratio_layer = np.where(npq_layer, fratio, np.nan)
            if np.all(np.isnan(fratio_layer)):
                return chlf
            idx_rmax = np.nanargmax(fratio_layer)
            r_max = fratio[idx_rmax]

            chl_corr = np.copy(chlf)
            fill = npq_layer & np.isfinite(bbp) & (~np.isnan(chlf))
            chl_corr[fill] = bbp[fill] * r_max

            self._explain(
                profile,
                branch="deep",
                depth=depth,
                ratio=fratio,
                mld=-mld,
                ipar15=-zi_par15,
                z_ref=-z_ref,
                r_max=r_max,
                rmax_depth=float(depth[idx_rmax]),
            )
        else:
            # --- Shallow-mixing X18_S08 hybrid (Terrats 2020) ----------------
            if not np.isfinite(mld):
                return chlf
            r, ipar_mid, e = 0.092, 261.0, 2.2  # XB18 sigmoid parameters.

            chl_corr = np.copy(chlf)
            below = (z > mld) & np.isfinite(z)
            # Sigmoid de-quenching below the MLD; clip PAR away from zero first.
            ipar_safe = np.clip(ipar[below], 1e-3, None)
            s = r + (1 - r) / (1 + (ipar_safe / ipar_mid) ** e)
            s = np.clip(s, r, 1.0)
            chl_corr[below] = chlf[below] / s

            # Ratio at the shallowest valid point just below the MLD.
            below_idx = np.where(below)[0]
            order = below_idx[np.argsort(z[below_idx])]
            r_mld = np.nan
            for k in order:
                if np.isfinite(chl_corr[k]) and np.isfinite(bbp[k]) and bbp[k] > 0:
                    r_mld = chl_corr[k] / bbp[k]
                    break
            if not np.isfinite(r_mld):
                return chlf

            above = (z <= mld) & np.isfinite(bbp) & (~np.isnan(chlf))
            chl_corr[above] = bbp[above] * r_mld

            self._explain(
                profile,
                branch="shallow",
                depth=depth,
                mld=-mld,
                ipar15=-zi_par15,
                z_ref=-mld,
                r_mld=r_mld,
                sigmoid_depth=-z[below],
                sigmoid_scale=s,
            )

        # Never let the correction reduce fluorescence (fmax ignores NaNs).
        return np.fmax(chlf, chl_corr)

    @staticmethod
    def _depth_of_ipar15(z, ipar):
        """Depth (positive-down m) where downwelling iPAR crosses 15, or NaN.

        Interpolates on the irregular profile grid; clamps to the deepest /
        shallowest sample when 15 lies outside the observed PAR range.
        """
        valid = np.isfinite(z) & np.isfinite(ipar)
        if np.sum(valid) < 2:
            return np.nan
        zi = z[valid]
        pi = ipar[valid]
        order = np.argsort(zi)  # surface -> deep
        zi = zi[order]
        pi = pi[order]
        if 15 <= np.min(pi):  # whole profile brighter than 15 -> deepest sample
            return float(zi[-1])
        if 15 >= np.max(pi):  # whole profile darker than 15 -> surface
            return 0.0
        # PAR decreases with depth; reverse so np.interp sees increasing x.
        return float(np.interp(15, pi[::-1], zi[::-1]))

    def apply_thomalla2018_quenching_correction(self, profile):
        """
        Apply the Thomalla et al. (2018, *L&O: Methods*, 16:132) NPQ
        correction ("this study").

        Each daytime profile is corrected against its most recent *preceding*
        night's mean fluorescence-to-backscatter ratio profile (built in
        :meth:`_build_night_references`). Above the quenching depth QD,
        fluorescence is reset to ``Flc_DT(z) = (Fl_NT(z)/bbp_NT(z)) * bbp_DT(z)``.
        Per the paper's intercomparison rule, the correction is kept only where
        it *raises* the fluorescence; otherwise the original value is retained.

        QD is the base of the quenching layer, found from the night-minus-day
        fluorescence difference within the euphotic zone (see
        :meth:`_quenching_depth`). The earliest daytime profiles have no
        preceding night and instead use the nearest following night.

        ``DEPTH`` is negative-down and is converted to positive-down metres to
        match the published formulation, the night reference, and the PAR fit.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        par = np.asarray(profile[self.par_var].values, dtype=float)
        z = -depth  # positive down (metres below surface)
        N = len(chlf)

        profile_number = int(profile["PROFILE_NUMBER"][0])
        day_night = getattr(self, "_thomalla_day_night", {})
        ref_idx = day_night.get(profile_number)
        sun_angle = self._sun_elevation(profile)
        if (
            sun_angle <= 0
            or ref_idx is None
            or N == 0
            or len(bbp) != N
            or np.all(np.isnan(chlf))
        ):
            return chlf

        ref = self._night_refs[ref_idx]
        zeu = estimate_euphotic_depth(par, z)
        if not np.isfinite(zeu) or zeu <= 0:
            return chlf

        # Night fl:bbp ratio and mean fluorescence interpolated onto day depths.
        ratio_at_z = np.interp(z, ref["z"], ref["ratio"])
        fl_night_at_z = np.interp(z, ref["z"], ref["fl"])

        qd = self._quenching_depth(z, chlf, fl_night_at_z, zeu)
        if not np.isfinite(qd):
            return chlf

        corrected = ratio_at_z * bbp
        chl_corr = np.copy(chlf)
        # Correct from the surface to the quenching depth, keeping the result
        # only where it raises the (quenched) daytime fluorescence.
        fill = (
            (z >= 0)
            & (z <= qd)
            & np.isfinite(bbp)
            & (~np.isnan(chlf))
            & np.isfinite(corrected)
            & (corrected > chlf)
        )
        chl_corr[fill] = corrected[fill]

        self._explain(
            profile,
            depth=depth,
            chlf=chlf,
            zeu=-zeu,
            z_ref=-qd,
            ref_idx=ref_idx,
        )
        return chl_corr

    @staticmethod
    def _quenching_depth(z, fl_day, fl_night, zeu):
        """Quenching depth QD (positive-down m) from the night-day fl difference.

        Follows Thomalla et al. (2018): within the euphotic zone the difference
        ``D(z) = Fl_NT(z) - Fl_DT(z)`` is anchored at its near-surface maximum
        (top 5 m) and QD is taken as the point, deeper than that anchor, giving
        the steepest gradient down to one of the five smallest absolute
        differences or a zero crossing of ``D``. Returns ``NaN`` when it cannot
        be resolved.
        """
        z = np.asarray(z, dtype=float)
        D = np.asarray(fl_night, dtype=float) - np.asarray(fl_day, dtype=float)
        mask = np.isfinite(z) & np.isfinite(D) & (z >= 0) & (z <= zeu)
        if np.sum(mask) < 3:
            return np.nan
        zz, DD = z[mask], D[mask]
        order = np.argsort(zz)  # surface -> deep
        zz, DD = zz[order], DD[order]

        # Anchor at the largest difference near the surface (top 5 m if present).
        top = zz <= 5
        anchor = np.argmax(np.where(top, DD, -np.inf)) if np.any(top) else np.argmax(DD)
        z_a, D_a = zz[anchor], DD[anchor]

        # Candidates: five smallest |D| deeper than the anchor, plus zero crossings.
        candidates = set()
        for i in np.argsort(np.abs(DD)):
            if zz[i] > z_a:
                candidates.add(int(i))
            if len(candidates) >= 5:
                break
        for i in range(len(DD) - 1):
            crossing = DD[i] == 0 or (DD[i] > 0) != (DD[i + 1] > 0)
            if crossing and zz[i + 1] > z_a:
                candidates.add(i + 1)
        if not candidates:
            return np.nan

        best_qd, best_gradient = np.nan, -np.inf
        for i in candidates:
            gradient = abs(D_a - DD[i]) / (zz[i] - z_a)
            if gradient > best_gradient:
                best_gradient, best_qd = gradient, float(zz[i])
        return best_qd

    def apply_swart2015_quenching_correction(self, profile):
        """Placeholder for the Swart et al. (2015) NPQ correction."""
        raise NotImplementedError(
            "The 'swart2015' quenching correction is not yet implemented."
        )

    def apply_sackmann2008_quenching_correction(self, profile):
        """Placeholder for the Sackmann et al. (2008) NPQ correction."""
        raise NotImplementedError(
            "The 'sackmann2008' quenching correction is not yet implemented."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    #: Display labels for the implemented methods (comparison panel titles).
    _METHOD_LABELS = {
        "none": "No correction",
        "xing2012": "Xing 2012",
        "biermann2015": "Biermann 2015",
        "xing2018": "Xing 2018",
        "terrats2020": "Terrats 2020",
        "hemsley2015": "Hemsley 2015",
        "thomalla2018": "Thomalla 2018",
    }

    def generate_diagnostics(self):
        """One figure summarising the quenching correction, in three columns.

        **Left** — every implemented method is run over the day profiles and
        scored against a nearest-in-time night reference (midnight vs midday
        fluorescence, paired per depth bin within the top
        ``COMPARE_SURFACE_LIMIT_METRES`` where quenching acts). Each panel shows
        the scatter, the 1:1 line, the regression fit, and RMSE/Bias/R2 in the
        legend; the lowest-RMSE correction method is given a highlighted border.

        **Middle** — an example day and night profile for the *configured*
        method (unchanged points, the original quenched values, and the
        corrected values), with the day profile marking the method's own
        quenching depth. A third panel shows *how* that method chose its
        correction reference: for per-profile methods the example profile's
        internals (the searched layer and the picked reference point); for the
        night-reference methods the deployment-wide object used (Hemsley's global
        night fl-bbp regression, Thomalla's night fl:bbp ratio profile).

        **Right** — the original and corrected fluorescence as depth-time
        sections, plus a map of where the correction was actually applied.
        """
        mpl.use("tkagg")

        if not hasattr(self, "sun_args"):
            self.log("Solar inputs unavailable; cannot build quenching diagnostics.")
            return

        fig = plt.figure(figsize=(19, 11), dpi=120)
        outer = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.35], wspace=0.32)

        self._draw_method_comparison(fig, outer[0, 0])
        self._draw_example_profiles(fig, outer[0, 1])
        self._draw_timeseries(fig, outer[0, 2])

        fig.suptitle(
            f"CHLA Quenching diagnostics — method: {self.method}  "
            f"({self.apply_to} -> {self.output_as})",
            fontsize=13,
            fontweight="bold",
        )
        plt.show(block=True)

    # --- Left column: method comparison -------------------------------

    def _draw_method_comparison(self, fig, subspec):
        pairs = self._day_night_pairs()
        if not pairs:
            ax = fig.add_subplot(subspec)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No day/night profile pairs\navailable for method comparison.",
                ha="center",
                va="center",
                fontsize=9,
            )
            return

        implemented = {
            "xing2012": (self.apply_xing2012_quenching_correction, {"MLD"}),
            "biermann2015": (
                self.apply_biermann2015_quenching_correction,
                {self.par_var},
            ),
            "xing2018": (
                self.apply_xing2018_quenching_correction,
                {"MLD", self.bbp_var, self.par_var},
            ),
            "terrats2020": (
                self.apply_terrats2020_quenching_correction,
                {"MLD", self.bbp_var, self.par_var},
            ),
        }
        # The night-reference methods can only be scored when their references
        # were built (i.e. when one of them is the configured method).
        if getattr(self, "_hemsley_regression", None) is not None:
            implemented["hemsley2015"] = (
                self.apply_hemsley2015_quenching_correction,
                {self.bbp_var, self.par_var},
            )
        if getattr(self, "_thomalla_day_night", None):
            implemented["thomalla2018"] = (
                self.apply_thomalla2018_quenching_correction,
                {self.bbp_var, self.par_var},
            )
        have = set(self.data_copy.data_vars)
        runnable = {k: fn for k, (fn, needs) in implemented.items() if needs <= have}

        day_pns = [d for d, _ in pairs]
        night_pns = [n for _, n in pairs]
        subsets = self._profile_subsets(set(day_pns) | set(night_pns))
        night_dv = self._raw_dv(night_pns, subsets)

        # 'none' = no-correction baseline, then each runnable method.
        results = {"none": self._score(self._raw_dv(day_pns, subsets), night_dv, pairs)}
        for key, fn in runnable.items():
            day_dv = self._run_method_over(fn, day_pns, subsets)
            results[key] = self._score(day_dv, night_dv, pairs)

        scored = {k: r for k, r in results.items() if k != "none" and r}
        best = min(scored, key=lambda k: scored[k]["rmse"]) if scored else None

        panels = ["none"] + list(runnable)
        gl = subspec.subgridspec(len(panels), 1, hspace=0.6)
        for i, key in enumerate(panels):
            ax = fig.add_subplot(gl[i, 0])
            self._draw_scatter_panel(
                ax, self._METHOD_LABELS[key], results.get(key), best == key
            )

    def _day_night_pairs(self):
        """List of ``(day_profile, night_profile)`` numbers, nearest in time.

        Capped at ``MAX_COMPARE_PROFILES`` day profiles (evenly sampled) so the
        comparison stays responsive; the cap is logged when it bites.
        """
        pns = [int(pn) for pn in self.sun_args.index]
        elev = {pn: self._sun_elevation_for(pn) for pn in pns}
        times = {
            pn: pd.to_datetime(self.sun_args.loc[pn, "TIME"]).value for pn in pns
        }
        day = [pn for pn in pns if elev[pn] > DAY_MIN_ELEVATION]
        night = [pn for pn in pns if elev[pn] < NIGHT_MAX_ELEVATION]
        if not day or not night:
            return []

        night_times = np.array([times[pn] for pn in night])
        pairs = [
            (d, night[int(np.argmin(np.abs(night_times - times[d])))]) for d in day
        ]
        if len(pairs) > MAX_COMPARE_PROFILES:
            keep = np.linspace(0, len(pairs) - 1, MAX_COMPARE_PROFILES).astype(int)
            pairs = [pairs[i] for i in keep]
            self.log(
                f"Method comparison capped to {MAX_COMPARE_PROFILES} day profiles "
                f"(of {len(day)}) for speed."
            )
        return pairs

    def _profile_subsets(self, profile_numbers):
        """Map ``profile_number -> single-profile subset`` of ``self.data_copy``."""
        pnum = self.data_copy["PROFILE_NUMBER"].values
        subsets = {}
        for pn in profile_numbers:
            idx = np.where(pnum == pn)[0]
            if idx.size:
                subsets[pn] = self.data_copy.isel(N_MEASUREMENTS=idx)
        return subsets

    def _raw_dv(self, profile_numbers, subsets):
        """``{pn: (depth, uncorrected apply_to)}`` for the given profiles."""
        out = {}
        for pn in profile_numbers:
            s = subsets.get(pn)
            if s is not None:
                out[pn] = (s["DEPTH"].values, s[self.apply_to].values)
        return out

    def _run_method_over(self, method_fn, profile_numbers, subsets):
        """``{pn: (depth, corrected fluorescence)}`` from running ``method_fn``."""
        out = {}
        for pn in profile_numbers:
            s = subsets.get(pn)
            if s is None:
                continue
            try:
                corrected = method_fn(s)
            except Exception:  # a single bad profile shouldn't sink the panel
                continue
            out[pn] = (s["DEPTH"].values, np.asarray(corrected, dtype=float))
        return out

    def _score(self, day_dv, night_dv, pairs):
        """Fit statistics for corrected-day vs night fluorescence across pairs.

        Only depth bins within ``COMPARE_SURFACE_LIMIT_METRES`` of the surface
        are scored: quenching (and hence the differences between methods) lives
        in the near-surface layer, so including the deep bins - where every
        method leaves the data unchanged and day already matches night - only
        dilutes the metric.
        """
        # DEPTH is negative-down, so the surface window is bins with key >= this.
        min_key = int(np.floor(-COMPARE_SURFACE_LIMIT_METRES / COMPARE_BIN_METRES))
        xs, ys = [], []
        for day_pn, night_pn in pairs:
            day = day_dv.get(day_pn)
            night = night_dv.get(night_pn)
            if day is None or night is None:
                continue
            day_bins = self._bin_medians(day[0], day[1])
            night_bins = self._bin_medians(night[0], night[1])
            for k in day_bins.keys() & night_bins.keys():
                if k < min_key:  # deeper than the surface window -> skip
                    continue
                xs.append(night_bins[k])
                ys.append(day_bins[k])
        return self._fit_stats(xs, ys)

    @staticmethod
    def _bin_medians(depth, values):
        """Median ``values`` per ``COMPARE_BIN_METRES`` depth bin, keyed by bin."""
        depth = np.asarray(depth, dtype=float)
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(depth) & np.isfinite(values)
        if not np.any(mask):
            return {}
        keys = np.floor(depth[mask] / COMPARE_BIN_METRES).astype(int)
        vals = values[mask]
        return {int(k): float(np.nanmedian(vals[keys == k])) for k in np.unique(keys)}

    @staticmethod
    def _fit_stats(xs, ys):
        """Regression + agreement stats of ``ys`` (day) against ``xs`` (night)."""
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2 or np.ptp(x) == 0:
            return None
        fit = linregress(x, y)
        resid = y - x
        return {
            "x": x,
            "y": y,
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r2": float(fit.rvalue ** 2),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "bias": float(np.mean(resid)),
            "n": int(x.size),
        }

    def _draw_scatter_panel(self, ax, label, stats, is_best):
        if not stats:
            ax.text(
                0.5,
                0.5,
                f"{label}\n(insufficient data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        x, y = stats["x"], stats["y"]
        ax.scatter(x, y, s=7, c="#3b7dd8", alpha=0.35, edgecolors="none")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * ((hi - lo) or 1.0)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, ls="--", c="0.5", lw=1)  # 1:1 line
        line_x = np.array(lims)
        ax.plot(
            line_x, stats["slope"] * line_x + stats["intercept"], c="#d1495b", lw=1.6
        )  # regression fit
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        sign = "+" if stats["intercept"] >= 0 else "-"
        legend = (
            f"y={stats['slope']:.2f}x {sign} {abs(stats['intercept']):.2f}\n"
            f"RMSE={stats['rmse']:.3f}\n"
            f"Bias={stats['bias']:+.3f}\n"
            f"R$^2$={stats['r2']:.2f}  (n={stats['n']})"
        )
        ax.text(
            0.03,
            0.97,
            legend,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )
        ax.set_title(
            label + ("  * best" if is_best else ""),
            fontsize=8,
            fontweight="bold" if is_best else "normal",
        )
        ax.tick_params(labelsize=6)
        ax.set_xlabel("night F", fontsize=6.5)
        ax.set_ylabel("day F (corr)", fontsize=6.5)
        if is_best:
            for spine in ax.spines.values():
                spine.set(color="#2e9e4f", linewidth=2.6)

    # --- Middle column: example profiles ------------------------------

    def _draw_example_profiles(self, fig, subspec):
        gm = subspec.subgridspec(3, 1, hspace=0.55)
        day_pn, night_pn = self._example_profiles()

        # Capture the configured method's decision internals for the day profile
        # so the day plot can mark the quenching depth and the bottom panel can
        # show *how* the correction reference was chosen.
        info = self._explain_profile(day_pn)
        z_ref = info.get("z_ref") if info else None

        self._draw_profile_change(
            fig.add_subplot(gm[0, 0]),
            day_pn,
            f"Example day profile (#{day_pn})",
            ref_depth=z_ref,
        )
        self._draw_profile_change(
            fig.add_subplot(gm[1, 0]), night_pn, f"Example night profile (#{night_pn})"
        )
        self._draw_decision_panel(fig.add_subplot(gm[2, 0]), info)

    def _explain(self, profile, **info):
        """Capture one profile's decision internals during diagnostics.

        A no-op on the normal pipeline run (``_explain_target`` unset). When the
        diagnostics set a target profile number, the configured method records
        the quantities behind its correction (reference depth/value, the layer
        it searched, etc.) so they can be drawn without re-deriving them.
        """
        target = getattr(self, "_explain_target", None)
        if target is not None and int(profile["PROFILE_NUMBER"][0]) == target:
            self._explain_info = info

    def _explain_profile(self, day_pn):
        """Run the configured method over one day profile to capture internals.

        Returns the dict recorded by :meth:`_explain`, or ``None`` if the method
        made no correction on that profile (e.g. missing inputs) or the profile
        is absent.
        """
        subsets = self._profile_subsets({day_pn})
        s = subsets.get(day_pn)
        if s is None:
            return None
        self._explain_info = None
        self._explain_target = day_pn
        try:
            self._method_function(s)
        except Exception:
            self._explain_info = None
        finally:
            self._explain_target = None
        return self._explain_info

    def _example_profiles(self):
        """Pick a representative day and night profile for the configured method.

        The day profile is the daytime profile the configured method changed
        most (or ``plot_profiles[0]`` if given); the night profile is the
        nearest-in-time nighttime profile.
        """
        pnum = self.data["PROFILE_NUMBER"].values
        change = np.abs(
            self.data[self.output_as].values - self.data_copy[self.apply_to].values
        )
        change[~np.isfinite(change)] = 0.0

        elev = {int(pn): self._sun_elevation_for(int(pn)) for pn in self.sun_args.index}
        total_change = {}
        for pn in np.unique(pnum[np.isfinite(pnum)]):
            total_change[int(pn)] = float(np.nansum(change[np.where(pnum == pn)[0]]))

        hint = int(self.plot_profiles[0]) if self.plot_profiles else None
        if hint in total_change and elev.get(hint, 0) > DAY_MIN_ELEVATION:
            day_pn = hint
        else:
            day_candidates = [p for p in total_change if elev.get(p, 0) > DAY_MIN_ELEVATION]
            pool = day_candidates or list(total_change)
            day_pn = max(pool, key=lambda p: total_change[p])

        night_candidates = [p for p in total_change if elev.get(p, 0) < NIGHT_MAX_ELEVATION]
        if night_candidates:
            day_t = pd.to_datetime(self.sun_args.loc[day_pn, "TIME"]).value
            night_pn = min(
                night_candidates,
                key=lambda p: abs(pd.to_datetime(self.sun_args.loc[p, "TIME"]).value - day_t),
            )
        else:
            night_pn = day_pn
        return day_pn, night_pn

    def _draw_profile_change(self, ax, profile_number, title, ref_depth=None):
        idx = np.where(self.data["PROFILE_NUMBER"].values == profile_number)[0]
        depth = self.data["DEPTH"].values[idx]
        orig = self.data_copy[self.apply_to].values[idx]
        corr = self.data[self.output_as].values[idx]

        valid = np.isfinite(depth) & np.isfinite(orig)
        depth, orig, corr = depth[valid], orig[valid], corr[valid]
        changed = np.isfinite(corr) & (np.abs(corr - orig) > 1e-9)

        # Faint connectors show how far each corrected point moved.
        for o, c, d in zip(orig[changed], corr[changed], depth[changed]):
            ax.plot([o, c], [d, d], c="0.85", lw=0.6, zorder=1)
        ax.scatter(
            orig[~changed], depth[~changed], s=16, c="#1f9e89", label="Unchanged", zorder=2
        )
        ax.scatter(
            orig[changed], depth[changed], s=16, c="0.6", label="Original (quenched)", zorder=2
        )
        ax.scatter(
            corr[changed], depth[changed], s=16, c="#d1495b", label="Corrected", zorder=3
        )
        # The method's own quenching depth: the boundary above which it lifted the
        # fluorescence to the chosen reference.
        if ref_depth is not None and np.isfinite(ref_depth):
            ax.axhline(
                ref_depth,
                ls="--",
                lw=1.2,
                c="#ff7f0e",
                zorder=4,
                label=f"Quenching depth ({-ref_depth:.0f} m)",
            )
        ax.set_xlabel(self.apply_to, fontsize=8)
        ax.set_ylabel("DEPTH", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)

    # --- Middle column: how the correction reference was chosen --------

    def _draw_decision_panel(self, ax, info):
        """Show *how* the configured method chose its correction reference.

        Per-profile methods (Xing 2012, Biermann, Xing 2018/Terrats) use the
        example day profile's captured internals; the night-reference methods
        (Hemsley, Thomalla) show the deployment-wide object they actually used.
        """
        method = self.method.lower()
        drawers = {
            "xing2012": self._decision_xing2012,
            "biermann2015": self._decision_biermann,
            "xing2018": self._decision_xing_terrats,
            "terrats2020": self._decision_xing_terrats,
            "hemsley2015": self._decision_hemsley,
            "thomalla2018": self._decision_thomalla,
        }
        drawer = drawers.get(method)
        if drawer is None:
            self._decision_unavailable(ax, "No decision diagnostic for this method.")
            return
        # Hemsley reads its global regression, not the per-profile info.
        if method != "hemsley2015" and not info:
            self._decision_unavailable(
                ax, "The method made no correction on the\nexample profile."
            )
            return
        drawer(ax, info)

    @staticmethod
    def _decision_unavailable(ax, message):
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8)

    def _decision_xing2012(self, ax, info):
        depth, chlf = info["depth"], info["chlf"]
        mld, z_ref, f_ref = info["mld"], info["z_ref"], info["f_ref"]
        valid = np.isfinite(depth) & np.isfinite(chlf)
        within = valid & (depth >= mld)
        ax.scatter(chlf[valid], depth[valid], s=12, c="0.7", label="F (all)")
        ax.scatter(chlf[within], depth[within], s=14, c="#3b7dd8", label="within MLD")
        ax.axhline(mld, ls=":", c="black", lw=1.1, label=f"MLD ({-mld:.0f} m)")
        ax.scatter(
            [f_ref], [z_ref], marker="*", s=200, c="#d1495b", zorder=5,
            label="reference Fmax",
        )
        ax.set_title("Xing 2012: reference = max F within MLD", fontsize=9)
        self._decision_axes(ax, self.apply_to)

    def _decision_biermann(self, ax, info):
        depth, chlf = info["depth"], info["chlf"]
        zeu, z_ref, f_ref = info["zeu"], info["z_ref"], info["f_ref"]
        valid = np.isfinite(depth) & np.isfinite(chlf)
        below = valid & (depth <= zeu)
        ax.scatter(chlf[valid], depth[valid], s=12, c="0.7", label="F (all)")
        ax.scatter(chlf[below], depth[below], s=14, c="#3b7dd8", label="below Zeu")
        ax.axhline(zeu, ls=":", c="#2ca02c", lw=1.1, label=f"Zeu ({-zeu:.0f} m)")
        ax.scatter(
            [f_ref], [z_ref], marker="*", s=200, c="#d1495b", zorder=5,
            label="max F below Zeu",
        )
        ax.set_title("Biermann 2015: reference = max F below Zeu", fontsize=9)
        self._decision_axes(ax, self.apply_to)

    def _decision_xing_terrats(self, ax, info):
        label = self._METHOD_LABELS.get(self.method.lower(), self.method)
        mld, ipar15 = info["mld"], info["ipar15"]
        if info.get("branch") == "shallow":
            # Terrats shallow-mixing branch: plot the sigmoid scaling applied
            # below the MLD (this is the de-quenching "decision" here).
            z, s = info["sigmoid_depth"], info["sigmoid_scale"]
            order = np.argsort(-z)
            ax.plot(s[order], z[order], c="#3b7dd8", lw=1.4, label="sigmoid scale s(z)")
            ax.axhline(mld, ls=":", c="black", lw=1.1, label=f"MLD ({-mld:.0f} m)")
            ax.set_title(f"{label} (shallow mixing): F/s below MLD", fontsize=9)
            self._decision_axes(ax, "de-quench scale s(z)")
            return
        ratio, z_ref, depth = info["ratio"], info["z_ref"], info["depth"]
        valid = np.isfinite(depth) & np.isfinite(ratio)
        npq = valid & (depth >= z_ref)
        ax.scatter(ratio[valid], depth[valid], s=12, c="0.7", label="F/bbp (all)")
        ax.scatter(ratio[npq], depth[npq], s=14, c="#3b7dd8", label="NPQ layer")
        ax.axhline(mld, ls=":", c="black", lw=1.1, label=f"MLD ({-mld:.0f} m)")
        if np.isfinite(ipar15):
            ax.axhline(
                ipar15, ls="--", c="#9467bd", lw=1.1, label=f"iPAR=15 ({-ipar15:.0f} m)"
            )
        ax.scatter(
            [info["r_max"]], [info["rmax_depth"]], marker="*", s=200, c="#d1495b",
            zorder=5, label="R_max",
        )
        ax.set_title(f"{label} (deep mixing): reference = max F/bbp", fontsize=9)
        self._decision_axes(ax, "F/bbp ratio")

    def _decision_hemsley(self, ax, info):
        reg = getattr(self, "_hemsley_regression", None)
        if not reg or "bbp" not in reg:
            self._decision_unavailable(ax, "No global night regression was fit.")
            return
        b, f = reg["bbp"], reg["fl"]
        m, c = reg["slope"], reg["intercept"]
        ax.scatter(b, f, s=8, c="#3b7dd8", alpha=0.4, edgecolors="none", label="night points")
        xs = np.array([np.nanmin(b), np.nanmax(b)])
        sign = "+" if c >= 0 else "-"
        ax.plot(
            xs, m * xs + c, c="#d1495b", lw=1.6,
            label=f"F = {m:.3g}*bbp {sign} {abs(c):.3g}",
        )
        ax.text(
            0.03, 0.97, f"R$^2$={reg['r2']:.2f}  (n={reg['n']})",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )
        ax.set_title("Hemsley 2015: global night fl–bbp regression", fontsize=9)
        ax.set_xlabel(self.bbp_var, fontsize=8)
        ax.set_ylabel(self.apply_to, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)

    def _decision_thomalla(self, ax, info):
        refs = getattr(self, "_night_refs", None)
        ref_idx = info.get("ref_idx")
        if not refs or ref_idx is None or ref_idx >= len(refs):
            self._decision_unavailable(ax, "No night reference for this profile.")
            return
        ref = refs[ref_idx]
        z_ref, zeu = info["z_ref"], info.get("zeu")
        # ref["z"] is positive-down; the DEPTH axis is negative-down.
        ax.plot(ref["ratio"], -ref["z"], c="#3b7dd8", lw=1.4, label="night fl:bbp ratio")
        if zeu is not None and np.isfinite(zeu):
            ax.axhline(zeu, ls=":", c="#2ca02c", lw=1.1, label=f"Zeu ({-zeu:.0f} m)")
        if np.isfinite(z_ref):
            ax.axhline(
                z_ref, ls="--", c="#ff7f0e", lw=1.2, label=f"QD ({-z_ref:.0f} m)"
            )
        ax.set_title("Thomalla 2018: night fl:bbp ratio & QD", fontsize=9)
        self._decision_axes(ax, "night fl:bbp ratio")

    @staticmethod
    def _decision_axes(ax, xlabel):
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("DEPTH", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)

    # --- Right column: depth-time sections ----------------------------

    def _draw_timeseries(self, fig, subspec):
        gr = subspec.subgridspec(3, 1, hspace=0.35)
        time = self.data["TIME"].values
        depth = self.data["DEPTH"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values

        # Keep points with a valid time/depth in the top TIMESERIES_DEPTH_LIMIT m.
        finite = (
            ~pd.isnull(time)
            & np.isfinite(depth)
            & (depth >= -TIMESERIES_DEPTH_LIMIT)
        )
        time, depth = time[finite], depth[finite]
        orig, corr = orig[finite], corr[finite]

        changed = np.isfinite(corr) & np.isfinite(orig) & (np.abs(corr - orig) > 1e-9)
        vmin, vmax = self._robust_vlim(orig, corr)

        ax1 = fig.add_subplot(gr[0, 0])
        sc1 = ax1.scatter(
            time, depth, c=orig, cmap="viridis", vmin=vmin, vmax=vmax,
            s=SECTION_MARKER_SIZE, rasterized=True,
        )
        ax1.set_title(f"Original fluorescence (top {TIMESERIES_DEPTH_LIMIT:.0f} m)", fontsize=9)

        ax2 = fig.add_subplot(gr[1, 0], sharex=ax1, sharey=ax1)
        sc2 = ax2.scatter(
            time, depth, c=corr, cmap="viridis", vmin=vmin, vmax=vmax,
            s=SECTION_MARKER_SIZE, rasterized=True,
        )
        ax2.set_title(f"Quenching-corrected fluorescence (top {TIMESERIES_DEPTH_LIMIT:.0f} m)", fontsize=9)

        ax3 = fig.add_subplot(gr[2, 0], sharex=ax1, sharey=ax1)
        ax3.scatter(
            time[~changed], depth[~changed], c="#d9d9d9",
            s=SECTION_MARKER_SIZE, rasterized=True, label="Unchanged",
        )
        ax3.scatter(
            time[changed], depth[changed], c="#d1495b",
            s=SECTION_MARKER_SIZE, rasterized=True, label="Corrected",
        )
        ax3.set_title("Quenching layer (corrected points)", fontsize=9)
        ax3.legend(fontsize=6.5, loc="lower right", framealpha=0.9, markerscale=8)

        # Overlay per-profile MLD, euphotic-depth and quenching-depth lines.
        line_t, line_mld, line_qd, line_zeu = self._section_reference_lines()
        for ax in (ax1, ax2, ax3):
            self._overlay_reference_lines(
                ax, line_t, line_mld, line_qd, line_zeu, legend=(ax is ax1)
            )
        ax1.set_ylim(-TIMESERIES_DEPTH_LIMIT, 0)

        for sc, ax in ((sc1, ax1), (sc2, ax2)):
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(self.apply_to, fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        for ax in (ax1, ax2):
            plt.setp(ax.get_xticklabels(), visible=False)
        for ax in (ax1, ax2, ax3):
            ax.set_ylabel("DEPTH", fontsize=8)
            ax.tick_params(labelsize=7)
        ax3.set_xlabel("TIME", fontsize=8)
        plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    def _section_reference_lines(self):
        """Per-profile MLD, euphotic depth and quenching depth (negative-down m).

        The quenching depth is the deepest point the correction actually changed
        in each profile - a method-agnostic marker matching the red 'Corrected'
        points. MLD is read from the ``MLD`` variable when present; the euphotic
        depth (Zeu, 1% light) is derived from the PAR profile when ``par_var`` is
        present. All are ``NaN`` where undefined (e.g. night, or a missing
        input), so the drawn lines break over those profiles. Sorted by time.
        """
        pnum = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values
        changed = np.isfinite(corr) & np.isfinite(orig) & (np.abs(corr - orig) > 1e-9)
        mld_all = self.data["MLD"].values if "MLD" in self.data.data_vars else None
        par_all = (
            self.data[self.par_var].values
            if self.par_var in self.data.data_vars
            else None
        )

        times, mld, qd, zeu = [], [], [], []
        for pn in self.sun_args.index:
            idx = np.where(pnum == pn)[0]
            if idx.size == 0:
                continue
            times.append(pd.to_datetime(self.sun_args.loc[pn, "TIME"]).to_datetime64())
            in_profile = changed[idx]
            qd.append(float(np.min(depth[idx][in_profile])) if np.any(in_profile) else np.nan)
            if mld_all is not None:
                finite_mld = mld_all[idx][np.isfinite(mld_all[idx])]
                mld.append(float(finite_mld[0]) if finite_mld.size else np.nan)
            else:
                mld.append(np.nan)
            if par_all is not None:
                # estimate_euphotic_depth wants positive-down depth; store as
                # negative-down to match the DEPTH axis.
                zeu_val = estimate_euphotic_depth(par_all[idx], -depth[idx])
                zeu.append(-zeu_val if np.isfinite(zeu_val) else np.nan)
            else:
                zeu.append(np.nan)

        times = np.asarray(times, dtype="datetime64[ns]")
        order = np.argsort(times)
        return (
            times[order],
            np.asarray(mld)[order],
            np.asarray(qd)[order],
            np.asarray(zeu)[order],
        )

    @staticmethod
    def _overlay_reference_lines(ax, times, mld, qd, zeu, legend=False):
        """Draw the MLD, euphotic-depth and quenching-depth lines on a panel.

        NaNs are dropped per line so valid points connect across the whole
        record: quenching depth and Zeu are only defined on daytime profiles, so
        plotting them with the NaN night gaps in place would leave the isolated
        valid points invisible.
        """
        drawn = False
        for values, colour, label in (
            (mld, "black", "MLD"),
            (zeu, "#2ca02c", "Euphotic depth (Zeu)"),
            (qd, "#ff7f0e", "Quenching depth"),
        ):
            valid = np.isfinite(values)
            if np.any(valid):
                ax.plot(times[valid], values[valid], c=colour, lw=1.1, label=label)
                drawn = True
        if legend and drawn:
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

    @staticmethod
    def _robust_vlim(*arrays):
        """2nd-98th percentile colour limits across the given arrays."""
        stacked = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size == 0:
            return None, None
        return float(np.percentile(stacked, 2)), float(np.percentile(stacked, 98))

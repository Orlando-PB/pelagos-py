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
#   [x] swart2015     (Swart et al. 2015)
#   [x] sackmann2008  (Sackmann et al. 2008)

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin
import pelagos_py.utils.diagnostics as diag
import pelagos_py.utils.palettes as palettes

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
MIDDAY_MIDNIGHT_WINDOW_HOURS = 1.5  #: solar-time half-window (h) around solar noon/midnight; only profiles this close are used in the Fig-4-style midday-vs-midnight regression. Widen if too few pairs result.
TIMESERIES_DEPTH_LIMIT = 300.0  #: max depth (m) shown in the timeseries; the window is dynamic but never deeper than this.
TIMESERIES_DEPTH_MIN = 50.0  #: min depth (m) the dynamic timeseries window is allowed to shrink to.
SECTION_MARKER_SIZE = 1.5  #: scatter marker size (points^2) for the section plots.
SECTION_MAX_POINTS = 1_000_000  #: DEBUG: cap on points drawn in the bottom "by correction status" panel; large values are slow but show the full cloud.

#: Colour/label per point category in the bottom section debug panel, in draw
#: order (later entries plot on top, so 'corrected' sits over 'uncorrected').
#: 'corrected' = value actually changed; 'uncorrected' = everything else.
SECTION_CATEGORY_STYLE = [
    ("uncorrected", "Uncorrected", "#3b6fb5"),
    ("corrected", "Corrected", "#e6cf8b"),
]

#: Night-reference tuning for the 'hemsley2015'/'thomalla2018' methods.
NIGHT_REF_BIN_METRES = 1.0  #: depth bin (m) for averaging nighttime profiles into a reference.
HEMSLEY_REGRESSION_DEPTH = 60.0  #: top-of-water depth (m) over which the Hemsley regression is fit.

#: A quenching correction should not lift CHLA far above the profile's own
#: maximum observed fluorescence. The backscatter-ratio methods
#: ('xing2018'/'terrats2020'/'thomalla2018'/'sackmann2008'/'swart2015') can,
#: when a near-zero (noise-level)
#: backscatter inflates the CHLA/bbp ratio. Warn once if a corrected value
#: exceeds this multiple of the profile's max input - judged on the output, not
#: the ratio, so a large-but-harmless ratio doesn't cry wolf. Usually means the
#: backscatter needs cleaning first.
CORRECTION_WARN_FACTOR = 5.0

#: Backscatter variables tried, in order, when the configured 'bbp_var' is
#: absent from the data. The despiked baseline is preferred; raw BBP700/BBP532
#: are fallbacks. If none are present the step halts.
BBP_VAR_FALLBACKS = ["BBP700_BASELINE", "BBP700", "BBP532_BASELINE", "BBP532"]


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
        "swart2015",
        "sackmann2008",
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
                "'hemsley2015' (global night fluorescence-bbp regression, needs BBP + PAR), "
                "'thomalla2018' (per-night fl:bbp ratio profile, needs BBP + PAR), "
                "'sackmann2008' (max fl:bbp ratio within the MLD, needs BBP + MLD) and "
                "'swart2015' (max fl:bbp ratio within the euphotic zone, needs BBP + PAR)."
            ),
        },
        "apply_to": {
            "type": str,
            "default": "CHLA",
            "description": "Name of the variable to apply the correction to.",
        },
        "bbp_var": {
            "type": str,
            "default": "BBP700_BASELINE",
            "description": (
                "Backscatter variable used by 'xing2018'/'terrats2020'/'thomalla2018'/"
                "'hemsley2015'/'sackmann2008'/'swart2015'. Defaults to the despiked "
                "'BBP700_BASELINE'; if that is "
                "absent the step falls back through "
                f"{BBP_VAR_FALLBACKS}, and halts if none are present."
            ),
        },
        "par_var": {
            "type": str,
            "default": "DOWNWELLING_PAR",
            "description": (
                "Downwelling PAR variable used to derive the euphotic depth "
                "('biermann2015'/'swart2015') and the iPAR=15 depth "
                "('xing2018'/'terrats2020')."
            ),
        },
        "interpolate_par": {
            "type": bool,
            "default": True,
            "description": (
                "For PAR-based methods, fill casts that lack usable PAR (e.g. a "
                "descent where the radiometer was not logged) by time-interpolating "
                "the PAR-vs-depth profile from the nearest valid cast on each side, "
                "so every cast can be corrected rather than only those carrying PAR. "
                "The observed PAR variable in the output is left unchanged."
            ),
        },
        "interpolate_par_max_gap_hours": {
            "type": float,
            "default": 12.0,
            "description": (
                "Only used when 'interpolate_par' is true: a cast is filled only if "
                "a valid-PAR cast exists within this many hours on at least one side."
            ),
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
        needs_mld = method_key in ("xing2012", "xing2018", "terrats2020", "sackmann2008")
        needs_par = method_key in (
            "biermann2015", "xing2018", "terrats2020", "hemsley2015", "thomalla2018",
            "swart2015",
        )
        needs_bbp = method_key in (
            "xing2018", "terrats2020", "hemsley2015", "thomalla2018",
            "sackmann2008", "swart2015",
        )
        missing = []
        if needs_mld and "MLD" not in self.data.data_vars:
            missing.append("MLD (add a Mixed Layer Depth step beforehand)")
        if needs_par and self.par_var not in self.data.data_vars:
            missing.append(f"{self.par_var} (PAR; set 'par_var' or add a step providing it)")
        if needs_bbp:
            resolved_bbp = self._resolve_bbp_var()
            if resolved_bbp is None:
                missing.append(
                    f"backscatter variable not found (looked for '{self.bbp_var}' and "
                    f"fallbacks {BBP_VAR_FALLBACKS}; set 'bbp_var' or add a BBP step "
                    "beforehand, e.g. 'BBP from Beta' + 'Isolate BBP Spikes')"
                )
            else:
                self.bbp_var = resolved_bbp
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

        # PAR is often logged on only one cast direction, so the other cast
        # yields no euphotic depth and would be skipped. Fill those casts from
        # neighbouring casts (into the working copies used by the loop and the
        # diagnostics; the exported PAR variable is left untouched).
        if needs_par and self.interpolate_par:
            par_dims = self.data[self.par_var].dims
            filled_par = self._fill_par_across_casts(
                np.asarray(data_subset[self.par_var].values, dtype=float),
                np.asarray(data_subset["DEPTH"].values, dtype=float),
                data_subset["PROFILE_NUMBER"].values,
            )
            data_subset = data_subset.assign(**{self.par_var: (par_dims, filled_par)})
            self.data_copy[self.par_var] = (par_dims, filled_par)

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

        # TEMP DEBUG: report why the biermann correction skipped/altered profiles,
        # and how Zeu (its window) compares to MLD (xing's window) on the same
        # profiles. Remove this block once the biermann behaviour is understood.
        dbg = getattr(self, "_biermann_dbg", None)
        if dbg is not None:
            def _stats(a):
                a = np.asarray(a, dtype=float)
                a = a[np.isfinite(a)]
                if not a.size:
                    return "none"
                return (
                    f"n={a.size} min={a.min():.1f} median={np.median(a):.1f} "
                    f"max={a.max():.1f} m"
                )

            mld_per_profile = []
            if "MLD" in self.data.data_vars:
                pnum_all = self.data["PROFILE_NUMBER"].values
                mld_all = np.asarray(self.data["MLD"].values, dtype=float)
                for pn in self.sun_args.index:
                    m = mld_all[pnum_all == pn]
                    m = m[np.isfinite(m)]
                    if m.size:
                        mld_per_profile.append(float(m[0]))

            self.log(
                "TEMP biermann tally: "
                f"total={dbg['total']} night={dbg['night']} "
                f"bad_input={dbg['bad_input']} zeu_nan={dbg['zeu_nan']} "
                f"empty_window={dbg['empty_window']} corrected={dbg['corrected']} "
                f"no_change={dbg['no_change']}"
            )
            self.log(f"TEMP biermann Zeu (valid profiles): {_stats(dbg['zeu'])}")
            self.log(f"TEMP biermann z_qd (reference depth): {_stats(dbg['z_qd'])}")
            self.log(f"TEMP biermann MLD (same profiles):   {_stats(mld_per_profile)}")
            self.log(
                "TEMP biermann points changed total = "
                f"{int(np.nansum(dbg['n_changed']))}"
            )
        # END TEMP DEBUG

        self.reconstruct_data()
        self.update_qc()

        # Generate new QC if a non-adjusted variable was used in processing (this causes an _ADJUSTED variable to be added)"
        if self.apply_to != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.apply_to}_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def _fill_par_across_casts(self, par, depth, pnum):
        """Fill casts lacking usable PAR from their nearest neighbours in time.

        A "donor" is any cast with at least four finite, positive PAR points
        (enough for a Kd fit). For each daytime cast without one, the PAR-vs-depth
        profile of the nearest donor on each side (within
        ``interpolate_par_max_gap_hours``) is evaluated at the cast's own depths
        and blended by time; a single-sided cast copies its one donor. No
        extrapolation beyond a donor's sampled depth range is made, and night
        casts are left alone (every method skips them regardless).

        Returns a copy of ``par`` with those casts filled; all other samples,
        and the observed PAR variable itself, are unchanged.
        """
        MIN_PTS = 4  # matches estimate_euphotic_depth's minimum for a Kd fit
        filled = np.array(par, dtype=float, copy=True)

        # Per-cast surface-fix time (ns) gathered in run(); orders casts in time.
        times = {
            int(pn): pd.to_datetime(self.sun_args.loc[pn, "TIME"]).value
            for pn in self.sun_args.index
        }

        # Donors: casts with enough finite positive PAR to define a light profile.
        donors = {}
        for pn in times:
            sel = pnum == pn
            z, p = depth[sel], par[sel]
            m = np.isfinite(z) & np.isfinite(p) & (p > 0)
            if np.count_nonzero(m) >= MIN_PTS:
                order = np.argsort(z[m])
                donors[pn] = (z[m][order], p[m][order])
        if not donors:
            self.log("Interpolate PAR: no cast has usable PAR to interpolate from.")
            return filled

        donor_pns = sorted(donors, key=lambda q: times[q])
        donor_t = np.array([times[q] for q in donor_pns])
        max_gap = self.interpolate_par_max_gap_hours * 3600e9  # hours -> ns

        def donor_par_at(pn, z_target):
            dz, dp = donors[pn]
            out = np.interp(z_target, dz, dp)
            out[(z_target < dz[0]) | (z_target > dz[-1])] = np.nan  # no extrapolation
            return out

        n_filled = 0
        for pn in times:
            if pn in donors or self._sun_elevation_for(pn) <= 0:
                continue  # has its own PAR, or is night (skipped by every method)
            t = times[pn]
            left = donor_t <= t
            b = donor_pns[np.where(left)[0][-1]] if left.any() else None
            a = donor_pns[np.where(~left)[0][0]] if (~left).any() else None
            if b is not None and t - times[b] > max_gap:
                b = None
            if a is not None and times[a] - t > max_gap:
                a = None
            if b is None and a is None:
                continue

            sel = pnum == pn
            zt = depth[sel]
            if b is not None and a is not None:
                pb, pa = donor_par_at(b, zt), donor_par_at(a, zt)
                w = (t - times[b]) / (times[a] - times[b])
                pv = (1 - w) * pb + w * pa
                # Where the blend is NaN (one donor out of depth range) but the
                # other donor covers it, keep the covering donor's value.
                pv = np.where(np.isnan(pv) & np.isfinite(pb), pb, pv)
                pv = np.where(np.isnan(pv) & np.isfinite(pa), pa, pv)
            else:
                pv = donor_par_at(b if b is not None else a, zt)
            filled[np.where(sel)[0]] = pv
            n_filled += 1

        self.log(
            f"Interpolate PAR: filled {n_filled} daytime cast(s) lacking usable PAR "
            f"from neighbouring casts (<= {self.interpolate_par_max_gap_hours:g} h)."
        )
        return filled

    def _resolve_bbp_var(self):
        """Return an available backscatter variable, preferring the configured one.

        The configured ``bbp_var`` is used if present; otherwise the
        ``BBP_VAR_FALLBACKS`` are tried in order (despiked baseline first, then
        raw), logging which one is used. Returns ``None`` if none are present so
        the caller can halt with a clear "not found" message.
        """
        candidates = [self.bbp_var] + [b for b in BBP_VAR_FALLBACKS if b != self.bbp_var]
        for name in candidates:
            if name in self.data.data_vars:
                if name != self.bbp_var:
                    self.log(
                        f"Backscatter variable '{self.bbp_var}' not found; "
                        f"using fallback '{name}'."
                    )
                return name
        return None

    def _warn_if_correction_blows_up(self, chlf, chl_corr):
        """Warn once if a correction lifted CHLA implausibly far above the input.

        The backscatter-ratio methods reset fluorescence to ``bbp * ratio``; a
        near-zero backscatter inflates the ratio and produces a corrected value
        far above anything observed. Judged on the corrected *output* (not the
        ratio), so a large-but-harmless ratio doesn't cry wolf. Suppressed during
        diagnostics so only the configured method's real output can warn. Fires at
        most once per step run.
        """
        if getattr(self, "_blowup_warned", False) or getattr(self, "_suppress_warn", False):
            return
        chlf = np.asarray(chlf, dtype=float)
        chl_corr = np.asarray(chl_corr, dtype=float)
        ref = np.nanmax(chlf) if np.any(np.isfinite(chlf)) else np.nan
        peak = np.nanmax(chl_corr) if np.any(np.isfinite(chl_corr)) else np.nan
        if not (np.isfinite(ref) and np.isfinite(peak)) or ref <= 0:
            return
        if peak <= CORRECTION_WARN_FACTOR * ref:
            return
        self._blowup_warned = True
        self.log_warn(
            f"This correction lifted {self.apply_to} to {peak:.1f} - {peak / ref:.0f}x "
            f"the profile's own maximum of {ref:.1f}. That usually means a near-zero "
            f"{self.bbp_var} inflated the {self.apply_to}/bbp ratio and blew up the "
            f"correction. Check {self.bbp_var} is cleaned (despiked / dark-offset "
            f"corrected) before this step - e.g. run an 'Isolate BBP Spikes' step and "
            f"set 'bbp_var' to its baseline."
        )

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
            or mld <= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        # --- Identify max F_Chl within the mixed layer (surface to MLD).
        within_mld = depth <= mld
        if not np.any(within_mld):
            return chlf

        chlf_mld = np.where(within_mld, chlf, np.nan)
        idx_max, chlf_max = np.nanargmax(chlf_mld), np.nanmax(chlf_mld)
        chlf_max_depth = float(depth[idx_max])

        # --- Apply correction: flatten everything shallower than the reference.
        chl_corr = np.copy(chlf)
        chl_corr[(depth <= chlf_max_depth) & (~np.isnan(chlf))] = chlf_max

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

    def _hours_from_solar_noon(self, profile_number):
        """Hours between a profile's surface fix and its nearest solar noon.

        Returns a value in [0, 12]: 0 at solar noon (peak sun, maximum
        quenching), 12 at solar midnight (no quenching). Uses the equation of
        time and longitude to convert the UTC fix to local apparent solar time,
        so "midday"/"midnight" track the sun rather than the clock regardless of
        longitude or season. Memoised per profile like the elevation lookup.
        """
        cache = getattr(self, "_solar_noon_cache", None)
        if cache is None:
            cache = self._solar_noon_cache = {}
        if profile_number not in cache:
            time, lat, long = self.sun_args.loc[profile_number].to_numpy()
            time_utc = pd.to_datetime(time).tz_localize("UTC")
            solpos = pvlib.solarposition.get_solarposition(time_utc, lat, long)
            eot = float(solpos["equation_of_time"].values[0])  # minutes
            utc_hours = time_utc.hour + time_utc.minute / 60 + time_utc.second / 3600
            # Local apparent solar time (hours): UTC + longitude offset + EoT.
            solar_hours = (utc_hours + long / 15.0 + eot / 60.0) % 24.0
            cache[profile_number] = abs(solar_hours - 12.0)
        return cache[profile_number]

    def apply_biermann2015_quenching_correction(self, profile):
        """
        Apply non-photochemical quenching (NPQ) correction following
        Biermann et al. (2015, *Ocean Science*, 11:83-91).

        The maximum fluorescence within the euphotic zone (surface to Zeu) is
        taken as the non-quenched reference; all shallower values are lifted to
        it. Zeu is derived per profile from the PAR profile (1% light level).
        Correction is applied only in daytime (solar elevation > 0).
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        par = np.asarray(profile[self.par_var].values, dtype=float)
        N = len(chlf)

        sun_angle = self._sun_elevation(profile)
        zeu = estimate_euphotic_depth(par, depth)

        # TEMP DEBUG: tally why each profile is/isn't corrected by biermann so
        # the run summary (in run()) can show what drives the point count.
        dbg = getattr(self, "_biermann_dbg", None)
        if dbg is None:
            dbg = self._biermann_dbg = {
                "total": 0, "night": 0, "bad_input": 0, "zeu_nan": 0,
                "empty_window": 0, "no_change": 0, "corrected": 0,
                "zeu": [], "z_qd": [], "n_changed": [],
            }
        dbg["total"] += 1
        if sun_angle <= 0:
            dbg["night"] += 1
        elif N == 0 or len(depth) != N or np.all(np.isnan(chlf)):
            dbg["bad_input"] += 1
        elif not np.isfinite(zeu) or zeu <= 0:
            dbg["zeu_nan"] += 1
        else:
            dbg["zeu"].append(float(zeu))
        # END TEMP DEBUG

        if (
            sun_angle <= 0
            or N == 0
            or len(depth) != N
            or not np.isfinite(zeu)
            or zeu <= 0
            or np.all(np.isnan(chlf))
        ):
            return chlf

        # Reference = max F_Chl within the euphotic zone (surface to Zeu).
        within_zeu = depth <= zeu
        chlf_within = np.where(within_zeu, chlf, np.nan)
        if np.all(np.isnan(chlf_within)):
            dbg["empty_window"] += 1  # TEMP DEBUG
            return chlf

        idx_max = np.nanargmax(chlf_within)
        z_qd = depth[idx_max]
        f_max = chlf[idx_max]

        # Lift everything shallower than the quenching depth to the reference.
        chl_corr = np.copy(chlf)
        chl_corr[(depth <= z_qd) & (~np.isnan(chlf))] = f_max

        # TEMP DEBUG: record the correction extent (depth of the reference max
        # and how many points actually moved) for the run summary.
        n_changed = int(np.sum(np.abs(chl_corr - chlf) > 1e-9))
        dbg["z_qd"].append(float(z_qd))
        dbg["n_changed"].append(n_changed)
        dbg["corrected" if n_changed else "no_change"] += 1
        # END TEMP DEBUG

        self._explain(
            profile,
            depth=depth,
            chlf=chlf,
            zeu=zeu,
            z_ref=z_qd,
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
        maximum).
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        par = np.asarray(profile[self.par_var].values, dtype=float)
        N = len(chlf)

        regression = getattr(self, "_hemsley_regression", None)
        sun_angle = self._sun_elevation(profile)
        zeu = estimate_euphotic_depth(par, depth)
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
        fill = (depth >= 0) & (depth <= zeu) & np.isfinite(bbp) & (~np.isnan(chlf))
        chl_corr[fill] = m * bbp[fill] + c

        self._explain(profile, depth=depth, chlf=chlf, zeu=zeu, z_ref=zeu)
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
        z_all = np.asarray(self.data["DEPTH"].values, dtype=float)
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

        On any condition that prevents a correction (night, missing inputs,
        degenerate profile) the uncorrected fluorescence is returned unchanged.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        ipar = np.asarray(profile[self.par_var].values, dtype=float)
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

        # MLD for this profile (one value, broadcast across its measurements).
        finite_mld = np.asarray(profile["MLD"].values, dtype=float)
        finite_mld = finite_mld[np.isfinite(finite_mld)]
        mld = float(finite_mld[0]) if finite_mld.size else np.nan

        # Depth at which iPAR crosses 15 umol m-2 s-1, on the irregular grid.
        zi_par15 = self._depth_of_ipar15(depth, ipar)

        # Shallow mixing: light penetrates below the mixed layer.
        shallow = hybrid and np.isfinite(zi_par15) and np.isfinite(mld) and zi_par15 > mld

        if not shallow:
            # --- Deep-mixing S08+ (Xing 2018) --------------------------------
            if not (np.isfinite(mld) and np.isfinite(zi_par15)):
                return chlf
            # NPQ layer: shallower than the shallower of MLD and the iPAR=15 depth.
            z_ref = min(mld, zi_par15)
            npq_layer = (depth <= z_ref) & np.isfinite(depth)
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
                mld=mld,
                ipar15=zi_par15,
                z_ref=z_ref,
                r_max=r_max,
                rmax_depth=float(depth[idx_rmax]),
            )
        else:
            # --- Shallow-mixing X18_S08 hybrid (Terrats 2020) ----------------
            if not np.isfinite(mld):
                return chlf
            r, ipar_mid, e = 0.092, 261.0, 2.2  # XB18 sigmoid parameters.

            chl_corr = np.copy(chlf)
            below = (depth > mld) & np.isfinite(depth)
            # Sigmoid de-quenching below the MLD; clip PAR away from zero first.
            ipar_safe = np.clip(ipar[below], 1e-3, None)
            s = r + (1 - r) / (1 + (ipar_safe / ipar_mid) ** e)
            s = np.clip(s, r, 1.0)
            chl_corr[below] = chlf[below] / s

            # Ratio at the shallowest valid point just below the MLD.
            below_idx = np.where(below)[0]
            order = below_idx[np.argsort(depth[below_idx])]
            r_mld = np.nan
            for k in order:
                if np.isfinite(chl_corr[k]) and np.isfinite(bbp[k]) and bbp[k] > 0:
                    r_mld = chl_corr[k] / bbp[k]
                    break
            if not np.isfinite(r_mld):
                return chlf

            above = (depth <= mld) & np.isfinite(bbp) & (~np.isnan(chlf))
            chl_corr[above] = bbp[above] * r_mld

            self._explain(
                profile,
                branch="shallow",
                depth=depth,
                mld=mld,
                ipar15=zi_par15,
                z_ref=mld,
                r_mld=r_mld,
                sigmoid_depth=depth[below],
                sigmoid_scale=s,
            )

        # Never let the correction reduce fluorescence (fmax ignores NaNs).
        result = np.fmax(chlf, chl_corr)
        self._warn_if_correction_blows_up(chlf, result)
        return result

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
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        par = np.asarray(profile[self.par_var].values, dtype=float)
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
        zeu = estimate_euphotic_depth(par, depth)
        if not np.isfinite(zeu) or zeu <= 0:
            return chlf

        # Night fl:bbp ratio and mean fluorescence interpolated onto day depths.
        ratio_at_z = np.interp(depth, ref["z"], ref["ratio"])
        fl_night_at_z = np.interp(depth, ref["z"], ref["fl"])

        qd = self._quenching_depth(depth, chlf, fl_night_at_z, zeu)
        if not np.isfinite(qd):
            return chlf

        corrected = ratio_at_z * bbp
        chl_corr = np.copy(chlf)
        # Correct from the surface to the quenching depth, keeping the result
        # only where it raises the (quenched) daytime fluorescence.
        fill = (
            (depth >= 0)
            & (depth <= qd)
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
            zeu=zeu,
            z_ref=qd,
            ref_idx=ref_idx,
        )
        self._warn_if_correction_blows_up(chlf, chl_corr)
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

    def apply_sackmann2008_quenching_correction(self, profile):
        """
        Apply the Sackmann et al. (2008, *Biogeosciences*, 5:2839) NPQ
        correction.

        Within the mixed layer the maximum fluorescence-to-backscatter ratio
        ``R_max = max(F_Chl / b_bp)`` is taken as the non-quenched reference (the
        night-time fl:bbp ratio is assumed uniform there). Fluorescence from the
        surface to the depth of ``R_max`` is reset to ``b_bp x R_max``. See
        :meth:`_apply_max_ratio_correction`.
        """
        return self._apply_max_ratio_correction(profile, window="mld")

    def apply_swart2015_quenching_correction(self, profile):
        """
        Apply the Swart et al. (2015, *J. Plankton Res.*, 37:635) NPQ
        correction.

        Same scheme as Sackmann et al. (2008) but the maximum
        fluorescence-to-backscatter ratio is sought within the euphotic zone
        (surface to Zeu, the 1% light level from the PAR profile) rather than the
        mixed layer. See :meth:`_apply_max_ratio_correction`.
        """
        return self._apply_max_ratio_correction(profile, window="zeu")

    def _apply_max_ratio_correction(self, profile, window):
        """Max fl:bbp-ratio NPQ correction shared by 'sackmann2008'/'swart2015'.

        Finds the largest (least-quenched) ``F_Chl / b_bp`` ratio within a search
        window and resets fluorescence to ``b_bp x R_max`` from the surface down
        to the depth of that maximum ratio (Table 1 of Thomalla et al. 2018). The
        window is the mixed layer (``window='mld'``, Sackmann 2008) or the
        euphotic zone (``window='zeu'``, Swart 2015).

        On any condition that prevents a correction (night, missing inputs,
        degenerate profile) the uncorrected fluorescence is returned unchanged.
        The correction is clamped so it never lowers fluorescence.
        """
        chlf = np.asarray(profile[self.apply_to].values, dtype=float)
        depth = np.asarray(profile["DEPTH"].values, dtype=float)
        bbp = np.asarray(profile[self.bbp_var].values, dtype=float)
        N = len(chlf)

        sun_angle = self._sun_elevation(profile)
        if (
            sun_angle <= 0
            or N == 0
            or len(bbp) != N
            or np.all(np.isnan(chlf))
            or np.all(np.isnan(bbp))
        ):
            return chlf

        # Search window: mixed layer (Sackmann) or euphotic zone (Swart).
        if window == "mld":
            finite_mld = np.asarray(profile["MLD"].values, dtype=float)
            finite_mld = finite_mld[np.isfinite(finite_mld)]
            z_win = float(finite_mld[0]) if finite_mld.size else np.nan
        else:
            par = np.asarray(profile[self.par_var].values, dtype=float)
            z_win = estimate_euphotic_depth(par, depth)
        if not np.isfinite(z_win) or z_win <= 0:
            return chlf

        within = (depth <= z_win) & np.isfinite(depth)
        fratio = np.divide(
            chlf, bbp, out=np.full_like(chlf, np.nan), where=(bbp != 0)
        )
        fratio_within = np.where(within, fratio, np.nan)
        if np.all(np.isnan(fratio_within)):
            return chlf

        idx_rmax = np.nanargmax(fratio_within)
        r_max = fratio[idx_rmax]
        rmax_depth = float(depth[idx_rmax])

        # Reset F to bbp x R_max from the surface to the depth of the max ratio.
        chl_corr = np.copy(chlf)
        fill = (depth <= rmax_depth) & np.isfinite(bbp) & (~np.isnan(chlf))
        chl_corr[fill] = bbp[fill] * r_max

        self._explain(
            profile,
            depth=depth,
            ratio=fratio,
            window=window,
            z_win=z_win,
            z_ref=rmax_depth,
            r_max=r_max,
            rmax_depth=rmax_depth,
        )

        # Never let the correction reduce fluorescence (fmax ignores NaNs).
        result = np.fmax(chlf, chl_corr)
        self._warn_if_correction_blows_up(chlf, result)
        return result

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
        "swart2015": "Swart 2015",
        "sackmann2008": "Sackmann 2008",
    }

    #: One- or two-line plain-language summary of each method's correction,
    #: shown as a caption beneath the decision panel.
    _METHOD_DESCRIPTIONS = {
        "xing2012": (
            "Quenching depth (QD) is the depth of max CHLA in range 0 – MLD.\n"
            "Correction sets all CHLA values above QD to that max CHLA."
        ),
        "biermann2015": (
            "Reference is the max CHLA within the euphotic zone (0 – Zeu).\n"
            "Correction lifts all CHLA above that depth to the reference."
        ),
        "xing2018": (
            "NPQ layer = 0 to the shallower of MLD and the iPAR=15 depth.\n"
            "R_max = max CHLA/bbp there; CHLA is reset to bbp × R_max."
        ),
        "terrats2020": (
            "Deep mixing uses Xing 2018 (max CHLA/bbp in the NPQ layer).\n"
            "Shallow mixing applies the XB18 sigmoid below MLD, bbp × ratio above."
        ),
        "hemsley2015": (
            "One global night CHLA–bbp regression is fit for the deployment.\n"
            "Daytime CHLA over the euphotic zone is set to slope × bbp + intercept."
        ),
        "thomalla2018": (
            "Each day profile uses its preceding night's CHLA:bbp ratio.\n"
            "Above QD, CHLA is set to (night CHLA:bbp) × day bbp where it raises it."
        ),
        "sackmann2008": (
            "R_max = max CHLA/bbp within the mixed layer (0 – MLD).\n"
            "From the surface to that depth, CHLA is reset to bbp × R_max."
        ),
        "swart2015": (
            "R_max = max CHLA/bbp within the euphotic zone (0 – Zeu).\n"
            "From the surface to that depth, CHLA is reset to bbp × R_max."
        ),
    }

    def generate_diagnostics(self):
        """One figure summarising the quenching correction.

        **Left** — a 5x2 grid of the no-correction baseline and every method
        option (unimplemented ones as placeholders). Each implemented method is
        run over *all* the day profiles and scored against each one's
        nearest-in-time night profile, paired per depth bin within the top
        ``COMPARE_SURFACE_LIMIT_METRES`` where quenching acts. Each panel shows
        the scatter, the 1:1 line, the regression fit, and RMSE/Bias/R2; the
        shared sample size ``n`` is noted once in the spare cell.

        **Top right** — the original and corrected fluorescence as depth-time
        sections, plus a map of which points the correction actually changed.

        **Bottom right** — an example day profile for the *configured* method
        (unchanged points, the original quenched values, and the corrected
        values), marking the method's own quenching depth, beside a panel
        showing *how* that method chose its correction reference: for
        per-profile methods the example profile's internals (the searched layer
        and the picked reference point); for the night-reference methods the
        deployment-wide object used (Hemsley's global night fl-bbp regression,
        Thomalla's night fl:bbp ratio profile). A short caption spanning both
        describes the configured method.
        """
        mpl.use("tkagg")

        # The panels re-run every method over many profiles; only the configured
        # method's real correction (above) should raise the blow-up warning.
        self._suppress_warn = True

        if not hasattr(self, "sun_args"):
            self.log("Solar inputs unavailable; cannot build quenching diagnostics.")
            return

        fig = plt.figure(figsize=(21, 12), dpi=120)
        # Left: the method-comparison grid (unchanged, full height). Right: the
        # depth-time sections along the top - wide and short - with the example
        # profile and its decision panel side by side on the bottom row.
        # Tight outer margins so the panels fill the figure (leaving just enough
        # for the suptitle, the outer tick/axis labels and the colourbars).
        outer = fig.add_gridspec(
            1, 2, width_ratios=[1.4, 2.2], wspace=0.14,
            left=0.045, right=0.965, top=0.93, bottom=0.055,
        )

        self._draw_method_comparison(fig, outer[0, 0])
        right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.3)
        self._draw_timeseries(fig, right[0, 0])
        self._draw_example_profiles(fig, right[1, 0])

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
            "sackmann2008": (
                self.apply_sackmann2008_quenching_correction,
                {"MLD", self.bbp_var},
            ),
            "swart2015": (
                self.apply_swart2015_quenching_correction,
                {self.bbp_var, self.par_var},
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

        # 5x2 grid: the no-correction baseline, every method option (any that
        # can't run for want of an input variable shown as placeholders), one
        # spare cell.
        panels = ["none"] + self.parameter_schema["method"]["options"]
        ncols, nrows = 2, 5
        gl = subspec.subgridspec(nrows, ncols, hspace=0.55, wspace=0.34)

        # x-label only on the lowest scatter panel of each column, so "night F"
        # never reads into the title of the panel below it.
        cells = []
        bottom_scatter = {}
        for i, key in enumerate(panels):
            row, col = divmod(i, ncols)
            is_placeholder = key not in implemented and key != "none"
            has_scatter = (not is_placeholder) and bool(results.get(key))
            cells.append((row, col, key, is_placeholder))
            if has_scatter:
                bottom_scatter[col] = max(bottom_scatter.get(col, -1), row)

        for row, col, key, is_placeholder in cells:
            ax = fig.add_subplot(gl[row, col])
            placeholder = (
                f"{self._METHOD_LABELS.get(key, key)}\n(not implemented)"
                if is_placeholder
                else None
            )
            self._draw_scatter_panel(
                ax,
                self._METHOD_LABELS.get(key, key),
                results.get(key),
                show_xlabel=(row == bottom_scatter.get(col)),
                show_ylabel=(col == 0),
                placeholder=placeholder,
            )

        # The spare bottom-right cell holds the shared sample size, so the
        # per-panel stats boxes don't each repeat it. n is the number of paired
        # day/night surface depth-bin medians behind every panel's statistics
        # (the same across panels, so one figure suffices).
        n_values = [r["n"] for r in results.values() if r]
        if n_values:
            spare = fig.add_subplot(gl[nrows - 1, ncols - 1])
            spare.axis("off")
            spare.text(
                0.5,
                0.5,
                f"n = {max(n_values)}\npaired day/night\nsurface depth-bin medians",
                ha="center",
                va="center",
                transform=spare.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.7", alpha=0.9),
            )

    def _day_night_pairs(self):
        """List of ``(midday_profile, midnight_profile)`` numbers, nearest in time.

        Faithful to the Thomalla et al. (2018) Fig. 4 comparison: only profiles
        near solar noon (peak sun, maximum quenching) and solar midnight (no
        quenching) are used, selected with a
        +/-``MIDDAY_MIDNIGHT_WINDOW_HOURS`` solar-time window (see
        :meth:`_hours_from_solar_noon`). Restricting to these extremes makes the
        regression the demanding worst-case test the figure intends; the sample
        size then comes from pairing per depth bin over the surface layer (see
        :meth:`_score`), not from admitting weakly-quenched dawn/dusk profiles.

        Each midday profile is paired with its nearest-in-time midnight profile.
        Capped at ``MAX_COMPARE_PROFILES`` midday profiles (evenly sampled) so
        the comparison stays responsive; the cap is logged when it bites.
        """
        pns = [int(pn) for pn in self.sun_args.index]
        times = {
            pn: pd.to_datetime(self.sun_args.loc[pn, "TIME"]).value for pn in pns
        }
        from_noon = {pn: self._hours_from_solar_noon(pn) for pn in pns}
        window = MIDDAY_MIDNIGHT_WINDOW_HOURS
        midday = [pn for pn in pns if from_noon[pn] <= window]
        midnight = [pn for pn in pns if from_noon[pn] >= 12.0 - window]
        self.log(
            f"Method comparison: {len(midday)} midday and {len(midnight)} "
            f"midnight profile(s) within +/-{window:g} h of solar noon/midnight."
        )
        if not midday or not midnight:
            self.log(
                "Too few midday/midnight profiles for the method comparison; "
                f"widen MIDDAY_MIDNIGHT_WINDOW_HOURS (currently {window:g} h)."
            )
            return []

        midnight_times = np.array([times[pn] for pn in midnight])
        pairs = [
            (d, midnight[int(np.argmin(np.abs(midnight_times - times[d])))])
            for d in midday
        ]
        if len(pairs) > MAX_COMPARE_PROFILES:
            keep = np.linspace(0, len(pairs) - 1, MAX_COMPARE_PROFILES).astype(int)
            pairs = [pairs[i] for i in keep]
            self.log(
                f"Method comparison capped to {MAX_COMPARE_PROFILES} midday profiles "
                f"(of {len(midday)}) for speed."
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
        # DEPTH is positive-down, so the surface window is bins with key <= this.
        max_key = int(np.floor(COMPARE_SURFACE_LIMIT_METRES / COMPARE_BIN_METRES))
        xs, ys = [], []
        for day_pn, night_pn in pairs:
            day = day_dv.get(day_pn)
            night = night_dv.get(night_pn)
            if day is None or night is None:
                continue
            day_bins = self._bin_medians(day[0], day[1])
            night_bins = self._bin_medians(night[0], night[1])
            for k in day_bins.keys() & night_bins.keys():
                if k > max_key:  # deeper than the surface window -> skip
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
        bias = float(np.mean(resid))
        mean_night = float(np.mean(x))
        return {
            "x": x,
            "y": y,
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r2": float(fit.rvalue ** 2),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "bias": bias,
            # Bias relative to the mean night fluorescence, as a percentage.
            "bias_pct": 100.0 * bias / mean_night if mean_night != 0 else np.nan,
            "n": int(x.size),
        }

    def _draw_scatter_panel(
        self, ax, label, stats,
        show_xlabel=True, show_ylabel=True, placeholder=None,
    ):
        if placeholder is not None:
            ax.text(
                0.5,
                0.5,
                placeholder,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="0.5",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set(color="0.85")
            return

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
            f"Bias={stats['bias_pct']:+.1f}%\n"
            f"R$^2$={stats['r2']:.2f}"
        )
        # Stats in the top-left corner, above the 1:1 line where the scatter is
        # sparse; the wider panels keep them clear of the data.
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
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=6)
        # Only the bottom row carries the "night F" x-label (it otherwise reads
        # into the panel below it); only the left column carries the y-label.
        if show_xlabel:
            ax.set_xlabel("night F", fontsize=6.5)
        if show_ylabel:
            ax.set_ylabel("day F (corr)", fontsize=6.5)

    # --- Middle column: example profiles ------------------------------

    def _draw_example_profiles(self, fig, subspec):
        # Two plots side by side - the example day profile and the decision
        # panel - with a single caption spanning beneath both.
        gm = subspec.subgridspec(
            2, 2, height_ratios=[1.0, 0.16], hspace=0.4, wspace=0.26
        )
        day_pn, _ = self._example_profiles()

        # Capture the configured method's decision internals for the day profile
        # so the day plot can mark the quenching depth and the decision panel can
        # show *how* the correction reference was chosen.
        info = self._explain_profile(day_pn)
        z_ref = info.get("z_ref") if info else None

        self._draw_profile_change(
            fig.add_subplot(gm[0, 0]),
            day_pn,
            f"Example day profile (#{day_pn})",
            ref_depth=z_ref,
        )
        self._draw_decision_panel(fig.add_subplot(gm[0, 1]), info)
        self._draw_method_description(fig.add_subplot(gm[1, :]))

    def _draw_method_description(self, ax):
        """Caption (below the decision panel) describing the configured method."""
        ax.axis("off")
        text = self._METHOD_DESCRIPTIONS.get(
            self.method.lower(), "No description available for this method."
        )
        # Reads as a plain caption for the decision panel above, not a boxed
        # legend. The descriptions are pre-wrapped so the lines stay within the
        # middle column and don't run into the side plots.
        ax.text(
            0.5,
            0.98,
            text,
            ha="center",
            va="top",
            fontsize=7.5,
            style="italic",
            color="0.35",
            linespacing=1.3,
            transform=ax.transAxes,
        )

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
        most; the night profile is the nearest-in-time nighttime profile.
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
                label=f"Quenching depth ({ref_depth:.0f} m)",
            )
        ax.set_xlabel(self.apply_to, fontsize=8)
        ax.set_ylabel("DEPTH", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
        # Cap the view at 200 m and invert so the surface is at the top; the
        # quenching layer and correction sit near the surface.
        bottom, top = ax.get_ylim()
        ax.set_ylim(min(top, 200.0), bottom)

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
            "sackmann2008": self._decision_max_ratio,
            "swart2015": self._decision_max_ratio,
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
        within = valid & (depth <= mld)
        ax.scatter(chlf[valid], depth[valid], s=12, c="0.7", label="F (all)")
        ax.scatter(chlf[within], depth[within], s=14, c="#3b7dd8", label="within MLD")
        ax.axhline(mld, ls=":", c="black", lw=1.1, label=f"MLD ({mld:.0f} m)")
        ax.scatter(
            [f_ref], [z_ref], marker="o", s=90, c="#d1495b",
            edgecolors="black", linewidths=0.6, zorder=5, label="reference Fmax",
        )
        ax.set_title("Xing 2012: reference = max F within MLD", fontsize=9)
        self._decision_axes(ax, self.apply_to)

    def _decision_biermann(self, ax, info):
        depth, chlf = info["depth"], info["chlf"]
        zeu, z_ref, f_ref = info["zeu"], info["z_ref"], info["f_ref"]
        valid = np.isfinite(depth) & np.isfinite(chlf)
        # Within the euphotic zone is shallower than Zeu.
        within = valid & (depth <= zeu)
        ax.scatter(chlf[valid], depth[valid], s=12, c="0.7", label="F (all)")
        ax.scatter(chlf[within], depth[within], s=14, c="#3b7dd8", label="within Zeu")
        ax.axhline(zeu, ls=":", c="#2ca02c", lw=1.1, label=f"Zeu ({zeu:.0f} m)")
        ax.scatter(
            [f_ref], [z_ref], marker="o", s=90, c="#d1495b",
            edgecolors="black", linewidths=0.6, zorder=5, label="max F within Zeu",
        )
        ax.set_title("Biermann 2015: reference = max F within Zeu", fontsize=9)
        self._decision_axes(ax, self.apply_to)

    def _decision_xing_terrats(self, ax, info):
        label = self._METHOD_LABELS.get(self.method.lower(), self.method)
        mld, ipar15 = info["mld"], info["ipar15"]
        if info.get("branch") == "shallow":
            # Terrats shallow-mixing branch: plot the sigmoid scaling applied
            # below the MLD (this is the de-quenching "decision" here).
            z, s = info["sigmoid_depth"], info["sigmoid_scale"]
            # Drop NaN sigmoid points (PAR is NaN on interleaved CTD-only rows);
            # a markerless line across non-contiguous finite points draws nothing.
            fin = np.isfinite(z) & np.isfinite(s)
            z, s = z[fin], s[fin]
            order = np.argsort(z)
            ax.plot(
                s[order], z[order], c="#3b7dd8", lw=1.4,
                marker="o", ms=2.5, label="sigmoid scale s(z)",
            )
            ax.axhline(mld, ls=":", c="black", lw=1.1, label=f"MLD ({mld:.0f} m)")
            ax.set_title(f"{label} (shallow mixing): F/s below MLD", fontsize=9)
            self._decision_axes(ax, "de-quench scale s(z)")
            return
        ratio, z_ref, depth = info["ratio"], info["z_ref"], info["depth"]
        r_max, rmax_depth = info["r_max"], info["rmax_depth"]
        valid = np.isfinite(depth) & np.isfinite(ratio)
        # Draw F/bbp as a profile line (ordered by depth) so its shape reads at a
        # glance, rather than as a cloud of points restating the x-axis.
        order = np.argsort(depth[valid])
        ax.plot(
            ratio[valid][order], depth[valid][order],
            c="0.6", lw=1.1, marker="o", ms=2.5, label="F/bbp profile",
        )
        # Shade the NPQ (quenching) layer: surface down to z_ref, the depths the
        # correction rewrites. z_ref is the shallower of MLD and the iPAR=15 depth.
        ax.axhspan(
            0, z_ref, color="#3b7dd8", alpha=0.12,
            label=f"NPQ layer (0–{z_ref:.0f} m)",
        )
        ax.axhline(mld, ls=":", c="black", lw=1.1, label=f"MLD ({mld:.0f} m)")
        if np.isfinite(ipar15):
            ax.axhline(
                ipar15, ls="--", c="#9467bd", lw=1.1, label=f"iPAR=15 ({ipar15:.0f} m)"
            )
        # R_max: the largest (least-quenched) F/bbp in the NPQ layer. The whole
        # layer is reset to bbp × R_max, so this vertical line is the target ratio.
        ax.axvline(r_max, ls="--", c="#d1495b", lw=1.1, zorder=4)
        ax.scatter(
            [r_max], [rmax_depth], marker="o", s=90, c="#d1495b",
            edgecolors="black", linewidths=0.6, zorder=5,
            label=f"R_max = max F/bbp ({r_max:.2g})",
        )
        ax.set_title(f"{label} (deep mixing): reset F to bbp × R_max", fontsize=9)
        self._decision_axes(ax, "F/bbp ratio")

    def _decision_max_ratio(self, ax, info):
        label = self._METHOD_LABELS.get(self.method.lower(), self.method)
        depth, ratio = info["depth"], info["ratio"]
        z_win, z_ref = info["z_win"], info["z_ref"]
        r_max, rmax_depth = info["r_max"], info["rmax_depth"]
        win_name = "Zeu" if info["window"] == "zeu" else "MLD"
        valid = np.isfinite(depth) & np.isfinite(ratio)
        order = np.argsort(depth[valid])
        ax.plot(
            ratio[valid][order], depth[valid][order],
            c="0.6", lw=1.1, marker="o", ms=2.5, label="F/bbp profile",
        )
        # Shade the corrected layer: surface down to the depth of the max ratio.
        ax.axhspan(
            0, z_ref, color="#3b7dd8", alpha=0.12,
            label=f"corrected (0–{z_ref:.0f} m)",
        )
        win_colour = "#2ca02c" if win_name == "Zeu" else "black"
        ax.axhline(
            z_win, ls=":", c=win_colour, lw=1.1, label=f"{win_name} ({z_win:.0f} m)"
        )
        # R_max: the largest (least-quenched) F/bbp in the search window; the
        # layer above its depth is reset to bbp × R_max.
        ax.axvline(r_max, ls="--", c="#d1495b", lw=1.1, zorder=4)
        ax.scatter(
            [r_max], [rmax_depth], marker="o", s=90, c="#d1495b",
            edgecolors="black", linewidths=0.6, zorder=5,
            label=f"R_max = max F/bbp ({r_max:.2g})",
        )
        ax.set_title(f"{label}: reset F to bbp × R_max", fontsize=9)
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
        ax.plot(ref["ratio"], ref["z"], c="#3b7dd8", lw=1.4, label="night fl:bbp ratio")
        if zeu is not None and np.isfinite(zeu):
            ax.axhline(zeu, ls=":", c="#2ca02c", lw=1.1, label=f"Zeu ({zeu:.0f} m)")
        if np.isfinite(z_ref):
            ax.axhline(
                z_ref, ls="--", c="#ff7f0e", lw=1.2, label=f"QD ({z_ref:.0f} m)"
            )
        ax.set_title("Thomalla 2018: night fl:bbp ratio & QD", fontsize=9)
        self._decision_axes(ax, "night fl:bbp ratio")

    @staticmethod
    def _decision_axes(ax, xlabel):
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("DEPTH", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
        # Cap at 200 m and invert (surface on top) to match the example-profile
        # plot above; the reference and searched layer sit near the surface.
        bottom, top = ax.get_ylim()
        ax.set_ylim(min(top, 200.0), bottom)

    # --- Right column: depth-time sections ----------------------------

    def _draw_timeseries(self, fig, subspec):
        # A dedicated thin colourbar column so ax1/ax2 (which have colourbars)
        # keep the same width as ax3 (which does not) - otherwise a stolen
        # colourbar shrinks the top two and their time axes stop lining up.
        gr = subspec.subgridspec(
            3, 2, width_ratios=[1, 0.02], wspace=0.015, hspace=0.35
        )
        time = self.data["TIME"].values
        depth = self.data["DEPTH"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values

        # Keep points with a valid time/depth and a real CHLA value in the top
        # TIMESERIES_DEPTH_LIMIT m. Rows with NaN CHLA (e.g. CTD samples on the
        # same measurement axis) can never be corrected, so plotting them only
        # buries the real fluorescence under grey "unchanged" points.
        finite = (
            ~pd.isnull(time)
            & np.isfinite(depth)
            & np.isfinite(orig)
            & (depth <= TIMESERIES_DEPTH_LIMIT)
        )
        time, depth = time[finite], depth[finite]
        orig, corr = orig[finite], corr[finite]

        vmin, vmax = self._robust_vlim(orig, corr)

        # Zoom the depth axis so the deepest corrected point sits about two-thirds
        # down: e.g. a deepest QD of 20 m gives a 0-30 m window. Falls back to the
        # full window when no profile was corrected, and never zooms out past it.
        line_qd = self._section_quenching_depths()
        finite_qd = line_qd[np.isfinite(line_qd)]
        if finite_qd.size:
            depth_limit = min(
                TIMESERIES_DEPTH_LIMIT,
                max(TIMESERIES_DEPTH_MIN, 1.5 * float(np.max(finite_qd))),
            )
        else:
            depth_limit = TIMESERIES_DEPTH_LIMIT

        chla_cmap = palettes.get_cmap("chlorophyll")
        ax1 = fig.add_subplot(gr[0, 0])
        sc1 = ax1.scatter(
            time, depth, c=orig, cmap=chla_cmap, vmin=vmin, vmax=vmax,
            s=SECTION_MARKER_SIZE, rasterized=True,
        )
        ax1.set_title(f"Original fluorescence (top {depth_limit:.0f} m)", fontsize=9)

        ax2 = fig.add_subplot(gr[1, 0], sharex=ax1, sharey=ax1)
        sc2 = ax2.scatter(
            time, depth, c=corr, cmap=chla_cmap, vmin=vmin, vmax=vmax,
            s=SECTION_MARKER_SIZE, rasterized=True,
        )
        ax2.set_title(f"Quenching-corrected fluorescence (top {depth_limit:.0f} m)", fontsize=9)

        # Colour every section point by why the correction did/didn't touch it,
        # drawing from its own (unfiltered) categorisation so NaN and not-in-
        # profile points show too - the shared arrays above drop those.
        ax3 = fig.add_subplot(gr[2, 0], sharex=ax1, sharey=ax1)
        cat_time, cat_depth, cat_key = self._section_point_categories(
            depth_limit, SECTION_MAX_POINTS
        )
        for z, (key, label, color) in enumerate(SECTION_CATEGORY_STYLE):
            sel = cat_key == key
            n = int(np.count_nonzero(sel))
            if not n:
                continue
            ax3.scatter(
                cat_time[sel], cat_depth[sel], c=color,
                s=SECTION_MARKER_SIZE, rasterized=True, zorder=2 + z,
                label=f"{label} ({n})",
            )
        ax3.set_title("Quenching layer — points by correction status", fontsize=9)
        ax3.legend(fontsize=6.0, loc="lower right", framealpha=0.9, markerscale=8)

        ax1.set_ylim(depth_limit, 0)  # positive-down: surface at top

        # Colourbars go in the reserved column; ax3's cell is blanked so its
        # plot width still matches the two above it.
        for sc, row in ((sc1, 0), (sc2, 1)):
            cax = fig.add_subplot(gr[row, 1])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label(self.apply_to, fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        fig.add_subplot(gr[2, 1]).axis("off")
        for ax in (ax1, ax2):
            plt.setp(ax.get_xticklabels(), visible=False)
        for ax in (ax1, ax2, ax3):
            ax.set_ylabel("DEPTH", fontsize=8)
            ax.tick_params(labelsize=7)
        ax3.set_xlabel("TIME", fontsize=8)
        plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    def _section_quenching_depths(self):
        """Per-profile quenching depth (positive-down m), used to zoom the sections.

        The quenching depth is the deepest point the correction actually changed
        in each profile; ``NaN`` where undefined (night, or a profile the
        correction left untouched). The median sets the section depth window.
        """
        pnum = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values
        changed = np.isfinite(corr) & np.isfinite(orig) & (np.abs(corr - orig) > 1e-9)

        qd = []
        for pn in self.sun_args.index:
            idx = np.where(pnum == pn)[0]
            if idx.size == 0:
                continue
            in_profile = changed[idx]
            qd.append(float(np.max(depth[idx][in_profile])) if np.any(in_profile) else np.nan)
        return np.asarray(qd)

    def _section_point_categories(self, depth_limit, max_points):
        """Split every section point into 'corrected' vs 'uncorrected'.

        Debug aid for the bottom section panel. Returns ``(time, depth, cat)``
        for the plotted points - those in a profile with a finite time/depth and
        a real CHLA value, within ``depth_limit`` and subsampled to
        ``max_points``. ``cat`` is 'corrected' where the value actually changed
        and 'uncorrected' otherwise (nighttime points included); NaN-CHLA rows
        (CTD-only samples) and points not in any profile are dropped.
        """
        pnum = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        time = self.data["TIME"].values
        orig = self.data_copy[self.apply_to].values
        corr = self.data[self.output_as].values
        changed = np.isfinite(corr) & np.isfinite(orig) & (np.abs(corr - orig) > 1e-9)

        cat = np.where(changed, "corrected", "uncorrected")

        # Show only in-profile points with a real value; NaN-CHLA rows (CTD-only
        # samples) and points not in any profile are excluded from the panel.
        plot_mask = (
            ~pd.isnull(time)
            & np.isfinite(depth)
            & np.isfinite(orig)
            & np.isfinite(pnum)
            & (depth <= depth_limit)
        )
        idx = np.where(plot_mask)[0]
        if idx.size > max_points:
            keep = np.linspace(0, idx.size - 1, max_points).astype(int)
            idx = idx[keep]
            self.log(
                f"Section debug panel: subsampled {int(plot_mask.sum())} points "
                f"to {max_points} for plotting."
            )
        return time[idx], depth[idx], cat[idx]

    @staticmethod
    def _robust_vlim(*arrays):
        """2nd-98th percentile colour limits across the given arrays."""
        stacked = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size == 0:
            return None, None
        return float(np.percentile(stacked, 2)), float(np.percentile(stacked, 98))

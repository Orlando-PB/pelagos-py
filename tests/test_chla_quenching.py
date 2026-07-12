"""Tests the step 'CHLA Quenching' (src/pelagos_py/steps/processing/chla_quenching.py).

Covers the newly implemented backscatter- and euphotic-depth-based NPQ methods
and their helpers. The per-profile correction methods are exercised directly
(the solar-elevation lookup is stubbed) so no real data or pvlib call is needed.
"""

#   Test module import
from pelagos_py.steps.processing import chla_quenching

import numpy as np
import xarray as xr
import pytest

Quenching = chla_quenching.chla_quenching_correction
estimate_euphotic_depth = chla_quenching.estimate_euphotic_depth


def make_profile(chlf, depth, bbp=None, ipar=None, mld=None, profile_number=101.0):
    """Single-profile dataset in the step's negative-down DEPTH convention."""
    n = len(chlf)
    data = {
        "PROFILE_NUMBER": ("N_MEASUREMENTS", np.full(n, profile_number)),
        "DEPTH": ("N_MEASUREMENTS", np.asarray(depth, dtype=float)),
        "CHLA": ("N_MEASUREMENTS", np.asarray(chlf, dtype=float)),
    }
    if bbp is not None:
        data["BBP700"] = ("N_MEASUREMENTS", np.asarray(bbp, dtype=float))
    if ipar is not None:
        data["DOWNWELLING_PAR"] = ("N_MEASUREMENTS", np.asarray(ipar, dtype=float))
    if mld is not None:
        data["MLD"] = ("N_MEASUREMENTS", np.full(n, float(mld)))
    return xr.Dataset(data)


def make_step(sun_angle=40.0):
    """Bare step instance with the sun-elevation lookup stubbed to ``sun_angle``."""
    step = Quenching.__new__(Quenching)
    step.apply_to = "CHLA"
    step.bbp_var = "BBP700"
    step.par_var = "DOWNWELLING_PAR"
    step._sun_elevation = lambda profile: sun_angle
    return step


# --- helpers ---------------------------------------------------------------


def test_estimate_euphotic_depth_recovers_1pct_level():
    """A clean exponential PAR profile yields Zeu = ln(100) / Kd."""
    z = np.arange(0, 60, 2.0)
    par = 100 * np.exp(-0.1 * z)  # Kd = 0.1 -> Zeu = 46.05 m
    assert estimate_euphotic_depth(par, z) == pytest.approx(46.05, abs=0.1)


def test_estimate_euphotic_depth_invalid_inputs_return_nan():
    z = np.arange(0, 60, 2.0)
    assert np.isnan(estimate_euphotic_depth(np.full(z.size, np.nan), z))  # no data
    assert np.isnan(estimate_euphotic_depth(np.full(z.size, 50.0), z))  # flat -> no slope


def test_depth_of_ipar15_interpolates_and_clamps():
    z = np.array([0, 10, 20, 30.0])
    assert Quenching._depth_of_ipar15(z, np.array([100, 40, 10, 2.0])) == pytest.approx(
        18.333, abs=1e-2
    )
    # Whole profile brighter than 15 -> deepest sample; darker -> surface.
    assert Quenching._depth_of_ipar15(z, np.array([100, 90, 80, 70.0])) == 30.0
    assert Quenching._depth_of_ipar15(z, np.array([10, 8, 5, 2.0])) == 0.0


# --- Biermann 2015 ---------------------------------------------------------


def test_biermann_lifts_shallow_to_max_below_zeu():
    z_pos = np.array([2, 6, 10, 15, 20, 30, 45, 60, 80.0])
    chlf = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1])
    ipar = 200 * np.exp(-0.12 * z_pos)  # Zeu ~ 38 m
    prof = make_profile(chlf, -z_pos, ipar=ipar)

    out = make_step().apply_biermann2015_quenching_correction(prof)

    # Reference is the max fluorescence below Zeu (0.4 at 45 m); everything
    # shallower than that quenching depth is set to it, deeper is untouched.
    assert out[-2:].tolist() == [0.2, 0.1]
    assert np.all(out[:-2] == pytest.approx(0.4))


def test_biermann_no_par_signal_returns_unchanged():
    z_pos = np.array([2, 6, 10, 15, 20, 30.0])
    chlf = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.8])
    prof = make_profile(chlf, -z_pos, ipar=np.full(z_pos.size, np.nan))
    out = make_step().apply_biermann2015_quenching_correction(prof)
    assert np.array_equal(out, chlf)


# --- Xing 2018 / Terrats 2020 ---------------------------------------------


def _bbp_profile():
    z_pos = np.array([2, 6, 10, 15, 20, 30, 45, 60, 80.0])
    chlf = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2, 0.1])
    bbp = np.array([2, 2, 2, 2, 2, 1.5, 1, 0.6, 0.4]) * 1e-3
    return z_pos, chlf, bbp


def test_xing2018_resets_npq_layer_to_bbp_times_rmax():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)  # iPAR=15 ~ 21.6 m, above MLD=25 -> deep
    prof = make_profile(chlf, -z_pos, bbp=bbp, ipar=ipar, mld=-25.0)

    out = make_step().apply_xing2018_quenching_correction(prof)

    # Within the NPQ layer bbp is constant and R_max hits the 1.0 peak, so the
    # suppressed near-surface points are lifted to 1.0; deeper points unchanged.
    assert np.all(out[:5] == pytest.approx(1.0))
    assert out[5:].tolist() == chlf[5:].tolist()


def test_terrats_shallow_mixing_runs_and_never_reduces():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 800 * np.exp(-0.05 * z_pos)  # iPAR=15 deep (~80 m) with shallow MLD
    prof = make_profile(chlf, -z_pos, bbp=bbp, ipar=ipar, mld=-10.0)

    out = make_step().apply_terrats2020_quenching_correction(prof)

    assert out.shape == chlf.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= chlf - 1e-12)  # correction never reduces fluorescence


def test_backscatter_methods_are_noop_at_night():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)
    prof = make_profile(chlf, -z_pos, bbp=bbp, ipar=ipar, mld=-25.0)
    out = make_step(sun_angle=-5.0).apply_terrats2020_quenching_correction(prof)
    assert np.array_equal(out, chlf, equal_nan=True)


# --- Hemsley 2015 ----------------------------------------------------------


def test_hemsley_replaces_euphotic_zone_with_bbp_estimate():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)  # Kd=0.12 -> Zeu ~ 38.4 m
    prof = make_profile(chlf, -z_pos, bbp=bbp, ipar=ipar)

    step = make_step()
    step._hemsley_regression = {"slope": 100.0, "intercept": 0.1}
    out = step.apply_hemsley2015_quenching_correction(prof)

    # Over the euphotic zone (z <= ~38 m) fluorescence becomes m*bbp + c;
    # deeper points are untouched.
    assert out[:6] == pytest.approx(100.0 * bbp[:6] + 0.1)
    assert out[6:].tolist() == chlf[6:].tolist()


def test_hemsley_no_regression_returns_unchanged():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)
    prof = make_profile(chlf, -z_pos, bbp=bbp, ipar=ipar)
    step = make_step()
    step._hemsley_regression = None
    assert np.array_equal(step.apply_hemsley2015_quenching_correction(prof), chlf)


# --- Thomalla 2018 ---------------------------------------------------------


def test_bin_night_averages_ratio_per_depth_bin():
    z = np.array([0.5, 1.5, 1.6, 2.5])
    fl = np.array([1.0, 2.0, 4.0, 6.0])
    bbp = np.full(4, 1e-3)
    ref = Quenching._bin_night(z, fl, bbp)
    assert ref["z"].tolist() == [0.5, 1.5, 2.5]
    # Middle bin averages fl (3.0) before taking the ratio 3.0 / 1e-3.
    assert ref["ratio"] == pytest.approx([1000.0, 3000.0, 6000.0])


def test_quenching_depth_picks_steepest_gradient_point():
    z = np.array([2, 6, 10, 15, 20, 30.0])
    fl_day = np.array([0.4, 0.5, 0.7, 0.9, 1.0, 0.75])
    fl_night = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.75])  # D returns to 0 by depth
    assert Quenching._quenching_depth(z, fl_day, fl_night, zeu=38.0) == 15.0


def test_thomalla_corrects_above_quenching_depth_and_only_raises():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)  # Zeu ~ 38 m
    prof = make_profile(chlf, -z_pos, bbp=bbp, ipar=ipar)

    step = make_step()
    # Night reference: constant fl:bbp ratio so corrected = 500*bbp (=1.0 at
    # surface), and a mean night fluorescence that is unquenched near surface.
    step._night_refs = [{"z": z_pos, "fl": 500.0 * bbp, "ratio": np.full(z_pos.size, 500.0)}]
    step._thomalla_day_night = {101: 0}

    out = step.apply_thomalla2018_quenching_correction(prof)

    # QD resolves to 15 m; surface points are lifted to 500*bbp (1.0) where that
    # exceeds the quenched value, and everything deeper is left unchanged.
    assert out[:4] == pytest.approx(1.0)
    assert out[4:].tolist() == chlf[4:].tolist()


def test_thomalla_unmapped_profile_returns_unchanged():
    z_pos, chlf, bbp = _bbp_profile()
    ipar = 200 * np.exp(-0.12 * z_pos)
    prof = make_profile(chlf, -z_pos, bbp=bbp, ipar=ipar)
    step = make_step()
    step._night_refs = []
    step._thomalla_day_night = {}  # this profile has no paired night
    assert np.array_equal(step.apply_thomalla2018_quenching_correction(prof), chlf)

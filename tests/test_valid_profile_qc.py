import numpy as np
import xarray as xr

from pelagos_py.steps.quality_control.valid_profile_qc import valid_profile_qc


def make_dataset(profile_numbers, depths):
    return xr.Dataset(
        {
            "PROFILE_NUMBER": ("N_MEASUREMENTS", np.array(profile_numbers, dtype=float)),
            "DEPTH": ("N_MEASUREMENTS", np.array(depths, dtype=float)),
        },
        coords={"N_MEASUREMENTS": range(len(profile_numbers))},
    )


def test_flag_assignment():
    """Exercises all four outcomes in one go:
    good (1), too short (4), out of depth range (3) and no profile (9)."""
    profile_numbers = [1, 1, 1, 2, 3, 3, 3, np.nan]
    depths = [-10, -20, -30, -10, 50, 60, 70, -5]
    #          \-- profile 1 --/  \2/  \- profile 3 -/  \-NaN-/

    qc = valid_profile_qc(
        make_dataset(profile_numbers, depths),
        profile_length=3,
        depth_range=(-1000, 0),
    )
    flags = qc.return_qc()

    expected = [1, 1, 1, 4, 3, 3, 3, 9]
    assert list(flags["PROFILE_NUMBER_QC"].values) == expected


def test_all_valid_profiles_pass():
    """Long-enough profiles with in-range depth are all flagged good (1)."""
    qc = valid_profile_qc(
        make_dataset([1, 1, 1, 2, 2, 2], [-5, -10, -15, -20, -25, -30]),
        profile_length=2,
        depth_range=(-1000, 0),
    )
    flags = qc.return_qc()

    assert list(flags["PROFILE_NUMBER_QC"].values) == [1] * 6

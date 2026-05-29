import numpy as np
import xarray as xr

from toolbox.steps.custom.qc.range_qc import range_qc


def _ds(**vars_):
    length = len(next(iter(vars_.values())))
    return xr.Dataset(
        {k: ("N_MEASUREMENTS", np.asarray(v, dtype=float)) for k, v in vars_.items()},
        coords={"N_MEASUREMENTS": range(length)},
    )


def test_outside_inside_and_equals_rules():
    """All three rule modes flag the points they should."""
    data = _ds(
        TEMP=[-5.0, 15.0, 45.0],     # outside [-2.5, 40] -> 4 on rows 0 and 2
        PRES=[-500.0, -1.0, 5.0],    # inside (-999, -2) -> 4 on row 0
        DOXY=[0.0, 12.0, -999.0],    # equals 0 -> 4; equals -999 -> 3
    )
    flags = range_qc(
        data,
        variable_ranges={
            "TEMP": {4: [-2.5, 40]},
            "PRES": {4: {"inside": [-999, -2]}},
            "DOXY": {4: 0.0, 3: {"equals": [-999.0]}},
        },
    ).return_qc()

    assert list(flags["TEMP_QC"].values) == [4, 1, 4]
    assert list(flags["PRES_QC"].values) == [4, 1, 1]
    assert list(flags["DOXY_QC"].values) == [4, 1, 3]


def test_also_flag_does_not_downgrade_worse_flag():
    """The fix under test: a less-severe also_flag must not overwrite a worse flag."""
    # CNDC's own rule flags row 0 as 4. PRES also_flags CNDC with 3 on the same row.
    # The 4 must survive.
    data = _ds(
        PRES=[-500.0, 5.0],
        CNDC=[999.0, 1.0],
    )
    flags = range_qc(
        data,
        variable_ranges={
            "PRES": {3: {"inside": [-999, -2]}},
            "CNDC": {4: [0, 100]},
        },
        also_flag={"PRES": ["CNDC"]},
    ).return_qc()

    assert list(flags["PRES_QC"].values) == [3, 1]
    assert list(flags["CNDC_QC"].values) == [4, 1]


def test_also_flag_propagates_to_untested_variable():
    """also_flag still propagates flags to a variable that has no rule of its own."""
    data = _ds(PRES=[-500.0, 5.0], TEMP=[10.0, 10.0])
    flags = range_qc(
        data,
        variable_ranges={"PRES": {4: {"inside": [-999, -2]}}},
        also_flag={"PRES": ["TEMP"]},
    ).return_qc()

    assert list(flags["PRES_QC"].values) == [4, 1]
    assert list(flags["TEMP_QC"].values) == [4, 1]

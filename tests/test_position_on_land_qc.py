import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.position_on_land_qc import position_on_land_qc
from utils.test_utils import create_mock_dataset

# Test configuration variables
# Ocean coordinates (Mid Atlantic, Central Pacific)
TEST_WATER_LATS = [0.0, 0.0]
TEST_WATER_LONS = [-30.0, -140.0]

# Land coordinates (Kansas USA, Alice Springs AUS)
TEST_LAND_LATS = [39.0, -23.7] 
TEST_LAND_LONS = [-98.0, 133.8]

# Missing or invalid coordinates
TEST_NAN_LATS = [np.nan, 39.0]
TEST_NAN_LONS = [-30.0, np.nan]


def test_missing_variables():
    data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
    qc_step = position_on_land_qc(data)
    
    with pytest.raises(KeyError):
        qc_step.return_qc()


@pytest.mark.parametrize("lats, lons, expected_flags", [
    (TEST_WATER_LATS, TEST_WATER_LONS, [1, 1]),
    (TEST_LAND_LATS, TEST_LAND_LONS, [4, 4]),
    (TEST_NAN_LATS, TEST_NAN_LONS, [1, 1]),
])
def test_locations(lats, lons, expected_flags):
    data = create_mock_dataset(lats=lats, lons=lons)
    qc_step = position_on_land_qc(data)
    flags = qc_step.return_qc()

    assert list(flags["LATITUDE_QC"].values) == expected_flags
    assert list(flags["LONGITUDE_QC"].values) == expected_flags


@patch("toolbox.steps.custom.qc.position_on_land_qc.plt.show")
def test_plot_diagnostics(mock_show):
    data = create_mock_dataset(lats=TEST_WATER_LATS, lons=TEST_WATER_LONS)
    qc_step = position_on_land_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_show.assert_called_once_with(block=True)
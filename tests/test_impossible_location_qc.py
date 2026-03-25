import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch

from toolbox.steps.custom.qc.impossible_location_qc import impossible_location_qc
from utils.test_utils import create_mock_dataset

TEST_GOOD_LATS = [45.0, -89.9]
TEST_GOOD_LONS = [179.9, -179.9]

TEST_BAD_LATS = [95.0, -100.0] 
TEST_BAD_LONS = [185.0, -185.0]

TEST_NAN_LATS = [np.nan, 45.0]
TEST_NAN_LONS = [10.0, np.nan]

TEST_ZERO_LATS = [0.0, 0.0]
TEST_ZERO_LONS = [0.0, 0.0]


def test_missing_variables():
    data = xr.Dataset({"TEMP": ("N_MEASUREMENTS", [10.0, 12.0])})
    qc_step = impossible_location_qc(data)
    
    with pytest.raises(KeyError):
        qc_step.return_qc()


@pytest.mark.parametrize("lats, lons, expected_lat, expected_lon", [
    (TEST_GOOD_LATS, TEST_GOOD_LONS, [1, 1], [1, 1]),
    (TEST_ZERO_LATS, TEST_ZERO_LONS, [1, 1], [1, 1]),
    (TEST_BAD_LATS, TEST_BAD_LONS, [4, 4], [4, 4]),
    (TEST_NAN_LATS, TEST_NAN_LONS, [9, 1], [1, 9]),
])
def test_locations(lats, lons, expected_lat, expected_lon):
    data = create_mock_dataset(lats=lats, lons=lons)
    qc_step = impossible_location_qc(data)
    flags = qc_step.return_qc()

    assert list(flags["LATITUDE_QC"].values) == expected_lat
    assert list(flags["LONGITUDE_QC"].values) == expected_lon


@patch("toolbox.steps.custom.qc.impossible_location_qc.plt.show")
def test_plot_diagnostics(mock_show):
    data = create_mock_dataset(lats=TEST_GOOD_LATS, lons=TEST_GOOD_LONS)
    qc_step = impossible_location_qc(data)
    
    qc_step.return_qc()
    qc_step.plot_diagnostics()
    
    mock_show.assert_called_once_with(block=True)
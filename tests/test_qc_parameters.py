"""Parameter-schema behaviour for QC checks and the schema-aware pipeline validator."""

import logging

import pytest

from pelagos_py.steps.quality_control.range_qc import range_qc
from pelagos_py.steps.quality_control.spike_qc import spike_qc
from pelagos_py.steps.quality_control.impossible_date_qc import impossible_date_qc
from pelagos_py.utils.valid_config_check import check_pipeline_variables


LOGGER = logging.getLogger("test")


# --- Dynamic QC resolve their I/O from parameters --------------------------------


def test_dynamic_qc_derives_required_variables():
    qc = range_qc(
        None,
        variable_ranges={"PRES": {4: [-5, 0]}},
        also_flag={"PRES": ["CNDC"]},
    )
    assert qc.required_variables == ["PRES"]
    assert set(qc.qc_outputs) == {"PRES_QC", "CNDC_QC"}


def test_optional_params_get_defaults():
    # also_flag/plot are optional; window_size defaults to 50.
    qc = spike_qc(None, variables={"PRES": 2})
    assert qc.window_size == 50
    assert qc.also_flag == {}
    assert "PROFILE_NUMBER" in qc.required_variables


def test_test_depth_range_adds_depth_requirement():
    without = range_qc(None, variable_ranges={"PRES": {4: [-5, 0]}})
    assert "DEPTH" not in without.required_variables

    with_range = range_qc(
        None, variable_ranges={"PRES": {4: [-5, 0]}}, test_depth_range=[-100, 0]
    )
    assert "DEPTH" in with_range.required_variables


def test_missing_required_param_raises():
    with pytest.raises(ValueError, match="variable_ranges"):
        range_qc(None, also_flag={})


def test_unknown_param_raises():
    with pytest.raises(ValueError, match="bogus"):
        impossible_date_qc(None, bogus=1)


# --- Validator is schema-aware (steps and nested QC) -----------------------------


def test_validator_flags_missing_required_step_param():
    steps = [{"name": "Data Export", "parameters": {"export_format": "netcdf"}}]
    with pytest.raises(ValueError, match="output_path"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_flags_missing_required_qc_param():
    steps = [{"name": "Apply QC", "parameters": {"qc_settings": {"range qc": {}}}}]
    with pytest.raises(ValueError, match="variable_ranges"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_passes_valid_qc_config():
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"impossible date qc": {}}},
        }
    ]
    assert check_pipeline_variables(steps, LOGGER) is True


# --- Validator catches QC variable ordering mistakes -----------------------------


def test_validator_flags_qc_var_produced_by_later_step():
    # PAR irregularity needs PROFILE_NUMBER, which "Find Profiles" produces — but
    # here it runs *after* the QC, so the requirement is unmet at that point.
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"PAR irregularity qc": {}}},
        },
        {"name": "Find Profiles", "parameters": {}},
    ]
    with pytest.raises(ValueError, match="PROFILE_NUMBER"):
        check_pipeline_variables(steps, LOGGER)


def test_validator_passes_qc_var_in_correct_order():
    steps = [
        {"name": "Find Profiles", "parameters": {}},
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"PAR irregularity qc": {}}},
        },
    ]
    # DOWNWELLING_PAR is also required by PAR irregularity but is a file-native
    # variable no step produces, so it must not be flagged.
    assert check_pipeline_variables(steps, LOGGER) is True


def test_validator_ignores_file_native_qc_requirement():
    # With no producing step in the pipeline, a missing requirement is assumed to
    # come from the input data file and is left for the run-time check.
    steps = [
        {
            "name": "Apply QC",
            "parameters": {"qc_settings": {"PAR irregularity qc": {}}},
        }
    ]
    assert check_pipeline_variables(steps, LOGGER) is True

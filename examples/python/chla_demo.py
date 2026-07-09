# TEMP: CHLA-focused cut of internal_config_demo.py — only the steps needed to
# reach the two CHLA steps (Deep Correction + Chla Quenching Correction), with
# their diagnostics turned on. The Chla Quenching MLD keys off DENSITY, so the
# CTD -> salinity -> density chain (and Find Profiles for PROFILE_NUMBER) is
# kept; the BBP, format-check and QC-only steps are dropped.
import yaml
from pelagos_py.pipeline import Pipeline

BASE_CONFIG_YAML = """
pipeline:
  name: CHLA Demo Pipeline
  description: Minimal pipeline exercising the CHLA steps with diagnostics on
  log_file: None

steps:
  - name: Load OG1
    parameters:
      file_path: examples/data/OG1/Nelson_646_R.nc
    diagnostics: false

  - name: Correct Values
    parameters:
      target_variable: CNDC
      slope: 10.0
      intercept: 0.0
      expected_range: [20, 45]
      corrected_units: mS/cm
    diagnostics: false

  - name: Interpolate Data
    parameters:
      qc_handling_settings:
        flag_filter_settings:
          PRES: [3, 4, 9]
          LATITUDE: [3, 4, 9]
          LONGITUDE: [3, 4, 9]
        reconstruction_behaviour: replace
    diagnostics: false

  - name: Derive CTD
    parameters:
      to_derive: [DEPTH]
    diagnostics: false

  - name: Find Profiles
    parameters:
    diagnostics: false

  - name: Salinity Adjustment
    parameters:
      qc_handling_settings:
        flag_filter_settings:
          CNDC: [3, 4, 9]
          TEMP: [3, 4, 9]
          PROFILE_NUMBER: [3, 4, 9]
        reconstruction_behaviour: reinsert
        flag_mapping:
          0: 5
          1: 5
          2: 5
      filter_window_size: 21
    diagnostics: false

  - name: Derive CTD
    parameters:
      to_derive: [PRAC_SALINITY, ABS_SALINITY, CONS_TEMP, DENSITY]
    diagnostics: false

  - name: Deep Correction
    parameters:
      apply_to: CHLA
      dark_value: null
      depth_var: PRES
      depth_threshold: 950
    diagnostics: true

  - name: Chla Quenching Correction
    parameters:
      method: Argo
      apply_to: CHLA
      mld_settings:
        threshold_on: DENSITY
        reference_depth: -10
        threshold: 0.05
      plot_profiles: [101, 200, 201, 300, 301, 400]
    diagnostics: true
"""


demo_config = yaml.safe_load(BASE_CONFIG_YAML)

p = Pipeline(config=demo_config)
p.run()

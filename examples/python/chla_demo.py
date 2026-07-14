# TEMP: CHLA-focused cut of internal_config_demo.py — only the steps needed to
# reach the two CHLA steps (Deep Correction + CHLA Quenching), with
# their diagnostics turned on. The Mixed Layer Depth step keys off DENSITY, so the
# CTD -> salinity -> density chain (and Find Profiles for PROFILE_NUMBER) is
# kept, along with the BBP chain (feeds the quenching step); the format-check
# and QC-only steps are dropped.
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

  # ---------------------- BACKSCATTER ------------------------
  # Convert raw backscatter angle (beta) into the particulate backscattering
  # coefficient BBP, then split the smooth baseline from isolated particle
  # spikes. The quenching step below uses the despiked BBP700_BASELINE so that
  # noise-level backscatter can't blow up the fl:bbp ratio.
  - name: BBP from Beta
    parameters:
      theta: 124               # Effective optical backscatter scattering angle (degrees)
      xfactor: 1.076           # Chi factor scaling particulate scattering to total backscatter
    diagnostics: false

  - name: Isolate BBP Spikes
    parameters:
      window_size: 50          # Filter window size in samples
      method: median           # Filter method used to determine the baseline
    diagnostics: false

  # ======================= CHLA SECTION START =======================
  # Mixed layer depth (defaults: auto method → DENSITY) — consumed by the
  # CHLA Quenching step below.
  - name: Mixed Layer Depth
    diagnostics: false

  # Global range test on the raw CHLA: probably-bad (3) outside 0.14-50,
  # bad (4) outside 0-100. Most-severe flag wins on overlap, so e.g. a
  # negative value is 4 and a 0.1 value is 3; anything in-band is good (1).
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            CHLA:
              3: [0.14, 50, outside]
              4: [0, 100, outside]
    diagnostics: false

  # Spike test on CHLA (per-profile MAD-style residual test).
  - name: Apply QC
    parameters:
      qc_settings:
        spike qc:
          variables:
            CHLA: 2
          window_size: 50
          plot: [CHLA]
    diagnostics: false

  - name: Deep Correction
    parameters:
      depth_threshold: 950
    diagnostics: false

  - name: CHLA Quenching
    parameters:
      method: thomalla2018
    diagnostics: true

  # Re-run the range test on the corrected CHLA_ADJUSTED, in case the deep
  # and quenching corrections pushed any values out of range.
  - name: Apply QC
    parameters:
      qc_settings:
        range qc:
          variable_ranges:
            CHLA_ADJUSTED:
              3: [0.14, 50, outside]
              4: [0, 100, outside]
    diagnostics: false
  # ======================== CHLA SECTION END ========================
"""


demo_config = yaml.safe_load(BASE_CONFIG_YAML)

p = Pipeline(config=demo_config)
p.run()

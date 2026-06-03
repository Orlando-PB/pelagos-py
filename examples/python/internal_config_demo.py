import matplotlib
matplotlib.use("TkAgg")

import os
import sys
import yaml

# Friendly Configuration Variables
INPUT_FILE = "/Users/orlpru/Desktop/Run_Pipeline/input/BIO_Carbon/Churchill_647.nc"
TOOLBOX_PATH = "../../src"
OUTPUT_DIRECTORY = "./"

BASE_CONFIG_YAML = """
pipeline:
  name: Example CTD Processing Pipeline
  description: A pipeline for processing CTD data
  visualisation: false

steps:
  - name: Load OG1
    parameters:
      file_path: PLACEHOLDER
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:
        impossible date qc: {}
        impossible location qc: {}
        position on land qc: {}
    diagnostics: false

  - name: Apply QC
    parameters:
      qc_settings:
        impossible range qc:
          variable_ranges:
            PRES:
              3: [-5, -2.4]
              4: [-.inf, -5]
          also_flag:
            PRES: [CNDC, TEMP]
          plot: [PRES]
        stuck value qc:
          variables:
            PRES: 2
          also_flag:
            PRES: [CNDC, TEMP]
          plot: [PRES]
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



  - name: Find Profiles Pitch
    parameters:
      qc_handling_settings:
        # Again we are using QC filtering, this time to remove bad DEPTH data. You may
        # have noticed that we never explicitly tested DEPTH with Apply QC. Instead,
        # it has inherited QC via a combination of its parent variables QC, PRES & LATITUDE.
        flag_filter_settings:
          DEPTH: [3, 4, 9]
      gradient_thresholds: [0, -0.025] # Cutoffs for defining minimum velocity for up & downcasts
      filter_window_sizes: [20s, 20s] # Window sizes for median & mean smoothing
      depth_column: DEPTH
    diagnostics: true

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
      plot_profiles_in_range: [100, 150]
    diagnostics: true

  - name: Derive CTD
    parameters:
      to_derive: [PRAC_SALINITY, ABS_SALINITY, CONS_TEMP, DENSITY]
    diagnostics: true

  - name: Chla Deep Correction
    parameters:
      apply_to: CHLA
      dark_value: null
      depth_threshold: -550
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
    diagnostics: false

  - name: BBP from Beta
    parameters:
      theta: 124
      xfactor: 1.076
    diagnostics: false

  - name: Isolate BBP Spikes
    parameters:
      window_size: 50
      method: median
    diagnostics: false
"""

# Import toolbox
sys.path.append(os.path.abspath(TOOLBOX_PATH))
from pelagos_py.pipeline import Pipeline, _setup_logging


def main():
    """Parses configuration, builds the pipeline, and runs it."""
    config = yaml.safe_load(BASE_CONFIG_YAML)
    
    for step in config["steps"]:
        if step["name"] == "Load OG1":
            step["parameters"]["file_path"] = INPUT_FILE
    
    pipe = Pipeline()
    pipe.global_parameters = config["pipeline"]
    pipe.logger = _setup_logging(out_dir=OUTPUT_DIRECTORY, log_file=None)
    
    pipe.build_steps(config["steps"])
    pipe.run()
    
    print("Pipeline execution complete.")


if __name__ == "__main__":
    main()
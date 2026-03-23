# This file is part of the NOC Autonomy Toolbox.
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

import os
import importlib
import pathlib
import yaml
from .base_step import REGISTERED_STEPS

try:
    from .base_test import REGISTERED_QC
except ImportError:
    REGISTERED_QC = {}

STEP_CLASSES = {}
QC_CLASSES = {}
STEP_DEPENDENCIES = {
    "QC: Salinity": ["Load OG1"],
}

def discover_steps():
    base_dir = pathlib.Path(__file__).parent.resolve()
    custom_dir = base_dir / "custom"
    print(f"[Discovery] Scanning for step modules in {custom_dir}")

    for py_file in custom_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        relative_path = py_file.resolve().relative_to(base_dir)
        module_name = ".".join(("toolbox.steps",) + relative_path.with_suffix("").parts)

        try:
            importlib.import_module(module_name)
        except Exception as e:
            pass

    STEP_CLASSES.update(REGISTERED_STEPS)
    QC_CLASSES.update(REGISTERED_QC)

discover_steps()

def create_step(step_config, context=None):
    if isinstance(step_config, str):
        if not os.path.exists(step_config):
            raise FileNotFoundError(f"Step config file not found: {step_config}")
        with open(step_config, "r") as f:
            step_config = yaml.safe_load(f) or {}
        if "name" not in step_config:
            raise ValueError(f"Invalid step YAML: missing 'name' key -> {step_config}")

    step_name = step_config.get("name")
    if not step_name:
        raise ValueError("Step config missing required 'name' field.")

    step_class = STEP_CLASSES.get(step_name)
    if not step_class:
        raise ValueError(
            f"Step '{step_name}' not recognised or missing @register_step. "
            f"Available: {list(STEP_CLASSES.keys())}"
        )

    parameters = step_config.get("parameters", {}) or {}
    diagnostics = bool(step_config.get("diagnostics", False))
    step = step_class(
        name=step_name,
        parameters=parameters,
        diagnostics=diagnostics,
        context=context,
    )

    if hasattr(step, "_parameters") and hasattr(step, "_reset_parameter_bridge"):
        step._parameters.update(step_config)
        step._reset_parameter_bridge(mirror_keys=["parameters", "diagnostics"])

    return step
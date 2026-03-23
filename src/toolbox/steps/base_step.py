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

from toolbox.utils.config_mirror import ConfigMirrorMixin
import warnings
import logging
import os

warnings.formatwarning = lambda msg, *args, **kwargs: f"{msg}\n"

REGISTERED_STEPS = {}

def register_step(cls):
    step_name = getattr(cls, "step_name", None)
    if step_name is None:
        raise ValueError(
            f"Class {cls.__name__} is missing required 'step_name' attribute."
        )
    REGISTERED_STEPS[step_name] = cls
    return cls

class BaseStep(ConfigMirrorMixin):
    parameter_schema = {}

    def __init__(self, name, parameters=None, diagnostics=False, context=None):
        self.name = name
        self.diagnostics = diagnostics
        self.context = context or {}
        
        self.logger = logging.getLogger(f"toolbox.pipeline.step.{self.name}")

        self.parameters = parameters or {}
        for param_key, param_meta in self.parameter_schema.items():
            if param_key not in self.parameters:
                self.parameters[param_key] = param_meta.get("default")

        self._init_config_mirror()
        
        self._parameters = {
            "name": self.name,
            "parameters": self.parameters,
            "diagnostics": self.diagnostics,
        }
        
        self._reset_parameter_bridge(mirror_keys=["parameters", "diagnostics"])

        for key, value in self.parameters.items():
            setattr(self, key, value)

        super().__init__()
        
    @classmethod
    def get_schema(cls):
        return cls.parameter_schema

    def is_web_mode(self):
        return os.environ.get("AUTONOMY_WEB_MODE") == "1"

    def run(self):
        raise NotImplementedError(f"Step '{self.name}' must implement a run() method.")
        return self.context

    def generate_diagnostics(self):
        pass

    def web_diagnostic_loop(self):
        self.log("Entering web diagnostic mode. Waiting for user input via dashboard...")
        import urllib.request
        import json
        import base64
        from io import BytesIO
        import time
        import matplotlib.pyplot as plt

        api_base = "http://127.0.0.1:8000/api/internal"

        while True:
            fig = self.create_diagnostic_plot()
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            plot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            payload = {
                "step_name": self.name,
                "parameters": self.parameters,
                "plot_b64": plot_b64,
                "generation_id": int(time.time() * 1000)
            }

            req = urllib.request.Request(
                f"{api_base}/pause",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req)

            action_taken = False
            while not action_taken:
                time.sleep(1)
                try:
                    resp = urllib.request.urlopen(f"{api_base}/status")
                    data = json.loads(resp.read().decode("utf-8"))
                    
                    if data.get("status") in ["regenerate", "continue", "cancel"]:
                        ack_req = urllib.request.Request(
                            f"{api_base}/ack",
                            data=b"{}",
                            headers={"Content-Type": "application/json"}
                        )
                        urllib.request.urlopen(ack_req)

                        if data["status"] == "cancel":
                            self.log("Pipeline cancelled by user during diagnostics.")
                            raise InterruptedError("Pipeline cancelled by user.")

                        new_params = data.get("parameters", {})
                        self.update_parameters(**new_params)

                        if data["status"] == "continue":
                            return
                        else:
                            action_taken = True
                except urllib.error.URLError:
                    pass

    def log(self, message):
        self.logger.info("[%s] %s", self.name, message)

    def log_warn(self, message, warning_type=UserWarning):
        self.logger.warning("[%s] %s", self.name, message)
        warnings.warn(f"[{self.name}] WARNING: {message}", warning_type)

    def check_data(self):
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            self.log(f"Data found in context.")

    def update_parameters(self, **kwargs):
        for k, v in kwargs.items():
            self.parameters[k] = v
            setattr(self, k, v)
        self._parameters["parameters"] = self.parameters

    def generate_config(self):
        self._sync_attributes_to_parameters()
        return dict(self._parameters)

    def save_config(self, path: str | None = None):
        import yaml, os
        cfg = self.generate_config()
        if path is None:
            safe_name = self.name.replace(" ", "_").lower()
            path = f"{safe_name}_step.yaml"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return cfg
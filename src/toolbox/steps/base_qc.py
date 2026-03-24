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

"""Base class definition for Quality Control tests."""

import os
import logging
import json
import base64
import time
import urllib.request
from io import BytesIO
import matplotlib.pyplot as plt

REGISTERED_QC = {}
"""Registry of explicitly registered QC test classes."""

# Keeping your original color map logic but as a list for faster indexing
flag_cols = [
    "gray",       # 0: NO_QC
    "blue",       # 1: GOOD
    "lightblue",  # 2: PROB_GOOD
    "orange",     # 3: PROB_BAD
    "red",        # 4: BAD
    "gray",       # 5: VALUE_CHANGED
    "gray",       # 6: NOT_USED
    "gray",       # 7: NOT_USED
    "cyan",       # 8: ESTIMATED
    "black",      # 9: MISSING
]

def register_qc(cls):
    """Decorator to mark QC tests for the ApplyQC step."""
    test_name = getattr(cls, "test_name", None)
    if test_name is None:
        raise ValueError(f"QC test {cls.__name__} missing 'test_name' attribute.")
    REGISTERED_QC[test_name] = cls
    return cls

class BaseTest:
    test_name = None
    parameter_schema = {} # Replaces expected_parameters
    required_variables = []
    qc_outputs = []

    def __init__(self, data, **kwargs):
        # Maintain your deep copy logic for safety
        self.data = data.copy(deep=True) if data is not None else None
        self.logger = logging.getLogger(f"toolbox.qc.{self.test_name}")
        
        self.parameters = {}
        # Fill parameters from schema defaults
        for param_key, param_meta in self.parameter_schema.items():
            self.parameters[param_key] = param_meta.get("default")

        # Overwrite with user-provided kwargs
        for k, v in kwargs.items():
            if k not in self.parameter_schema:
                raise KeyError(f"Unexpected parameter for {self.test_name}: {k}")
            self.parameters[k] = v

        # Set attributes so self.max_speed works in the test code
        for k, v in self.parameters.items():
            setattr(self, k, v)

        self.flags = None

    def is_web_mode(self):
        return os.environ.get("AUTONOMY_WEB_MODE") == "1"

    def update_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.parameter_schema:
                self.parameters[k] = v
                setattr(self, k, v)

    def return_qc(self):
        """Implemented by sub-classes."""
        raise NotImplementedError("QC tests must implement return_qc().")

    def web_diagnostic_loop(self):
        """Shared diagnostic loop for web dashboard."""
        api_base = "http://127.0.0.1:8000/api/internal"

        while True:
            fig = self.create_diagnostic_plot()
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            plot_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            payload = {
                "step_name": f"QC: {self.test_name}",
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
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    
                    if resp_data.get("status") in ["regenerate", "continue", "cancel"]:
                        ack_req = urllib.request.Request(
                            f"{api_base}/ack", 
                            data=b"{}", 
                            headers={"Content-Type": "application/json"}
                        )
                        urllib.request.urlopen(ack_req)

                        if resp_data["status"] == "cancel":
                            raise InterruptedError("Pipeline cancelled by user.")

                        new_params = resp_data.get("parameters", {})
                        self.update_parameters(**new_params)

                        if resp_data["status"] == "continue":
                            return
                        else:
                            action_taken = True
                except Exception:
                    pass
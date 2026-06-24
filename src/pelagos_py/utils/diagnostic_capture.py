# This file is part of pelagos_py.
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

"""Capture step diagnostic figures for the PDF report.

Steps draw their diagnostics with matplotlib and display them with
``plt.show()`` when ``diagnostics`` is enabled. To include those plots in the
report regardless of a step's own ``diagnostics`` setting, the pipeline
force-enables diagnostics and runs each step inside :func:`capture_figures`,
which redirects ``plt.show`` so the figures are written to disk instead of being
displayed. The saved paths are then embedded by the report writer.
"""

import contextlib
import functools
import os

import matplotlib.pyplot as plt


def _save_and_close_open_figures(outdir: str, step_name: str, images: list) -> None:
    """Save every open matplotlib figure to ``outdir`` and close it.

    Each saved path is appended to ``images``. Saving failures are swallowed so
    capturing a diagnostic can never interrupt the pipeline; the figure is always
    closed so figures are neither displayed nor leaked between steps.
    """
    safe = step_name.replace(os.sep, "_").replace(" ", "_")
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = os.path.join(outdir, f"{safe}_{len(images) + 1}.png")
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            images.append(path)
        except Exception:  # noqa: BLE001 - a capture failure must never be fatal
            pass
        finally:
            plt.close(fig)


@contextlib.contextmanager
def capture_figures(outdir: str, step_name: str, images: list, suppress_text: bool = False):
    """Redirect ``plt.show`` so diagnostic figures are saved instead of displayed.

    Within the context, every ``plt.show`` call (and any figures still open when
    the context exits) is written to ``outdir`` and its path recorded in
    ``images``. Figures are closed so nothing is shown interactively and figures
    do not leak between steps.

    Parameters
    ----------
    outdir : str
        Directory the figures are written into.
    step_name : str
        Name of the step producing the figures (used to name the files).
    images : list
        List that captured image paths are appended to, in the order produced.
    suppress_text : bool, optional
        When ``True``, stdout is silenced for the duration of the context. Used
        when diagnostics are force-enabled only to capture figures for the
        report: a step whose diagnostics are textual (``print`` rather than a
        plot, e.g. the dataset summary from Load Data) would otherwise dump that
        text to the console even though the user never enabled diagnostics.
    """
    original_show = plt.show

    def _capture_show(*args, **kwargs):
        _save_and_close_open_figures(outdir, step_name, images)

    plt.show = _capture_show
    try:
        with contextlib.ExitStack() as stack:
            if suppress_text:
                devnull = stack.enter_context(open(os.devnull, "w"))
                stack.enter_context(contextlib.redirect_stdout(devnull))
            yield
    finally:
        plt.show = original_show
        #   Catch any figures a diagnostic left open without calling show().
        _save_and_close_open_figures(outdir, step_name, images)


def make_diagnostics_safe(step) -> None:
    """Wrap a step's diagnostic methods so a failure never aborts the step.

    Used when diagnostics are force-enabled to capture plots for the report: a
    user who did not opt into diagnostics should never have the pipeline fail
    because of a diagnostic-only error. The error is logged as a warning and the
    step's core processing continues.
    """
    for attr in ("generate_diagnostics", "plot_diagnostics"):
        method = getattr(step, attr, None)
        if not callable(method):
            continue

        @functools.wraps(method)
        def safe(*args, _method=method, **kwargs):
            try:
                return _method(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - diagnostics must never be fatal
                step.logger.warning(
                    "[%s] Diagnostic generation for the report failed: %s",
                    step.name,
                    exc,
                )
                return None

        setattr(step, attr, safe)

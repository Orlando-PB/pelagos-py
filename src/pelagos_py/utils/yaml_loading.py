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

"""Shared YAML loading for config files.

PyYAML implements the YAML 1.1 float rule, whose resolver only recognises
scientific notation that carries *both* a decimal point and a signed exponent
(``3.0e-2``, ``1.5e+3``). Common forms such as ``3e-2``, ``1e3``, ``-2e-2`` and
``3E-2`` are therefore loaded as **strings**, which then fail a step/QC
parameter's ``float`` type check and halt the pipeline.

This module provides a :func:`safe_load` that behaves like ``yaml.safe_load`` but
with a float resolver matching the YAML 1.2 / JSON intent, so integer-mantissa
scientific notation is parsed as a ``float``. Use it for every config read so
numeric values resolve consistently regardless of how they are written.
"""

import re

import yaml


class PelagosSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` with a float resolver that accepts ``3e-2``-style numbers."""


# Mirrors PyYAML's default float resolver but adds an integer-mantissa exponent
# branch (``[0-9]+[eE][+-]?[0-9]+``) so e.g. ``3e-2`` and ``1e3`` resolve to float.
_FLOAT_RE = re.compile(
    r"""^(?:
        [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
       |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
       |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
       |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
       |[-+]?\.(?:inf|Inf|INF)
       |\.(?:nan|NaN|NAN))$""",
    re.X,
)

PelagosSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", _FLOAT_RE, list("-+0123456789.")
)


def safe_load(stream):
    """Parse YAML like ``yaml.safe_load``, resolving ``3e-2``-style floats.

    Parameters
    ----------
    stream : str or file-like
        YAML text or an open file handle.

    Returns
    -------
    The parsed Python object (``None`` for an empty document).
    """
    return yaml.load(stream, Loader=PelagosSafeLoader)

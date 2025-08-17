# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical metrics for tensor maps and images."""

from __future__ import annotations
from skais_mapper.profile import _sanitize_ndim, radial_histogram
from skais_mapper._compat import TORCH_AVAILABLE
from typing import Literal, TYPE_CHECKING

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
else:
    from skais_mapper import _torch_stub as __stub  # noqa
    from skais_mapper._torch_stub import *  # noqa: F401,F403

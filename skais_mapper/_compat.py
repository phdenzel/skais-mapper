# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility checks for optional dependencies."""

import importlib.util
from typing import Final

def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return importlib.util.find_spec("torch") is not None

class OptionalDependencyNotAvailable(ImportError):
    pass

TORCH_AVAILABLE: Final = is_torch_available()

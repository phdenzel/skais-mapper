# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test configuration."""

import pytest
import matplotlib


@pytest.fixture(scope="session", autouse=True)
def show_plots(request) -> bool:
    """Check if the user wants to display plots."""
    verbose = request.config.getoption("verbose")
    return verbose >= 1

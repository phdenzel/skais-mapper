# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configuration and runtime utility functions."""

import uuid
import skais_mapper


__all__ = ["get_run_id", "set_run_id"]


def get_run_id(
    length: int = 8,
) -> str:
    """Fetch a run-specific identifier."""
    return str(skais_mapper.RUN_UID).replace("-", "")[:length]


def set_run_id(run_id: uuid.UUID | str | None = None):
    """Set the run-specific identifier."""
    if run_id is None:
        run_id = uuid.uuid4()
    if not isinstance(run_id, uuid.UUID):
        run_id = uuid.UUID(run_id)
    skais_mapper.RUN_UID = run_id

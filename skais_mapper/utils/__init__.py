# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""All utility functions can be imported from here."""

from chuchichaestli.data.cache import nbytes
from skais_mapper.utils.helper import current_time, compress_encode, extract_decode, alias_kw
from skais_mapper.utils.config import get_run_id, set_run_id
from skais_mapper.utils.primes import next_prime
from skais_mapper.utils.colors import SkaisColors, SkaisColorMaps


__all__ = [
    "nbytes",
    "current_time",
    "compress_encode",
    "extract_decode",
    "alias_kw",
    "get_run_id",
    "set_run_id",
    "next_prime",
    "SkaisColors",
    "SkaisColorMaps",
]

# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generic helper functions and other stuff."""

import datetime
import base64
import lzma
from collections.abc import Callable
import functools


def current_time(
    date_only: bool = True, as_str: bool = True, no_delim: bool = True
) -> datetime.datetime | str:
    """Get current time as string.

    Args:
        date_only: If True, only the date is returned.
        as_str: If True, a string is returned instead of a datetime object.
        no_delim: If a string is returned, remove the standard delimiters.
    """
    t = datetime.datetime.now()
    if date_only:
        t = t.date()
    if as_str:
        t = str(t)
        if no_delim:
            t = t.replace("-", "")
    return t


def compress_encode(string: str) -> str:
    """Compress and encode a string for shorter representations.

    Args:
        string: String to be encoded.
    """
    compressed_data = lzma.compress(string.encode("utf-8"))
    encoded_data = base64.b64encode(compressed_data)
    return encoded_data.decode("utf-8")


def extract_decode(string: str) -> str:
    """Decompress and decode a short representation string.

    Args:
        string: Compressed base64 string to be decoded.
    """
    compressed_data = base64.b64decode(string)
    original_data = lzma.decompress(compressed_data)
    return original_data.decode("utf-8")


def alias_kw(key: str, alias: str) -> Callable:
    """Decorator for aliasing a keyword argument in a function.

    Args:
        key: Name of keyword argument in function to alias
        alias: Alias that can be used for this keyword argument
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            alias_value = kwargs.get(alias)
            if alias_value:
                kwargs[key] = alias_value
            if alias in kwargs:
                del kwargs[alias]
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

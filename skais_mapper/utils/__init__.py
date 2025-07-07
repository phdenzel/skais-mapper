"""
skais_mapper.utils module

@author: phdenzel
"""

from chuchichaestli.data.cache import nbytes
from skais_mapper.utils.helper import (
    current_time,
    compress_encode,
    extract_decode,
)
from skais_mapper.utils.config import (
    get_run_id,
    set_run_id,
    save_config,
    print_config,
    load_config,
)
from skais_mapper.utils.primes import next_prime
from skais_mapper.utils.colors import SkaisColors, SkaisColorMaps


__all__ = [
    "nbytes",
    "current_time",
    "compress_encode",
    "extract_decode",
    "get_run_id",
    "set_run_id",
    "save_config",
    "print_config",
    "load_config",
    "next_prime",
    "SkaisColors",
    "SkaisColorMaps",
]

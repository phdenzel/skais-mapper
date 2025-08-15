from skais_mapper._compat import OptionalDependencyNotAvailable

_MSG = (
    "The 'skais_mapper.metrics' module requires the optional dependency 'torch'.\n"
    "Install the package with: pip install 'skais-mapper[nn]'"
)

def __getattr__(name: str):
    """Raise ImportError."""
    raise OptionalDependencyNotAvailable(_MSG)

def __dir__():
    """Make dir() somewhat informative without importing torch."""
    return []

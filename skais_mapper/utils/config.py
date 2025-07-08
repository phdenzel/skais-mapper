"""
skais_mapper.utils.config module: Configuration and runtime utility functions

@author: phdenzel
"""

import uuid
import json
import warnings
from pathlib import Path
from skais_mapper.utils import compress_encode, extract_decode
import skais_mapper


__all__ = ["get_run_id", "set_run_id", "save_config", "print_config", "load_config"]


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


def save_config(
    config: dict,
    path: str | Path,
    exclude_git_state: bool = False,
    include_git_diff: bool = False,
    **kwargs,
) -> Path:
    """Save the configuration in a JSON file.

    Args:
        config: Dictionary containing the entire run configuration.
        path: Path to the checkpoint directory or file.
        exclude_git_state: If False, the git state is added to the config.
        include_git_diff: If True, the git diff is added to the config
          (as shorter encoded string for readability).
        kwargs: Additional keyword arguments for compatibility.
    """
    path = Path(path)
    if path.is_dir():
        path = path / f"{get_run_id()}.json"
    elif path.suffix != ".json":
        path = path.with_suffix(".json")
    config["run_uid"] = str(skais_mapper.RUN_UID)
    if not exclude_git_state:
        config["git_state"] = skais_mapper.GIT_STATE
    if include_git_diff:
        config["git_diff"] = [compress_encode(d) for d in skais_mapper.GIT_DIFF]
    else:
        config["git_is_dirty"] = True
    with path.open("w") as f:
        json_dict = {}
        for k, v in config.items():
            if isinstance(v, Path):
                json_dict[k] = str(v)
            elif callable(v):
                continue
            else:
                json_dict[k] = v
        json.dump(json_dict, f, indent=4)
    return path


def print_config(
    config: dict | None = None,
    exclude_git_state: bool = False,
    include_git_diff: bool = False,
    **kwargs,
):
    """Print the configuration (instead of saving as JSON file).

    Args:
        config: Dictionary containing the entire run configuration.
        exclude_git_state: If False, the git state is added to the config.
        include_git_diff: If True, the git diff is added to the config
          (as shorter encoded string for readability).
        kwargs: Additional keyword arguments for compatibility.
    """
    if config is None:
        config = skais_mapper.parse_args()
    config["run_uid"] = str(skais_mapper.RUN_UID)
    if not exclude_git_state:
        config["git_state"] = skais_mapper.GIT_STATE
    if include_git_diff:
        config["git_diff"] = [compress_encode(d) for d in skais_mapper.GIT_DIFF]
    else:
        config["git_is_dirty"] = len(skais_mapper.GIT_DIFF) > 1
    json_dict = {}
    for k, v in config.items():
        if isinstance(v, Path):
            json_dict[k] = str(v)
        elif callable(v):
            continue
        else:
            json_dict[k] = v
    print("Configuration:")
    print(json.dumps(json_dict, indent=4))


def load_config(filename: str | Path, write_globals: bool = True, **kwargs):
    """Load the configuration from a JSON file.

    Args:
        filename: Path to the configuration file
        write_globals: Overwrite global runtime variables
        kwargs: Additional keyword arguments for compatibility.
    """
    with Path(filename).open("r") as f:
        json_dict = json.load(f)
        if write_globals and "run_uid" in json_dict:
            skais_mapper.RUN_UID = uuid.UUID(json_dict["run_uid"])
        if write_globals and "git_state" in json_dict:
            skais_mapper.GIT_STATE = json_dict["git_state"]
        if write_globals and "git_diff" in json_dict:
            skais_mapper.GIT_DIFF = [extract_decode(d) for d in json_dict["git_diff"]]
        elif "git_is_dirty" in json_dict:
            warnings.warn(
                "The input configuration is dirty and may not be loaded as before.\n"
                "Try inspecting the encoded `git_diff` (if present)..."
            )
        for k in json_dict:
            if (k.endswith("_dir") or k == "root") and json_dict[k] is not None:
                json_dict[k] = Path(json_dict[k])
    return json_dict

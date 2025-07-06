"""
skais_mapper.utils.config module: Configuration and runtime utility functions

@author: phdenzel
"""

import uuid
import json
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from skais_mapper.utils import compress_encode, extract_decode
import skais_mapper


__all__ = ["get_run_id", "set_run_id", "parse_args", "save_config", "print_config", "load_config"]


def get_run_id(length: int = 8) -> str:
    """Fetch a run-specific identifier."""
    return str(skais_mapper.RUN_UID).replace("-", "")[:length]


def set_run_id(run_id: uuid.UUID | None = None):
    """Set the run-specific identifier."""
    if run_id is None:
        run_id = uuid.uuid4()
    skais_mapper.RUN_UID = run_id


def parse_args(**kwargs) -> dict:
    """Parse arguments."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument("--id", type=str, default=get_run_id(), help="Name/UUID of the job.")
    # General arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Set verbosity.")

    # Convert to dict
    args, _ = parser.parse_known_args()
    configs = vars(args)
    return configs


def save_config(
    config: dict,
    path: str | Path,
    include_git_state: bool = False,
) -> Path:
    """Save the configuration in a JSON file.

    Args:
        config: Dictionary containing the entire run configuration.
        path: Path to the checkpoint directory or file.
        include_git_state: If True, the git state is added to the config.
    """
    path = Path(path)
    if path.is_dir():
        path = path / f"{get_run_id()}.json"
    elif path.suffix != ".json":
        path = path.with_suffix(".json")
    config["run_uid"] = str(skais_mapper.RUN_UID)
    if include_git_state:
        config["git_state"] = skais_mapper.GIT_STATE
        config["git_diff"] = [compress_encode(d) for d in skais_mapper.GIT_DIFF]
    with path.open("w") as f:
        json_dict = {}
        for k, v in config.items():
            if isinstance(v, Path):
                json_dict[k] = str(v)
            else:
                json_dict[k] = v
        json.dump(json_dict, f, indent=4)
    return path


def print_config(
    config: dict | None = None,
    include_git_state: bool = False,
):
    """Print the configuration (instead of saving as JSON file).

    Args:
        config: Dictionary containing the entire run configuration.
        include_git_state: If True, the git state is added to the config.
    """
    if config is None:
        config = parse_args()
    config["run_uid"] = str(skais_mapper.RUN_UID)
    if include_git_state:
        config["git_state"] = skais_mapper.GIT_STATE
        config["git_diff"] = [compress_encode(d) for d in skais_mapper.GIT_DIFF]
    json_dict = {}
    for k, v in config.items():
        if isinstance(v, Path):
            json_dict[k] = str(v)
        else:
            json_dict[k] = v
    print(json.dumps(json_dict, indent=4))


def load_config(
    filename: str | Path,
    write_globals: bool = True,
):
    """Load the configuration from a JSON file.

    Args:
      filename (str | Path): Path to the configuration file
      write_globals (bool): Overwrite global runtime variables
    """
    with Path(filename).open("r") as f:
        json_dict = json.load(f)
        if write_globals and "run_uid" in json_dict:
            skais_mapper.RUN_UID = uuid.UUID(json_dict["run_uid"])
        if write_globals and "git_state" in json_dict:
            skais_mapper.GIT_STATE = json_dict["git_state"]
        if write_globals and "git_diff" in json_dict:
            skais_mapper.GIT_DIFF = [extract_decode(d) for d in json_dict["git_diff"]]
        for k in json_dict:
            if k.endswith("_dir") or k == "root":
                json_dict[k] = Path(json_dict[k])
    return json_dict

# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""The main module: sets global variables and registers OmegaConf resolvers."""

import uuid
import git
from omegaconf import OmegaConf
import skais_mapper.__about__
import skais_mapper.rotations
import skais_mapper.cosmology
import skais_mapper.raytrace
import skais_mapper.illustris
import skais_mapper.simobjects
import skais_mapper.plotting
import skais_mapper.utils
import skais_mapper.configure
import skais_mapper.generate


RUN_UID = uuid.uuid4()
repository = git.Repo(search_parent_directories=True)
GIT_STATE = repository.head.object.hexsha
GIT_DIFF = [
    str(diff)
    for diff in
    repository.index.diff(None, create_patch=True)
    + repository.index.diff("HEAD", create_patch=True)
]

if OmegaConf.has_resolver("get_run_id"):
    OmegaConf.clear_resolver("get_run_id")
OmegaConf.register_new_resolver("get_run_id", skais_mapper.utils.get_run_id)

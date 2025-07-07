"""
skais_mapper module

@author: phdenzel
"""

import uuid
import git
import skais_mapper.rotations
import skais_mapper.cosmology
import skais_mapper.raytrace
import skais_mapper.illustris
import skais_mapper.simobjects
import skais_mapper.utils
from skais_mapper.__main__ import parse_args


RUN_UID = uuid.uuid4()
repository = git.Repo(search_parent_directories=True)
GIT_STATE = repository.head.object.hexsha
GIT_DIFF = [
    str(diff)
    for diff in
    repository.index.diff(None, create_patch=True)
    + repository.index.diff("HEAD", create_patch=True)
]

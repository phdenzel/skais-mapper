"""
skais_mapper.utils.helper module: Helper functions and other stuff

@author: phdenzel
"""

import uuid


def uuid_wlen(length: int = 8):
    """
    UUID of a given length

    Args:
        length: length of the generated UUID
    """
    uuidstr = str(uuid.uuid1()).replace("-", "")[:length]
    while len(uuidstr) < length:
        uuidstr += str(uuid.uuid1()).replace("-", "")[: (length - len(uuidstr))]
    return uuidstr

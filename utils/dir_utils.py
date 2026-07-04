import os
from natsort import natsorted
from glob import glob


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_last_path(path, session):
    x = natsorted(glob(os.path.join(path, '*%s' % session)))[-1]
    return x

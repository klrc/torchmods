import os
import tarfile
import re
from .hash import md5


def _safeopen(path):
    """ Check if the path exists, if not, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def compress(src, save_root, tar_name=None, ignore=None):
    """ Generating a compressed package using tar.bz2 (for md5 timestep issue).
        If src is a directory, the entire source inside the root would be compressed.
        The tar.bz2 file is named with md5 hash by default.
        param ignore: a regex pattern to ignore some files. (works only in directory mode)
    """
    tmp_path = _safeopen(save_root) + '/_tmp_compress.tar.bz2'
    with tarfile.open(tmp_path, 'w:bz2') as tar:
        if os.path.isdir(src):
            for dir_path, _, names in os.walk(src):
                for name in names:
                    full_path = os.path.join(dir_path, name)
                    if ignore is None or not re.match(re.compile(ignore), full_path):
                        print(f'add: {full_path}')
                        tar.add(full_path, os.path.join(dir_path.replace(src, ''), name))
                    else:
                        print(f'ignore: {full_path}')
        else:
            tar.add(src, os.path.split(src)[1])
    if tar_name is None:
        tar_name = str(md5(tmp_path))
    os.rename(tmp_path, tmp_path.replace(os.path.split(tmp_path)[1], tar_name + '.tar.bz2'))
    return tar_name

import os


def find_path(paths):
    for path in paths:
        try:
            assert type(path) is str
            if os.path.exists(path):
                return path
        except Exception as e:
            raise e
    return None

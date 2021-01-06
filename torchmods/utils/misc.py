from functools import wraps


def raise_err(func):
    @wraps(func)
    def core(*args, **kwargs):
        out, err = func(*args, **kwargs)
        if len(err) > 0:
            raise err[0]
    return core


def try_callback(func):
    @wraps(func)
    def core(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return [], []
        except Exception as e:
            return [], [e]
    return core

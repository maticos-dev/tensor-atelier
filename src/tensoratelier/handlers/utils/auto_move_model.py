import inspect
from functools import wraps


def auto_move_model(func):

    @wraps(func)
    def wrapped(self, **kwargs):

        self._accelerator_handler._move_model(kwargs["model"])
        return func(self, **kwargs)

    return wrapped

import inspect
from functools import wraps


def auto_move_model(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        model = list(bound.arguments.keys())["model"]

        self._accelerator_handler._move_model(model)
        return func(self, model, *args, **kwargs)

    return wrapped

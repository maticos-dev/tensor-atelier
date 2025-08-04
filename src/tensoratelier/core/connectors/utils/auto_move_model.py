from functools import wraps


def auto_move_model(func):
    @wraps(func)
    def wrapped(self, model, *args, **kwargs):
        self._accelerator_handler._move_model(model)
        return func(self, model, *args, **kwargs)

    return wrapped

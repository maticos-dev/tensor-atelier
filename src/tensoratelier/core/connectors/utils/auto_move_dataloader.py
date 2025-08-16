import inspect
from functools import wraps

from tensoratelier.core import AtelierDataLoader


def auto_move_dataloader(func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        args = list(bound.arguments.keys())

        # self is trainer instance
        converted_dataloader = AtelierDataLoader(
            self, args["dataloader"], args["train_val_split"]
        )

        return func(self, *args, **kwargs)

    return wrapped

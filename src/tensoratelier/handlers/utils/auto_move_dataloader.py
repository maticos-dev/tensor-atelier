from functools import wraps

from tensoratelier.core import AtelierDataLoader


def auto_move_dataloader(func):

    @wraps(func)
    def wrapped(self, **kwargs):

        # self is trainer instance
        converted_dataloader = AtelierDataLoader(
            kwargs["dataloader"],
            self,
            kwargs["train_val_split"],
            kwargs["device"]
        )

        kwargs.update({"dataloader": converted_dataloader})

        return func(self, **kwargs)

    return wrapped

from functools import wraps


def auto_move_dataloader(func):

    @wraps(func)
    def wrapped(self, **kwargs):

        # self is trainer instance
        # Import here to avoid circular imports
        from tensoratelier.core import AtelierDataLoader
        
        converted_dataloader = AtelierDataLoader(
            self,
            kwargs["dataloader"],
            kwargs["train_val_split"],
            kwargs["device"]
        )

        kwargs.update({"dataloader": converted_dataloader})

        return func(self, **kwargs)

    return wrapped

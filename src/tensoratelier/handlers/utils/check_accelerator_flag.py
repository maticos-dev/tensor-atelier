from functools import wraps
from tensoratelier.accelerators.registry import ACCELERATOR_REGISTRY


def check_accelerator_flag(func):

    @wraps(func)
    def wrapped(self, **kwargs):

        accelerator_cls = ACCELERATOR_REGISTRY._get(kwargs["accelerator"])

        if not accelerator_cls:
            raise ValueError(
                "You selected an invalid accelerator name: 'accelerator="
                f"{kwargs["accelerator"]!r}'. Available names are: auto,"
                f"{', '.join(ACCELERATOR_REGISTRY._registry.keys())}."
            )

        return func(self, **kwargs)

    return wrapped

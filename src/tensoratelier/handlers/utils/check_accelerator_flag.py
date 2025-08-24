from functools import wraps
from tensoratelier.accelerators.registry import ACCELERATOR_REGISTRY


def check_accelerator_flag(func):

    @wraps(func)
    def wrapped(self, **kwargs):

        accelerator_cls = ACCELERATOR_REGISTRY._get(kwargs["accelerator_flag"])

        if not accelerator_cls:
            raise ValueError(
                f"You selected an invalid accelerator name: 'accelerator_flag={kwargs['accelerator_flag']}'. "
                f"Available names are: auto, {', '.join(ACCELERATOR_REGISTRY._registry.keys())}."
            )

        return func(self, **kwargs)

    return wrapped

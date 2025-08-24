import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, cast

_T = TypeVar("_T", bound=Callable[..., Any])


def _wrap_args(fn: _T) -> Any:
    @wraps(fn)
    def insert_env_defaults(self: Any, *args: Any, **kwargs: Any) -> Any:
        cls = self.__class__

        if args:
            cls_arg_names = inspect.signature(cls).parameters

            # zip excludes the cls_arg_names that are part of kwargs
            kwargs.update(dict(zip(cls_arg_names, args)))

        return fn(self, **kwargs)

    return cast(_T, insert_env_defaults)

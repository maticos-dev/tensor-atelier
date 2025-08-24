import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, cast

_T = TypeVar("_T", bound=Callable[..., Any])


def _wrap_args(fn: _T) -> Any:
    @wraps(fn)
    def insert_env_defaults(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Get the function signature, not the class signature
        fn_sig = inspect.signature(fn)
        fn_params = list(fn_sig.parameters.keys())
        
        # Skip 'self' parameter
        if fn_params and fn_params[0] == 'self':
            fn_params = fn_params[1:]

        if args:
            # zip excludes the fn_params that are part of kwargs
            kwargs.update(dict(zip(fn_params, args)))

        return fn(self, **kwargs)

    return cast(_T, insert_env_defaults)

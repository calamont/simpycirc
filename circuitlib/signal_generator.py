import inspect
import numpy as np
from scipy import signal
from functools import wraps


def partial(func, **kwargs):

    argspec = inspect.getfullargspec(func)
    func_kwargs = set(argspec.kwonlyargs)
    try:
        supplied_kwargs = set({**argspec.kwonlydefaults, **kwargs}.keys())
    except TypeError:
        pass  # signal function has no keyword-only arguments
    else:
        missing_kwargs = func_kwargs.difference(supplied_kwargs)
        if missing_kwargs:
            raise ValueError(f"{func.__name__} missing values for: {missing_kwargs}.")

    @wraps(func)
    def wrapper(time):
        return func(time, **kwargs)

    wrapper.__name__ = func.__name__
    return wrapper


# Each function needs to have the exact same template
# def template_signal(time, value, period):
#     pass


def DC(time, value):
    """Constant DC value."""
    return np.full_like(time, value)


def sin(time, value, *, period, x_offset=0, y_offset=0):
    return value * np.sin(((time - x_offset) * 2 * np.pi) / period) + y_offset


def sawtooth(time, value, *, period, mod=0, x_offset=0, y_offset=0):
    """Sawtooth wave signal.

    Args:
        time:
        value (float):
        period (float):
        offset (float):
        mod (float, optional):
    """
    return (
        value * signal.sawtooth(((time - x_offset) * 2 * np.pi) / period, mod)
        + y_offset
    )


def square(time, value, *, period, mod=0.5, x_offset=0, y_offset=0):
    return (
        value * signal.square(((time - x_offset) * 2 * np.pi) / period, mod) + y_offset
    )


def template_signal(time, value, *, period):
    # All values after the `value` parameter are keyword-only arguments
    pass

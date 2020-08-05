"""Signal generator functions."""
import numpy as np
from scipy import signal
from functools import wraps


def partial(func, **kwargs):

    req_kwargs = list(func.__code__.co_varnames)
    req_kwargs.remove("time")
    missing_kwargs = set(req_kwargs).difference(set(kwargs.keys()))
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


def DC(time, value, period):
    """Constant DC value."""
    return np.full_like(time, value)


def sin(time, value, period, x_offset, y_offset, mod):
    return np.sin(((time - x_offset) * 2 * np.pi) / period) + y_offset


def sawtooth(time, value, period, offset, mod):
    """Sawtooth wave signal.

    Args:
        time:
        value (float):
        period (float):
        offset (float):
        mod (float, optional):
    """
    return signal.sawtooth((time * 2 * np.pi) / period, mod) + offset


def square(time, value, period, offset, mod):
    return signal.square((time * 2 * np.pi) / period, mod) + offset


def template_signal(time, value, period):
    pass

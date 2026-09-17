from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast
import torch
import math

T = TypeVar("T")
FuncType = Callable[..., Any]


def exists(val: Optional[Any]) -> bool:
    """
    Check if a value exists (is not None).

    Args:
        val: The value to check for existence

    Returns:
        bool: True if the value is not None, False otherwise
    """
    return val is not None


def default(val: Optional[T], d: Union[T, Callable[[], T]]) -> T:
    """
    Return val if it exists, otherwise return default d.

    Args:
        val: The value to check
        d: The default value or function to return if val is None

    Returns:
        The value val if it exists, otherwise the default value
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def once(fn: FuncType) -> FuncType:
    """
    Decorator to ensure a function is only called once.

    Args:
        fn: The function to wrap

    Returns:
        Wrapped function that will only execute on the first call
    """
    called = False

    @wraps(fn)
    def inner(x: Any) -> Optional[Any]:
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return cast(FuncType, inner)


print_once = once(print)


def cast_tuple(t: Any, length: int = 1) -> Tuple[Any, ...]:
    """
    Cast input to a tuple of specified length.

    Args:
        t: The input to cast to a tuple
        length: The desired length of the output tuple

    Returns:
        A tuple of the specified length containing the input value(s)
    """
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer: int, denom: int) -> bool:
    """
    Check if number is divisible by another.

    Args:
        numer: The numerator (number to be divided)
        denom: The denominator (number to divide by)

    Returns:
        bool: True if numer is divisible by denom, False otherwise
    """
    return (numer % denom) == 0


def log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    Safe logarithm with minimum clamping to avoid numerical issues.

    Args:
        t: The input tensor
        eps: Small epsilon value for numerical stability

    Returns:
        Tensor containing the log of the input with minimum clamping
    """
    return torch.log(t.clamp(min=eps))

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def rescaling(scaling_factor: int, rescale: int, y: int) -> int:
    """Applies integer rescaling to a value based on the given scaling factor.

    Args:
        scaling_factor (int): The divisor to apply when rescaling.
            Must be provided if `rescale` is True.
        rescale (int): Whether to apply rescaling. (0 -> no rescaling, 1 -> rescaling).
        Y (int): The value to be rescaled.

    Raises:
        NotImplementedError: If `rescale` is 1 but `scaling_factor` is not provided.
        NotImplementedError: If `rescale` is not 0 or 1.

    Returns:
        int: The rescaled value if `rescale` is True, otherwise the original value.
    """
    if rescale == 1:
        if scaling_factor is None:
            msg = "scaling_factor must be specified when rescale=1"
            raise ValueError(msg)
        return y // scaling_factor
    if rescale == 0:
        return y
    msg = f"Rescale must be 0 or 1, got {rescale}"
    raise ValueError(msg)


def parse_attr(attr: str, default: T) -> T:
    """Parses an attribute list of strings into a list of integers.

    Args:
        attr (str): Attribute to parse. If a string, it must be
                    comma-separated integers (e.g., "1, 2, 3").
                    If None, returns `default`.
        default (T): Default value to return if `attr` is None.

    Raises:
        ValueError: If `attr` is a string but cannot be parsed into integers.

    Returns:
        T: Parsed list of integers if attr is provided, otherwise the default value.
    """
    if attr is None:
        return default
    try:
        return [int(x.strip()) for x in attr.split(",")]
    except ValueError as e:
        msg = f"Invalid attribute format: {attr}"
        raise ValueError(msg) from e

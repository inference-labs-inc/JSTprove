

from typing import Any, List


def rescaling(scaling_factor: int, rescale: int, Y: int) -> int:
    """Applies integer rescaling to a value based on the given scaling factor.

    Args:
        scaling_factor (int): The divisor to apply when rescaling. Must be provided if `rescale` is True.
        rescale (int): Whether to apply rescaling. (0 -> no rescaling, 1 -> rescaling).
        Y (int): The value to be rescaled.

    Raises:
        NotImplementedError: If `rescale` is 1 but `scaling_factor` is not provided.
        NotImplementedError: If `rescale` is not 0 or 1.

    Returns:
        int: The rescaled value if `rescale` is True, otherwise the original value.
    """
    if rescale == 1:
        if scaling_factor == None:
            raise NotImplementedError("scaling factor must be specified")
        return (Y // scaling_factor)
    elif rescale == 0:
        return Y
    else:
        raise NotImplementedError("Rescale must be 0 or 1")
    
def parse_attr(attr: str, default: Any) -> List[Any]:
    """Parses an attribute list of strings into a list of integers.

    Args:
        attr (str): Attribute to parse. If a string, it must be 
                    comma-separated integers (e.g., "1, 2, 3").
                    If None, returns `default`.
        default (Any): Default value to return if `attr` is None.

    Raises:
        ValueError: If `attr` is a string but cannot be parsed into integers.

    Returns:
        List[Any]: Parsed list of integers.
    """    
    if attr is None:
        return default
    try:
        return [int(x.strip()) for x in attr.split(",")]
    except ValueError as e:
        raise ValueError(f"Invalid attribute format: {attr}") from e
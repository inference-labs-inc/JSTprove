from __future__ import annotations

from python.core.circuits.base import Circuit
from python.core.utils.general_layer_functions import GeneralLayerFunctions


class ZKModelBase(GeneralLayerFunctions, Circuit):
    """
    Abstract base class for Zero-Knowledge (ZK) ML models.

    This class provides a standard interface for ZK circuit ML models.
    Instantiates Circuit and GeneralLayerFunctions.

    Subclasses must implement the constructor to define the model's
    architecture, layers, and circuit details.
    """

    def __init__(self: ZKModelBase) -> None:
        """Initialize the ZK model. Must be overridden by subclasses

        Raises:
            NotImplementedError: If called on the base class directly.
        """
        msg = "Must implement __init__"
        raise NotImplementedError(msg)

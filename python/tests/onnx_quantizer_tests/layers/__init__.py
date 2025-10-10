from .base import BaseLayerConfigProvider, LayerTestConfig
from .factory import TestLayerFactory

# Auto-discover and make available all config providers
# This triggers the discovery process when the package is imported
_all_configs = TestLayerFactory.get_layer_configs()

# Export the factory and base classes
__all__ = [
    "BaseLayerConfigProvider",
    "LayerTestConfig",
    "TestLayerFactory",
]

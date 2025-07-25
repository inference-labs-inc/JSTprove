import os
import importlib
import inspect
from typing import Dict, List
from python.testing.core.tests.onnx_quantizer_tests.layers.base import LayerTestConfig, BaseLayerConfigProvider



class TestLayerFactory:
    """Factory for creating test configurations for different layer types"""
    
    _providers = {}
    _initialized = False
    
    @classmethod
    def _discover_providers(cls):
        """Automatically discover all BaseLayerConfigProvider subclasses"""
        if cls._initialized:
            return
            
        # Get the directory where this module is located
        current_dir = os.path.dirname(__file__)
        
        # Find all Python files in the directory (except __init__.py, base.py, factory.py)
        config_files = [
            f[:-3] for f in os.listdir(current_dir) 
            if f.endswith('_config.py') and f != '__init__.py'
        ]
        
        # Import each config module and find BaseLayerConfigProvider subclasses
        for module_name in config_files:
            try:
                module = importlib.import_module(f'.{module_name}', package=__package__)
                
                # Find all classes in the module that are subclasses of BaseLayerConfigProvider
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseLayerConfigProvider) and 
                        obj is not BaseLayerConfigProvider):
                        
                        # Instantiate the provider and register it
                        provider_instance = obj()
                        cls._providers[provider_instance.layer_name] = provider_instance
                        
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
        
        cls._initialized = True
    
    @classmethod
    def get_layer_configs(cls) -> Dict[str, LayerTestConfig]:
        """Get test configurations for all supported layers"""
        cls._discover_providers()  # Ensure providers are loaded
        return {
            name: provider.get_config() 
            for name, provider in cls._providers.items()
        }
    
    @classmethod
    def get_layer_config(cls, layer_name: str) -> LayerTestConfig:
        """Get test configuration for a specific layer"""
        cls._discover_providers()  # Ensure providers are loaded
        if layer_name not in cls._providers:
            raise ValueError(f"No config provider found for layer: {layer_name}")
        return cls._providers[layer_name].get_config()
    
    @classmethod
    def get_available_layers(cls) -> List[str]:
        """Get list of all available layer types"""
        cls._discover_providers()  # Ensure providers are loaded
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, provider: BaseLayerConfigProvider):
        """Register a new config provider"""
        cls._providers[provider.layer_name] = provider
import os
import importlib
import inspect
from typing import Dict, List, Optional, Set, Tuple
from .base import LayerTestConfig, BaseLayerConfigProvider, TestSpec, SpecType


class TestLayerFactory:
    """Enhanced factory for creating test configurations for different layer types"""
    
    _providers = {}
    _initialized = False
    
    @classmethod
    def _discover_providers(cls):
        """Automatically discover all BaseLayerConfigProvider subclasses"""
        if cls._initialized:
            return
            
        current_dir = os.path.dirname(__file__)
        config_files = [
            f[:-3] for f in os.listdir(current_dir) 
            if f.endswith('_config.py') and f != '__init__.py'
        ]
        
        for module_name in config_files:
            try:
                module = importlib.import_module(f'.{module_name}', package=__package__)
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseLayerConfigProvider) and 
                        obj is not BaseLayerConfigProvider):
                        
                        provider_instance = obj()
                        cls._providers[provider_instance.layer_name] = provider_instance
                        
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
        
        cls._initialized = True
    
    # Existing methods (keep your current implementation)
    @classmethod
    def get_layer_configs(cls) -> Dict[str, LayerTestConfig]:
        """Get test configurations for all supported layers"""
        cls._discover_providers()
        return {
            name: provider.get_config() 
            for name, provider in cls._providers.items()
        }
    
    @classmethod
    def get_layer_config(cls, layer_name: str) -> LayerTestConfig:
        """Get test configuration for a specific layer"""
        cls._discover_providers()
        if layer_name not in cls._providers:
            raise ValueError(f"No config provider found for layer: {layer_name}")
        return cls._providers[layer_name].get_config()
    
    @classmethod
    def get_available_layers(cls) -> List[str]:
        """Get list of all available layer types"""
        cls._discover_providers()
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, provider: BaseLayerConfigProvider):
        """Register a new config provider"""
        cls._providers[provider.layer_name] = provider
    
    # NEW enhanced methods for test specifications
    @classmethod
    def get_all_test_cases(cls) -> List[Tuple[str, LayerTestConfig, TestSpec]]:
        """Get all test cases as (layer_name, config, test_spec) tuples"""
        cls._discover_providers()
        test_cases = []
        
        for layer_name, provider in cls._providers.items():
            config = provider.get_config()
            test_specs = provider.get_test_specs()
            
            # If no test specs defined, create a basic valid test
            if not test_specs:
                from .base import TestSpec, SpecType
                test_specs = [TestSpec("basic", SpecType.VALID, "Basic test")]
            
            for spec in test_specs:
                test_cases.append((layer_name, config, spec))
        
        return test_cases
    
    @classmethod
    def get_test_cases_by_type(cls, test_type: SpecType) -> List[Tuple[str, LayerTestConfig, TestSpec]]:
        """Get test cases of a specific type"""
        return [(layer, config, spec) for layer, config, spec in cls.get_all_test_cases() 
                if spec.spec_type == test_type]
    
    @classmethod
    def get_test_cases_by_layer(cls, layer_name: str) -> List[Tuple[str, LayerTestConfig, TestSpec]]:
        """Get test cases for a specific layer"""
        return [(layer, config, spec) for layer, config, spec in cls.get_all_test_cases() 
                if layer == layer_name]
    
    @classmethod
    def get_test_cases_by_tag(cls, tag: str) -> List[Tuple[str, LayerTestConfig, TestSpec]]:
        """Get test cases with a specific tag"""
        print("TESTTEST2")
        # print(tag)
        print([(layer, config, spec) for layer, config, spec in cls.get_all_test_cases() 
                if tag in spec.tags])
        if [(layer, config, spec) for layer, config, spec in cls.get_all_test_cases() 
                if tag in spec.tags] != []:
            raise
        return [(layer, config, spec) for layer, config, spec in cls.get_all_test_cases() 
                if tag in spec.tags]
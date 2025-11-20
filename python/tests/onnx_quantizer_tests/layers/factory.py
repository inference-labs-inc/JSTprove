from __future__ import annotations

import importlib
import inspect
import logging
import typing
from pathlib import Path

from .base import BaseLayerConfigProvider, LayerTestConfig, LayerTestSpec, SpecType

logger = logging.getLogger(__name__)


class TestLayerFactory:
    """Enhanced factory for creating test configurations for different layer types"""

    _providers: typing.ClassVar = {}
    _initialized = False

    @classmethod
    def _discover_providers(cls: TestLayerFactory) -> None:
        """Automatically discover all BaseLayerConfigProvider subclasses"""
        if cls._initialized:
            return

        current_dir = Path(__file__).parent
        config_files = [
            f.stem
            for f in Path(current_dir).iterdir()
            if f.is_file() and f.name.endswith("_config.py") and f.name != "__init__.py"
        ]
        logger.debug("Discovered config files: %s", config_files)

        for module_name in config_files:
            try:
                module = importlib.import_module(f".{module_name}", package=__package__)

                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseLayerConfigProvider)
                        and obj is not BaseLayerConfigProvider
                    ):

                        provider_instance = obj()
                        cls._providers[provider_instance.layer_name] = provider_instance

            except ImportError as e:  # noqa: PERF203
                msg = f"Warning: Could not import {module_name}: {e}"
                logger.warning(msg)

        cls._initialized = True

    # Existing methods (keep your current implementation)
    @classmethod
    def get_layer_configs(cls) -> dict[str, LayerTestConfig]:
        """Get test configurations for all supported layers"""
        cls._discover_providers()
        logger.debug("Retrieved layer configs: %s", list(cls._providers.keys()))
        return {
            name: provider.get_config() for name, provider in cls._providers.items()
        }

    @classmethod
    def get_layer_config(cls, layer_name: str) -> LayerTestConfig:
        """Get test configuration for a specific layer"""
        cls._discover_providers()
        if layer_name not in cls._providers:
            msg = f"No config provider found for layer: {layer_name}"
            raise ValueError(msg)
        return cls._providers[layer_name].get_config()

    @classmethod
    def get_available_layers(cls) -> list[str]:
        """Get list of all available layer types"""
        cls._discover_providers()
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, provider: BaseLayerConfigProvider) -> None:
        """Register a new config provider"""
        cls._providers[provider.layer_name] = provider

    # NEW enhanced methods for test specifications
    @classmethod
    def get_all_test_cases(cls) -> list[tuple[str, LayerTestConfig, LayerTestSpec]]:
        """Get all test cases as (layer_name, config, test_spec) tuples"""
        cls._discover_providers()
        test_cases = []

        for layer_name, provider in cls._providers.items():
            config = provider.get_config()
            test_specs = provider.get_test_specs()

            # If no test specs defined, create a basic valid test
            if not test_specs:

                test_specs = [LayerTestSpec("basic", SpecType.VALID, "Basic test")]

            test_cases.extend((layer_name, config, spec) for spec in test_specs)

        return test_cases

    @classmethod
    def get_test_cases_by_type(
        cls,
        test_type: SpecType,
    ) -> list[tuple[str, LayerTestConfig, LayerTestSpec]]:
        """Get test cases of a specific type"""
        return [
            (layer, config, spec)
            for layer, config, spec in cls.get_all_test_cases()
            if spec.spec_type == test_type
        ]

    @classmethod
    def get_test_cases_by_layer(
        cls,
        layer_name: str,
    ) -> list[tuple[str, LayerTestConfig, LayerTestSpec]]:
        """Get test cases for a specific layer"""
        return [
            (layer, config, spec)
            for layer, config, spec in cls.get_all_test_cases()
            if layer == layer_name
        ]

    @classmethod
    def get_test_cases_by_tag(
        cls,
        tag: str,
    ) -> list[tuple[str, LayerTestConfig, LayerTestSpec]]:
        """Get test cases with a specific tag"""
        result = [
            (layer, config, spec)
            for layer, config, spec in cls.get_all_test_cases()
            if tag in spec.tags
        ]
        logger.debug("Found tests %s", result)
        if not result:
            msg = f"No test cases found for tag: {tag}"
            raise ValueError(msg)
        return result

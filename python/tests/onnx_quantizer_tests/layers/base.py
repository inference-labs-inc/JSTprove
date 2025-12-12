from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class SpecType(Enum):
    """Types of test specifications that can be run"""

    VALID = "valid"
    ERROR = "error"
    EDGE_CASE = "edge_case"
    E2E = "e2e"


@dataclass
class LayerTestSpec:
    """Individual test specification that can be applied to a LayerTestConfig"""

    name: str
    spec_type: SpecType
    description: str = ""

    # Overrides for the base config
    attr_overrides: dict[str, Any] = field(default_factory=dict)
    initializer_overrides: dict[str, np.ndarray] = field(default_factory=dict)
    input_overrides: list[str] = field(default_factory=list)
    input_shape_overrides: dict[str, list[int]] = field(default_factory=dict)
    output_shape_overrides: dict[str, list[int]] = field(default_factory=dict)

    # Error test specific
    expected_error: type | None = None
    error_match: str | None = None

    # Custom validation
    custom_validator: Callable | None = None

    # Test metadata
    tags: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    # Omit attributes
    omit_attrs: list[str] = field(default_factory=list)

    # Remove __post_init__ validation - we'll validate in the builder instead


class LayerTestConfig:
    """Enhanced configuration class for layer-specific test data"""

    def __init__(
        self: LayerTestConfig,
        op_type: str,
        valid_inputs: list[str],
        valid_attributes: dict[str, Any],
        required_initializers: dict[str, np.ndarray],
        input_shapes: dict[str, list[int]] | None = None,
        output_shapes: dict[str, list[int]] | None = None,
    ) -> None:
        self.op_type = op_type
        self.valid_inputs = valid_inputs
        self.valid_attributes = valid_attributes
        self.required_initializers = required_initializers
        self.input_shapes = input_shapes or {"input": [1, 16, 224, 224]}
        self.output_shapes = output_shapes or {f"{op_type.lower()}_output": [1, 10]}

    def create_node(
        self: LayerTestConfig,
        name_suffix: str = "",
        **attr_overrides: dict[str, Any],
    ) -> onnx.NodeProto:
        """Create a valid node for this layer type"""
        attrs = {**self.valid_attributes, **attr_overrides}
        return helper.make_node(
            self.op_type,
            inputs=self.valid_inputs,
            outputs=[f"{self.op_type.lower()}_output{name_suffix}"],
            name=f"test_{self.op_type.lower()}{name_suffix}",
            **attrs,
        )

    def create_initializers(
        self: LayerTestConfig,
        **initializer_overrides: dict[str, Any],
    ) -> dict[str, onnx.TensorProto]:
        """Create initializer tensors for this layer"""
        initializers = {}
        combined_inits = {**self.required_initializers, **initializer_overrides}
        for name, data in combined_inits.items():
            # Special handling for shape tensors in Reshape, etc.
            if name == "shape":
                tensor = numpy_helper.from_array(data.astype(np.int64), name=name)
            else:
                tensor = numpy_helper.from_array(data.astype(np.float32), name=name)
            initializers[name] = tensor
        return initializers

    def create_test_model(self, test_spec: LayerTestSpec) -> onnx.ModelProto:
        """Create a complete model for a specific test case"""

        # Determine node-level inputs.
        # If dev overrides inputs explicitly,
        # respect that; otherwise use original valid_inputs.
        inputs = test_spec.input_overrides or self.valid_inputs

        # Prepare attributes and remove omitted attributes if specified
        attrs = {**self.valid_attributes, **test_spec.attr_overrides}
        for key in getattr(test_spec, "omit_attrs", []):
            attrs.pop(key, None)

        # Create initializers (may introduce overrides)
        initializers = self.create_initializers(**test_spec.initializer_overrides)

        # Apply shape overrides
        input_shapes = {**self.input_shapes, **test_spec.input_shape_overrides}
        output_shapes = {**self.output_shapes, **test_spec.output_shape_overrides}

        # ----------------------------------------
        # REMOVE graph inputs that are also initializers
        # ----------------------------------------
        initializer_names = set(initializers.keys())

        # Also remove shapes for initializer inputs
        for init_name in initializer_names:
            input_shapes.pop(init_name, None)

        # Create ONNX input value infos ONLY from filtered inputs
        graph_inputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in input_shapes.items()
        ]

        # Outputs stay unchanged
        graph_outputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in output_shapes.items()
        ]

        node = helper.make_node(
            self.op_type,
            inputs=inputs,
            outputs=[f"{self.op_type.lower()}_output"],
            name=f"test_{self.op_type.lower()}_{test_spec.name}",
            **attrs,
        )

        # Build the graph
        graph = helper.make_graph(
            nodes=[node],
            name=f"{self.op_type.lower()}_test_graph_{test_spec.name}",
            inputs=graph_inputs,
            outputs=graph_outputs,
            initializer=list(initializers.values()),
        )

        return helper.make_model(graph)


class TestSpecBuilder:
    """Builder for creating test specifications"""

    def __init__(self, name: str, spec_type: SpecType) -> None:
        self._spec = LayerTestSpec(name=name, spec_type=spec_type)

    def description(self, desc: str) -> TestSpecBuilder:
        self._spec.description = desc
        return self

    def override_attrs(self, **attrs: dict[str, Any]) -> TestSpecBuilder:
        self._spec.attr_overrides.update(attrs)
        return self

    def omit_attrs(self, *attrs: str) -> TestSpecBuilder:
        self._spec.omit_attrs.extend(attrs)
        return self

    def override_initializer(self, name: str, data: np.ndarray) -> TestSpecBuilder:
        self._spec.initializer_overrides[name] = data
        return self

    def override_inputs(self, *inputs: str) -> TestSpecBuilder:
        self._spec.input_overrides = list(inputs)
        return self

    def override_input_shapes(self, **shapes: dict[str, list[int]]) -> TestSpecBuilder:
        self._spec.input_shape_overrides.update(shapes)
        return self

    def override_output_shapes(self, **shapes: dict[str, list[int]]) -> TestSpecBuilder:
        self._spec.output_shape_overrides.update(shapes)
        return self

    def expects_error(
        self,
        error_type: type,
        match: str | None = None,
    ) -> TestSpecBuilder:
        if self._spec.spec_type != SpecType.ERROR:
            msg = "expects_error can only be used with ERROR spec type"
            raise ValueError(msg)
        self._spec.expected_error = error_type
        self._spec.error_match = match
        return self

    def tags(self, *tags: str) -> TestSpecBuilder:
        self._spec.tags.extend(tags)
        return self

    def skip(self, reason: str) -> TestSpecBuilder:
        self._spec.skip_reason = reason
        return self

    def build(self) -> LayerTestSpec:
        # Validate before building
        if self._spec.spec_type == SpecType.ERROR and not self._spec.expected_error:
            msg = (
                f"Error test {self._spec.name} must"
                " specify expected_error using .expects_error()"
            )
            raise ValueError(msg)
        return self._spec


# Convenience functions
def valid_test(name: str) -> TestSpecBuilder:
    return TestSpecBuilder(name, SpecType.VALID)


def error_test(name: str) -> TestSpecBuilder:
    return TestSpecBuilder(name, SpecType.ERROR)


def edge_case_test(name: str) -> TestSpecBuilder:
    return TestSpecBuilder(name, SpecType.EDGE_CASE)


def e2e_test(name: str) -> TestSpecBuilder:
    return TestSpecBuilder(name, SpecType.E2E)


class BaseLayerConfigProvider(ABC):
    """Abstract base class for layer config providers"""

    @abstractmethod
    def get_config(self) -> LayerTestConfig:
        """Return the base configuration for this layer"""

    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Return the layer name/op_type"""

    def get_test_specs(self) -> list[LayerTestSpec]:
        """Return test specifications for this layer (override for custom tests)"""
        return []

    def get_valid_test_specs(self) -> list[LayerTestSpec]:
        """Get only valid test specifications"""
        return [
            spec for spec in self.get_test_specs() if spec.spec_type == SpecType.VALID
        ]

    def get_error_test_specs(self) -> list[LayerTestSpec]:
        """Get only error test specifications"""
        return [
            spec for spec in self.get_test_specs() if spec.spec_type == SpecType.ERROR
        ]

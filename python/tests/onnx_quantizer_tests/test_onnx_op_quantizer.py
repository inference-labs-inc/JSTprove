from __future__ import annotations

from typing import TYPE_CHECKING, Any

import onnx
import pytest
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, helper

from python.core.model_processing.onnx_quantizer.exceptions import (
    MissingHandlerError,
    QuantizationError,
    UnsupportedOpError,
)

if TYPE_CHECKING:
    from python.core.model_processing.onnx_quantizer.layers.base import ScaleConfig

# Optional: mock layers if needed
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)


# Mocks
class MockHandler:
    def __init__(self: MockHandler) -> None:
        self.called_quantize = False
        self.called_supported = False

    def quantize(
        self: MockHandler,
        node: NodeProto,
        graph: GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, TensorProto],
    ) -> list[NodeProto]:
        _ = graph, scale_config, initializer_map
        self.called_quantize = True
        return [node]  # Return the original node as a list for simplicity

    def check_supported(
        self: MockHandler,
        node: NodeProto,
        initializer_map: dict[str, TensorProto],
    ) -> None:
        _ = initializer_map
        self.called_supported = True
        if node.name == "bad_node":
            msg = "Invalid node parameters"
            raise ValueError(msg)


# Fixtures
@pytest.fixture
def quantizer() -> ONNXOpQuantizer:
    return ONNXOpQuantizer()


@pytest.fixture
def dummy_node() -> NodeProto:
    return helper.make_node("FakeOp", inputs=["x"], outputs=["y"])


@pytest.fixture
def valid_node() -> NodeProto:
    return helper.make_node("Dummy", inputs=["x"], outputs=["y"], name="good_node")


@pytest.fixture
def invalid_node() -> NodeProto:
    return helper.make_node("Dummy", inputs=["x"], outputs=["y"], name="bad_node")


@pytest.fixture
def dummy_model(valid_node: NodeProto, invalid_node: NodeProto) -> ModelProto:
    graph = helper.make_graph(
        [valid_node, invalid_node],
        "test_graph",
        inputs=[],
        outputs=[],
        initializer=[helper.make_tensor("x", TensorProto.FLOAT, [1], [0.5])],
    )
    return helper.make_model(graph)


# Tests


@pytest.mark.unit
def test_check_model_raises_on_unsupported_op() -> None:
    quantizer = ONNXOpQuantizer()

    unsupported_node = helper.make_node("UnsupportedOp", ["x"], ["y"])
    graph = helper.make_graph([unsupported_node], "test_graph", [], [])
    model = helper.make_model(graph)

    with pytest.raises(UnsupportedOpError):
        quantizer.check_model(model)


@pytest.mark.unit
def test_check_layer_invokes_check_supported() -> None:
    quantizer = ONNXOpQuantizer()
    handler = MockHandler()
    quantizer.register("FakeOp", handler)

    node = helper.make_node("FakeOp", ["x"], ["y"])
    initializer_map = {}

    quantizer.check_layer(node, initializer_map)
    # Check that check_supported is called
    assert handler.called_supported


@pytest.mark.unit
def test_get_initializer_map_returns_correct_dict() -> None:
    quantizer = ONNXOpQuantizer()

    tensor = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[1.0],
    )
    graph = helper.make_graph([], "test_graph", [], [], [tensor])
    model = helper.make_model(graph)

    init_map = quantizer.get_initializer_map(model)
    # Test initializer in map
    assert "W" in init_map
    # Test initializer map lines up
    assert init_map["W"] == tensor
    # Enhanced: check tensor properties
    assert init_map["W"].data_type == TensorProto.FLOAT
    assert init_map["W"].dims == [1]
    assert onnx.numpy_helper.to_array(init_map["W"])[0] == 1.0


@pytest.mark.unit
def test_quantize_with_unregistered_op_warns(dummy_node: NodeProto) -> None:
    quantizer = ONNXOpQuantizer()
    graph = helper.make_graph([], "g", [], [])
    with pytest.raises(UnsupportedOpError) as excinfo:
        _ = quantizer.quantize(dummy_node, graph, 1, 1, {}, rescale=False)

    captured = str(excinfo.value)
    assert "Unsupported op type: 'FakeOp'" in captured


# Could be unit or integration?
@pytest.mark.unit
def test_check_model_raises_unsupported(dummy_model: ModelProto) -> None:
    quantizer = ONNXOpQuantizer()
    quantizer.handlers = {"Dummy": MockHandler()}

    # Remove one node to simulate unsupported ops
    dummy_model.graph.node.append(helper.make_node("FakeOp", ["a"], ["b"]))

    with pytest.raises(UnsupportedOpError) as excinfo:
        quantizer.check_model(dummy_model)

    assert "FakeOp" in str(excinfo.value)


@pytest.mark.unit
def test_check_layer_missing_handler(valid_node: NodeProto) -> None:
    quantizer = ONNXOpQuantizer()
    with pytest.raises(MissingHandlerError) as exc_info:
        quantizer.check_layer(valid_node, {})

    assert QuantizationError("").GENERIC_MESSAGE in str(exc_info.value)
    assert "No quantization handler registered for operator type 'Dummy'." in str(
        exc_info.value,
    )


@pytest.mark.unit
def test_check_layer_with_bad_handler(invalid_node: NodeProto) -> None:
    quantizer = ONNXOpQuantizer()
    quantizer.handlers = {"Dummy": MockHandler()}

    # This error is created in our mock handler
    with pytest.raises(ValueError, match="Invalid node parameters"):
        quantizer.check_layer(invalid_node, {})


@pytest.mark.unit
def test_get_initializer_map_extracts_all() -> None:
    one_f = 1.0
    two_f = 2.0
    count_init = 2
    tensor1 = helper.make_tensor("a", TensorProto.FLOAT, [1], [one_f])
    tensor2 = helper.make_tensor("b", TensorProto.FLOAT, [1], [two_f])
    graph = helper.make_graph([], "g", [], [], initializer=[tensor1, tensor2])
    model = helper.make_model(graph)

    quantizer = ONNXOpQuantizer()
    init_map = quantizer.get_initializer_map(model)
    assert init_map["a"].float_data[0] == one_f
    assert init_map["b"].float_data[0] == two_f

    # Enhanced: check all properties
    assert len(init_map) == count_init
    assert init_map["a"].name == "a"
    assert init_map["a"].data_type == TensorProto.FLOAT
    assert init_map["a"].dims == [1]
    assert init_map["b"].name == "b"
    assert init_map["b"].data_type == TensorProto.FLOAT
    assert init_map["b"].dims == [1]
    # Using numpy_helper for consistency
    assert onnx.numpy_helper.to_array(init_map["a"])[0] == one_f
    assert onnx.numpy_helper.to_array(init_map["b"])[0] == two_f


@pytest.mark.unit
def test_check_layer_skips_handler_without_check_supported() -> None:
    class NoCheckHandler:
        def quantize(self, *args: tuple, **kwargs: dict[str, Any]) -> None:
            pass  # no check_supported

    quantizer = ONNXOpQuantizer()
    quantizer.register("NoCheckOp", NoCheckHandler())

    node = helper.make_node("NoCheckOp", ["x"], ["y"])
    # Should not raise
    quantizer.check_layer(node, {})


@pytest.mark.unit
def test_register_overwrites_handler() -> None:
    quantizer = ONNXOpQuantizer()
    handler1 = MockHandler()
    handler2 = MockHandler()

    quantizer.register("Dummy", handler1)
    quantizer.register("Dummy", handler2)

    assert quantizer.handlers["Dummy"] is handler2


@pytest.mark.unit
def test_check_empty_model() -> None:
    model = helper.make_model(helper.make_graph([], "empty", [], []))
    quantizer = ONNXOpQuantizer()
    # Should not raise
    quantizer.check_model(model)

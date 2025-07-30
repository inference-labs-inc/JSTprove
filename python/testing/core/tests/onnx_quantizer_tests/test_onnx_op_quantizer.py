import pytest
import onnx
from onnx import helper, TensorProto

from python.testing.core.utils.onnx_quantizer.onnx_op_quantizer import ONNXOpQuantizer
from python.testing.core.utils.onnx_quantizer.exceptions import UnsupportedOpError, InvalidParamError, GENERIC_ERROR_MESSAGE

# Optional: mock layers if needed
from python.testing.core.utils.onnx_quantizer.layers.conv import ConvQuantizer
from python.testing.core.utils.onnx_quantizer.layers.relu import ReluQuantizer

# Mocks
class MockHandler:
    def __init__(self):
        self.called_quantize = False
        self.called_supported = False

    def quantize(self, node, rescale, graph, scale, scale_base, initializer_map):
        self.called_quantize = True
        return f"quantized:{node.op_type}"

    def check_supported(self, node, initializer_map):
        self.called_supported = True
        if node.name == "bad_node":
            raise ValueError("Invalid node parameters")
        
# Fixtures
@pytest.fixture
def quantizer():
    return ONNXOpQuantizer()

@pytest.fixture
def dummy_node():
    return helper.make_node("FakeOp", inputs=["x"], outputs=["y"])

@pytest.fixture
def valid_node():
    return helper.make_node("Dummy", inputs=["x"], outputs=["y"], name="good_node")

@pytest.fixture
def invalid_node():
    return helper.make_node("Dummy", inputs=["x"], outputs=["y"], name="bad_node")

@pytest.fixture
def dummy_model(valid_node, invalid_node):
    graph = helper.make_graph(
        [valid_node, invalid_node],
        "test_graph",
        inputs=[],
        outputs=[],
        initializer=[
            helper.make_tensor("x", TensorProto.FLOAT, [1], [0.5])
        ]
    )
    model = helper.make_model(graph)
    return model

# Tests
@pytest.mark.unit
def test_register_and_quantize_calls_handler(quantizer, dummy_node):
    handler = MockHandler()
    quantizer.register("FakeOp", handler)

    result = quantizer.quantize(
        node=dummy_node,
        rescale=False,
        graph=onnx.GraphProto(),
        scale=1,
        scale_base=255,
        initializer_map={}
    )

    assert handler.called_quantize
    assert result == "quantized:FakeOp"

@pytest.mark.unit
def test_quantize_with_no_handler_returns_node(quantizer, dummy_node):
    result = quantizer.quantize(
        node=dummy_node,
        rescale=False,
        graph=onnx.GraphProto(),
        scale=1,
        scale_base=255,
        initializer_map={}
    )
    # Check no handle
    assert result == dummy_node

@pytest.mark.unit
def test_check_model_raises_on_unsupported_op():
    quantizer = ONNXOpQuantizer()

    unsupported_node = helper.make_node("UnsupportedOp", ["x"], ["y"])
    graph = helper.make_graph([unsupported_node], "test_graph", [], [])
    model = helper.make_model(graph)

    with pytest.raises(UnsupportedOpError):
        quantizer.check_model(model)

@pytest.mark.unit
def test_check_layer_invokes_check_supported():
    quantizer = ONNXOpQuantizer()
    handler = MockHandler()
    quantizer.register("FakeOp", handler)

    node = helper.make_node("FakeOp", ["x"], ["y"])
    initializer_map = {}

    quantizer.check_layer(node, initializer_map)
    # Check that check_supported is called
    assert handler.called_supported

@pytest.mark.unit
def test_get_initializer_map_returns_correct_dict():
    quantizer = ONNXOpQuantizer()

    tensor = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[1.0]
    )
    graph = helper.make_graph([], "test_graph", [], [], [tensor])
    model = helper.make_model(graph)

    init_map = quantizer.get_initializer_map(model)
    # Test initializer in map
    assert "W" in init_map
    # Test initializer map lines up
    assert init_map["W"] == tensor

@pytest.mark.unit
def test_quantize_with_unregistered_op_warns(dummy_node, capsys):
    quantizer = ONNXOpQuantizer()
    result = quantizer.quantize(dummy_node, False, None, 1, 1, {})
    assert result == dummy_node
    captured = capsys.readouterr()
    assert "No quantizer implemented for op_type: FakeOp" in captured.out

# Could be unit or integration?
@pytest.mark.unit
def test_check_model_raises_unsupported(dummy_model):
    quantizer = ONNXOpQuantizer()
    quantizer.handlers = {"Dummy": MockHandler()}

    # Remove one node to simulate unsupported ops
    dummy_model.graph.node.append(helper.make_node("FakeOp", ["a"], ["b"]))
    
    with pytest.raises(UnsupportedOpError) as excinfo:
        quantizer.check_model(dummy_model)

    assert "FakeOp" in str(excinfo.value)

@pytest.mark.unit
def test_check_layer_missing_handler(valid_node):
    quantizer = ONNXOpQuantizer()
    with pytest.raises(UnsupportedOpError) as exc_info:
        quantizer.check_layer(valid_node, {})
    
    assert GENERIC_ERROR_MESSAGE in str(exc_info.value)
    assert "Unsupported op type: 'Dummy' in node 'good_node'" in str(exc_info.value)

@pytest.mark.unit
def test_check_layer_with_bad_handler(invalid_node):
    quantizer = ONNXOpQuantizer()
    quantizer.handlers = {"Dummy": MockHandler()}

    # This error is created in our mock handler
    with pytest.raises(ValueError, match="Invalid node parameters"):
        quantizer.check_layer(invalid_node, {})

@pytest.mark.unit
def test_get_initializer_map_extracts_all():
    tensor1 = helper.make_tensor("a", TensorProto.FLOAT, [1], [1.0])
    tensor2 = helper.make_tensor("b", TensorProto.FLOAT, [1], [2.0])
    graph = helper.make_graph([], "g", [], [], initializer=[tensor1, tensor2])
    model = helper.make_model(graph)

    quantizer = ONNXOpQuantizer()
    init_map = quantizer.get_initializer_map(model)
    assert init_map["a"].float_data[0] == 1.0
    assert init_map["b"].float_data[0] == 2.0

@pytest.mark.unit
def test_check_layer_skips_handler_without_check_supported():
    class NoCheckHandler:
        def quantize(self, *args, **kwargs): pass  # no check_supported

    quantizer = ONNXOpQuantizer()
    quantizer.register("NoCheckOp", NoCheckHandler())

    node = helper.make_node("NoCheckOp", ["x"], ["y"])
    # Should not raise
    quantizer.check_layer(node, {})

@pytest.mark.unit
def test_register_overwrites_handler():
    quantizer = ONNXOpQuantizer()
    handler1 = MockHandler()
    handler2 = MockHandler()

    quantizer.register("Dummy", handler1)
    quantizer.register("Dummy", handler2)

    assert quantizer.handlers["Dummy"] is handler2

@pytest.mark.unit
def test_check_empty_model():
    model = helper.make_model(helper.make_graph([], "empty", [], []))
    quantizer = ONNXOpQuantizer()
    # Should not raise
    quantizer.check_model(model)

@pytest.mark.unit
def test_conv_with_unsupported_stride_should_fail():
    factory = TestLayerFactory()
    config = factory.get_layer_config("Conv_UnsupportedStride")
    
    model = TestONNXOpQuantizer().create_model_with_layers(["Conv_UnsupportedStride"], {"Conv_UnsupportedStride": config})
    
    quantizer = ONNXOpQuantizer()
    
    with pytest.raises(ValueError, match="Unsupported stride"):  # adjust match as needed
        quantizer.quantize_model(
            model=model,
            rescale=False,
            scale=1,
            scale_base=255,
        )

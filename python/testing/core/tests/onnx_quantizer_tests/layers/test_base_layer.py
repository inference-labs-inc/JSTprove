import onnx
import pytest
import numpy as np
from onnx import helper, TensorProto, numpy_helper
from python.testing.core.utils.onnx_quantizer.exceptions import UnsupportedOpError
from python.testing.core.utils.onnx_quantizer.layers.base import BaseOpQuantizer
# from python.testing.core.utils.onnx_quantizer.helpers import replace_input_references
from unittest.mock import MagicMock, patch

from python.testing.core.utils.onnx_quantizer.onnx_op_quantizer import ONNXOpQuantizer


class DummyQuantizer(BaseOpQuantizer):
    def __init__(self):
        self.new_initializers = []


@pytest.fixture
def dummy_tensor():
    return numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), name="W")


@pytest.fixture
def dummy_bias():
    return numpy_helper.from_array(np.array([1.0, 2.0]), name="B")


@pytest.fixture
def dummy_node():
    return helper.make_node(
        "DummyOp",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        name="DummyOp"
    )


@pytest.fixture
def dummy_graph():
    return helper.make_graph([], "dummy_graph", inputs=[], outputs=[])


@pytest.fixture
def initializer_map(dummy_tensor, dummy_bias):
    return {
        "W": dummy_tensor,
        "B": dummy_bias
    }


@pytest.fixture
def minimal_model():
    graph = onnx.helper.make_graph(
        nodes=[],  # No nodes
        name="EmptyGraph",
        inputs=[],
        outputs=[],
        initializer=[],
    )
    return onnx.helper.make_model(graph)

@pytest.fixture
def unsupported_model():
    node = onnx.helper.make_node("UnsupportedOp", ["X"], ["Y"])
    graph = onnx.helper.make_graph(
        nodes=[node],
        name="UnsupportedGraph",
        inputs=[],
        outputs=[],
        initializer=[],
    )
    return onnx.helper.make_model(graph)

@pytest.mark.unit
def test_quantize_raises_not_implemented():
    quantizer = BaseOpQuantizer()
    with pytest.raises(NotImplementedError, match="Must implement quantize method for used layer"):
        quantizer.quantize(None, False, None, 1, 1, {})

@pytest.mark.unit
def test_check_supported_returns_none(dummy_node):
    quantizer = DummyQuantizer()
    assert quantizer.check_supported(dummy_node, {}) is None

@pytest.mark.unit
def test_rescale_layer_modifies_node_output(dummy_node, dummy_graph):
    quantizer = DummyQuantizer()
    result_nodes = quantizer.rescale_layer(dummy_node, scale_base=10, scale=2, graph=dummy_graph)

    assert len(result_nodes) == 2
    assert dummy_node.output[0] == "Y_raw"
    assert result_nodes[1].op_type == "Div"
    assert result_nodes[1].output[0] == "Y"

    # Check if scale tensor added
    assert quantizer.new_initializers
    scale_tensor = quantizer.new_initializers[0]
    assert scale_tensor.name.endswith("_scale")
    assert onnx.numpy_helper.to_array(scale_tensor)[0] == 100

@pytest.mark.unit
def test_quantize_w_and_b_adds_initializers(dummy_node, initializer_map):
    quantizer = DummyQuantizer()

    with patch("python.testing.core.utils.onnx_quantizer.layers.base.create_quantized_initializer") as mock_qinit:
        mock_qinit.side_effect = lambda tensor, scale_exponent, scale, scale_base: (
            numpy_helper.from_array(np.array([1, 2, 3]), name=f"{tensor.name}_q"),
            f"{tensor.name}_q"
        )

        new_inputs = quantizer.quantize_w_and_b(
            dummy_node, scale=3, scale_base=10, initializer_map=initializer_map
        )

    assert new_inputs[1] == "W_q"
    assert new_inputs[2] == "B_q"
    assert len(quantizer.new_initializers) == 2

@pytest.mark.unit
def test_add_nodes_w_and_b_creates_mul_and_cast(dummy_node, dummy_graph, initializer_map):
    quantizer = DummyQuantizer()
    nodes, new_inputs = quantizer.add_nodes_w_and_b(
        dummy_node, scale=2, scale_base=10, initializer_map=initializer_map, graph=dummy_graph
    )

    assert len(nodes) == 4  # Mul + Cast for W, Mul + Cast for B
    assert nodes[0].op_type == "Mul"
    assert nodes[1].op_type == "Cast"
    assert nodes[2].op_type == "Mul"
    assert nodes[3].op_type == "Cast"
    assert new_inputs == ["X", "W_scaled_cast", "B_scaled_cast"]
    assert len(quantizer.new_initializers) == 2

@pytest.mark.unit
def test_insert_scale_node_creates_mul_and_cast(dummy_tensor, dummy_graph):
    quantizer = DummyQuantizer()
    output_name, mul_node, cast_node = quantizer.insert_scale_node(
        dummy_tensor, scale_base=10, scale=1, graph=dummy_graph
    )

    assert mul_node.op_type == "Mul"
    assert cast_node.op_type == "Cast"
    assert "_scaled" in mul_node.output[0]
    assert output_name.endswith("_cast")
    assert len(quantizer.new_initializers) == 1
    assert quantizer.new_initializers[0].name.endswith("_scale")
    assert onnx.numpy_helper.to_array(quantizer.new_initializers[0])[0] == 10.0
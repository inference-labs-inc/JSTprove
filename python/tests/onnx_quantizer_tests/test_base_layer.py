from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

if TYPE_CHECKING:
    from typing import Any

    from onnx import GraphProto, ModelProto, NodeProto, TensorProto

from python.core.model_processing.onnx_quantizer.exceptions import (
    HandlerImplementationError,
)
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class DummyQuantizer(BaseOpQuantizer):
    def __init__(self: DummyQuantizer) -> None:
        self.new_initializers = []


@pytest.fixture
def dummy_tensor() -> TensorProto:
    return numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), name="W")


@pytest.fixture
def dummy_bias() -> TensorProto:
    return numpy_helper.from_array(np.array([1.0, 2.0]), name="B")


@pytest.fixture
def dummy_node() -> NodeProto:
    return helper.make_node(
        "DummyOp",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        name="DummyOp",
    )


@pytest.fixture
def dummy_graph() -> GraphProto:
    return helper.make_graph([], "dummy_graph", inputs=[], outputs=[])


@pytest.fixture
def initializer_map(
    dummy_tensor: TensorProto,
    dummy_bias: TensorProto,
) -> dict[str, TensorProto]:
    return {"W": dummy_tensor, "B": dummy_bias}


@pytest.fixture
def minimal_model() -> ModelProto:
    graph = onnx.helper.make_graph(
        nodes=[],  # No nodes
        name="EmptyGraph",
        inputs=[],
        outputs=[],
        initializer=[],
    )
    return onnx.helper.make_model(graph)


@pytest.fixture
def unsupported_model() -> ModelProto:
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
def test_quantize_raises_not_implemented() -> None:
    quantizer = BaseOpQuantizer()
    with pytest.raises(
        HandlerImplementationError,
    ) as excinfo:
        quantizer.quantize(
            node=None,
            graph=None,
            scale_config=ScaleConfig(exponent=1, base=1, rescale=False),
            initializer_map={},
        )
    assert "quantize() not implemented in subclass." in str(excinfo.value)


@pytest.mark.unit
def test_check_supported_returns_none(dummy_node: NodeProto) -> None:
    quantizer = DummyQuantizer()
    with pytest.raises(HandlerImplementationError) as excinfo:
        quantizer.check_supported(dummy_node, {})

    assert (
        "Handler implementation error for operator 'DummyQuantizer':"
        " check_supported() not implemented in subclass." in str(excinfo.value)
    )


@pytest.mark.unit
def test_rescale_layer_modifies_node_output(
    dummy_node: NodeProto,
    dummy_graph: GraphProto,
) -> None:
    quantizer = DummyQuantizer()
    result_nodes = quantizer.rescale_layer(
        dummy_node,
        scale_base=10,
        scale_exponent=2,
        graph=dummy_graph,
    )
    total_scale = 100.0
    count_nodes = 2

    assert len(result_nodes) == count_nodes
    assert dummy_node.output[0] == "Y_raw"
    assert result_nodes[1].op_type == "Div"
    assert result_nodes[1].output[0] == "Y"

    # Check if scale tensor added
    assert len(quantizer.new_initializers) == 1
    scale_tensor = quantizer.new_initializers[0]
    assert scale_tensor.name.endswith("_scale")
    assert scale_tensor.data_type == onnx.TensorProto.INT64
    assert onnx.numpy_helper.to_array(scale_tensor)[0] == total_scale

    # Validate that result_nodes are valid ONNX nodes
    for node in result_nodes:
        assert isinstance(node, onnx.NodeProto)
        assert node.name
        assert node.op_type
        assert node.input
        assert node.output

    # Check Div node inputs: should divide Y_raw by scale
    div_node = result_nodes[1]
    assert len(div_node.input) == count_nodes
    assert div_node.input[0] == "Y_raw"
    assert div_node.input[1] == scale_tensor.name


@pytest.mark.unit
def test_add_nodes_w_and_b_creates_mul_and_cast(
    dummy_node: NodeProto,
    dummy_graph: GraphProto,
    initializer_map: dict[str, Any],
) -> None:
    _ = dummy_graph
    quantizer = DummyQuantizer()
    exp = 2
    base = 10
    nodes, new_inputs = quantizer.add_nodes_w_and_b(
        dummy_node,
        scale_exponent=exp,
        scale_base=base,
        initializer_map=initializer_map,
    )
    four = 4
    two = 2

    assert len(nodes) == four  # Mul + Cast for W, Mul + Cast for B
    assert nodes[0].op_type == "Mul"
    assert nodes[1].op_type == "Cast"
    assert nodes[2].op_type == "Mul"
    assert nodes[3].op_type == "Cast"
    assert new_inputs == ["X", "W_scaled_cast", "B_scaled_cast"]
    assert len(quantizer.new_initializers) == two

    weight_scaled = base**exp
    bias_scaled = base ** (exp * 2)

    # Enhanced assertions: check node inputs/outputs and tensor details
    # Mul for W: input W and W_scale, output W_scaled
    assert nodes[0].input == ["W", "W_scale"]
    assert nodes[0].output == ["W_scaled"]
    # Cast for W: input W_scaled, output W_scaled_cast
    assert nodes[1].input == ["W_scaled"]
    assert nodes[1].output == ["W_scaled_cast"]
    # Similarly for B
    assert nodes[2].input == ["B", "B_scale"]
    assert nodes[2].output == ["B_scaled"]
    assert nodes[3].input == ["B_scaled"]
    assert nodes[3].output == ["B_scaled_cast"]

    # Check scale tensors
    w_scale = quantizer.new_initializers[0]
    b_scale = quantizer.new_initializers[1]
    assert w_scale.name == "W_scale"
    assert b_scale.name == "B_scale"
    assert onnx.numpy_helper.to_array(w_scale)[0] == weight_scaled  # 10**2
    assert onnx.numpy_helper.to_array(b_scale)[0] == bias_scaled


@pytest.mark.unit
def test_insert_scale_node_creates_mul_and_cast(
    dummy_tensor: TensorProto,
    dummy_graph: GraphProto,
) -> None:
    _ = dummy_graph
    quantizer = DummyQuantizer()
    output_name, mul_node, cast_node = quantizer.insert_scale_node(
        dummy_tensor,
        scale_base=10,
        scale_exponent=1,
    )

    assert mul_node.op_type == "Mul"
    assert cast_node.op_type == "Cast"
    assert "_scaled" in mul_node.output[0]
    assert output_name.endswith("_cast")
    assert len(quantizer.new_initializers) == 1
    assert quantizer.new_initializers[0].name.endswith("_scale")
    ten = 10.0
    assert onnx.numpy_helper.to_array(quantizer.new_initializers[0])[0] == ten

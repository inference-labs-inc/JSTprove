# This file performs very basic integration tests on each registered quantizer

import numpy as np
import onnx
import pytest
from onnx import helper

from python.core.model_processing.onnx_quantizer.layers.base import ScaleConfig
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)
from python.tests.onnx_quantizer_tests import TEST_RNG_SEED


@pytest.fixture
def dummy_graph() -> onnx.GraphProto:
    return onnx.GraphProto()


def mock_initializer_map(input_names: list[str]) -> dict[str, onnx.TensorProto]:
    rng = np.random.default_rng(TEST_RNG_SEED)
    return {
        name: onnx.helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[2, 2],  # minimal shape
            vals=rng.random(4, dtype=np.float32).tolist(),
        )
        for name in input_names
    }


def get_required_input_names(op_type: str) -> list[str]:
    try:
        schema = onnx.defs.get_schema(op_type)
        return [
            inp.name or f"input{i}"
            for i, inp in enumerate(schema.inputs)
            if inp.option != 1
        ]  # 1 = optional
    except Exception:
        return ["input0"]  # fallback


def validate_quantized_node(node_result: onnx.NodeProto, op_type: str) -> None:
    """Validate a single quantized node."""
    assert isinstance(node_result, onnx.NodeProto), f"Invalid node type for {op_type}"
    assert node_result.op_type, f"Missing op_type for {op_type}"
    assert node_result.output, f"Missing outputs for {op_type}"

    try:
        # Create a minimal graph with dummy IOs to satisfy ONNX requirements
        temp_graph = onnx.GraphProto()
        temp_graph.name = "temp_graph"

        for inp in node_result.input:
            if not any(vi.name == inp for vi in temp_graph.input):
                temp_graph.input.append(
                    onnx.helper.make_tensor_value_info(
                        inp,
                        onnx.TensorProto.FLOAT,
                        [1],
                    ),
                )

        for out in node_result.output:
            if not any(vi.name == out for vi in temp_graph.output):
                temp_graph.output.append(
                    onnx.helper.make_tensor_value_info(
                        out,
                        onnx.TensorProto.FLOAT,
                        [1],
                    ),
                )

        temp_graph.node.append(node_result)

        # Explicit opset imports for default and contrib domains
        temp_model = onnx.helper.make_model(
            temp_graph,
            opset_imports=[
                onnx.helper.make_opsetid("", 22),
                onnx.helper.make_opsetid("ai.onnx.contrib", 1),
            ],
        )

        onnx.checker.check_model(temp_model)
    except onnx.checker.ValidationError as e:
        pytest.fail(f"ONNX node validation failed for {op_type}: {e}")


@pytest.mark.integration
@pytest.mark.parametrize("op_type", list(ONNXOpQuantizer().handlers.keys()))
def test_registered_quantizer_quantize(
    op_type: str,
    dummy_graph: onnx.GraphProto,
) -> None:
    quantizer = ONNXOpQuantizer()
    handler = quantizer.handlers[op_type]

    inputs = get_required_input_names(op_type)
    dummy_initializer_map = mock_initializer_map(inputs)

    dummy_node = helper.make_node(
        op_type=op_type,
        inputs=inputs,
        outputs=["dummy_output"],
    )

    result = handler.quantize(
        node=dummy_node,
        graph=dummy_graph,
        scale_config=ScaleConfig(exponent=10, base=2, rescale=True),
        initializer_map=dummy_initializer_map,
    )
    assert result is not None

    # Enhanced assertions: validate result type and structure
    if isinstance(result, list):
        assert len(result) > 0, f"Quantize returned empty list for {op_type}"
        for node_result in result:
            validate_quantized_node(node_result, op_type)
    else:
        if inputs:
            # Only assert if this op actually requires inputs
            assert (
                result.input
            ), f"Missing inputs for {op_type}; required_inputs={inputs}"

        validate_quantized_node(result, op_type)

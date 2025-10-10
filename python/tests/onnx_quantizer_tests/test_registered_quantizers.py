# This file performs very basic integration tests on each registered quantizer

import numpy as np
import onnx
import pytest
from onnx import helper

from python.core.model_processing.onnx_quantizer.layers.base import ScaleConfig
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)


@pytest.fixture
def dummy_graph() -> onnx.GraphProto:
    return onnx.GraphProto()


def mock_initializer_map(input_names: list[int]) -> dict[str, onnx.NodeProto]:
    rng = np.random.default_rng()
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

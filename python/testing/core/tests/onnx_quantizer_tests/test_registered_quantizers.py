# This file performs very basic integration tests on each registered quantizer

import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper

from python.testing.core.utils.onnx_quantizer.onnx_op_quantizer import ONNXOpQuantizer
from python.testing.core.utils.onnx_quantizer.exceptions import UnsupportedOpError


@pytest.fixture
def dummy_graph():
    return onnx.GraphProto()

def mock_initializer_map(input_names):
    return {
        name: onnx.helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[2, 2],  # minimal shape
            vals=np.random.rand(4).astype(np.float32).tolist(),
        )
        for name in input_names
    }

def get_required_input_names(op_type: str):
    try:
        schema = onnx.defs.get_schema(op_type)
        return [inp.name or f"input{i}" for i, inp in enumerate(schema.inputs) if inp.option != 1]  # 1 = optional
    except Exception:
        return ["input0"]  # fallback

@pytest.mark.integration
@pytest.mark.parametrize("op_type", list(ONNXOpQuantizer().handlers.keys()))
def test_registered_quantizer_quantize(op_type, dummy_graph):
    quantizer = ONNXOpQuantizer()
    handler = quantizer.handlers[op_type]

    inputs = get_required_input_names(op_type)
    dummy_initializer_map = mock_initializer_map(inputs)

    dummy_node = helper.make_node(
        op_type=op_type,
        inputs=inputs,
        outputs=["dummy_output"]
    )

    result = handler.quantize(
        dummy_node,
        rescale=True,
        graph=dummy_graph,
        scale=10,
        scale_base=2,
        initializer_map=dummy_initializer_map
    )
    assert result is not None
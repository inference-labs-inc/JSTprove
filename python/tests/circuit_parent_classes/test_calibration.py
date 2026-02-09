from __future__ import annotations

import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from python.core.model_processing.converters.onnx_converter import (
    _MIN_N_BITS,
    ONNXConverter,
    ONNXLayer,
)


def _make_conv_model() -> tuple[onnx.ModelProto, list[ONNXLayer]]:
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])

    rng = np.random.default_rng(42)
    w_data = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
    b_data = rng.standard_normal(1).astype(np.float32)

    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 1],
    )

    graph = helper.make_graph(
        [conv_node],
        "conv_graph",
        [x_info],
        [y_info],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    architecture_layers = [
        ONNXLayer(
            id=0,
            name="conv0",
            op_type="Conv",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            shape={"Y": [1, 1, 4, 4]},
            tensor=None,
            params={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
            opset_version_number=13,
        ),
    ]
    return model, architecture_layers


def _make_relu_model() -> tuple[onnx.ModelProto, list[ONNXLayer]]:
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    relu_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu0")

    graph = helper.make_graph([relu_node], "relu_graph", [x_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    architecture_layers = [
        ONNXLayer(
            id=0,
            name="relu0",
            op_type="Relu",
            inputs=["X"],
            outputs=["Y"],
            shape={"Y": [1, 4]},
            tensor=None,
            params=None,
            opset_version_number=13,
        ),
    ]
    return model, architecture_layers


def _make_conv_relu_model() -> tuple[onnx.ModelProto, list[ONNXLayer]]:
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])

    rng = np.random.default_rng(42)
    w_data = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
    b_data = rng.standard_normal(1).astype(np.float32)

    w_init = numpy_helper.from_array(w_data, name="W")
    b_init = numpy_helper.from_array(b_data, name="B")

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["conv_out"],
        name="conv0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 1],
    )
    relu_node = helper.make_node(
        "Relu",
        inputs=["conv_out"],
        outputs=["Y"],
        name="relu0",
    )

    graph = helper.make_graph(
        [conv_node, relu_node],
        "conv_relu_graph",
        [x_info],
        [y_info],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    architecture_layers = [
        ONNXLayer(
            id=0,
            name="conv0",
            op_type="Conv",
            inputs=["X", "W", "B"],
            outputs=["conv_out"],
            shape={"conv_out": [1, 1, 4, 4]},
            tensor=None,
            params={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
            opset_version_number=13,
        ),
        ONNXLayer(
            id=1,
            name="relu0",
            op_type="Relu",
            inputs=["conv_out"],
            outputs=["Y"],
            shape={"Y": [1, 1, 4, 4]},
            tensor=None,
            params=None,
            opset_version_number=13,
        ),
    ]
    return model, architecture_layers


def _make_unit_conv_model() -> tuple[onnx.ModelProto, list[ONNXLayer]]:
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1, 1])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 1])

    w_data = np.ones((1, 1, 1, 1), dtype=np.float32)
    w_init = numpy_helper.from_array(w_data, name="W")

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        name="conv0",
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    graph = helper.make_graph(
        [conv_node],
        "unit_conv_graph",
        [x_info],
        [y_info],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    architecture_layers = [
        ONNXLayer(
            id=0,
            name="conv0",
            op_type="Conv",
            inputs=["X", "W"],
            outputs=["Y"],
            shape={"Y": [1, 1, 1, 1]},
            tensor=None,
            params={"kernel_shape": [1, 1], "pads": [0, 0, 0, 0], "strides": [1, 1]},
            opset_version_number=13,
        ),
    ]
    return model, architecture_layers


def _call_compute_n_bits(
    model: onnx.ModelProto,
    arch_layers: list[ONNXLayer],
    rescale_config: dict[str, bool] | None = None,
    scale_base: int = 2,
    scale_exponent: int = 18,
    input_bounds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, int]:
    converter = ONNXConverter()
    converter.model = model
    return converter.compute_n_bits_from_bounds(
        architecture_layers=arch_layers,
        rescale_config=rescale_config or {},
        scale_base=scale_base,
        scale_exponent=scale_exponent,
        input_bounds=input_bounds,
    )


class TestComputeNBitsFromBounds:
    @pytest.mark.unit
    def test_conv_returns_n_bits(self) -> None:
        model, arch_layers = _make_conv_model()
        result = _call_compute_n_bits(model, arch_layers)
        assert isinstance(result, dict)
        assert "conv0" in result

    @pytest.mark.unit
    def test_relu_returns_n_bits(self) -> None:
        model, arch_layers = _make_relu_model()
        result = _call_compute_n_bits(model, arch_layers)
        assert isinstance(result, dict)
        assert "relu0" in result

    @pytest.mark.unit
    def test_conv_relu_both_layers(self) -> None:
        model, arch_layers = _make_conv_relu_model()
        result = _call_compute_n_bits(model, arch_layers)
        assert "conv0" in result
        assert "relu0" in result

    @pytest.mark.unit
    def test_n_bits_at_least_minimum(self) -> None:
        model, arch_layers = _make_conv_model()
        result = _call_compute_n_bits(model, arch_layers)
        for layer_name, n_bits in result.items():
            assert (
                n_bits >= _MIN_N_BITS
            ), f"n_bits for {layer_name} is {n_bits}, expected >= {_MIN_N_BITS}"

    @pytest.mark.unit
    def test_empty_architecture(self) -> None:
        model, _ = _make_conv_model()
        result = _call_compute_n_bits(model, [])
        assert result == {}

    @pytest.mark.unit
    def test_deterministic(self) -> None:
        model, arch_layers = _make_conv_model()
        r1 = _call_compute_n_bits(model, arch_layers)
        r2 = _call_compute_n_bits(model, arch_layers)
        assert r1 == r2

    @pytest.mark.unit
    def test_wider_bounds_increase_n_bits(self) -> None:
        model, arch_layers = _make_conv_model()
        narrow = _call_compute_n_bits(model, arch_layers, input_bounds=(0.0, 1.0))
        wide = _call_compute_n_bits(model, arch_layers, input_bounds=(0.0, 10.0))
        for layer_name in narrow:
            assert wide[layer_name] >= narrow[layer_name]

    @pytest.mark.unit
    def test_known_conv_value(self) -> None:
        model, arch_layers = _make_unit_conv_model()
        result = _call_compute_n_bits(model, arch_layers)
        alpha = 2**18
        real_max = 1.0
        raw = alpha * real_max + 1
        expected_n_bits = max(math.ceil(math.log2(raw)) + 1, _MIN_N_BITS)
        assert result["conv0"] == expected_n_bits

    @pytest.mark.unit
    def test_unknown_op_passthrough(self) -> None:
        model, _ = _make_relu_model()
        custom_layer = ONNXLayer(
            id=0,
            name="custom0",
            op_type="CustomOp",
            inputs=["X"],
            outputs=["Y"],
            shape={"Y": [1, 4]},
            tensor=None,
            params=None,
            opset_version_number=13,
        )
        result = _call_compute_n_bits(model, [custom_layer])
        assert isinstance(result, dict)
        assert "custom0" not in result

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from python.core.model_processing.converters.onnx_converter import (
    ONNXConverter,
    ONNXLayer,
)

_MIN_N_BITS = 16


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


def _quantize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    converter = ONNXConverter()
    converter.model = model
    converter.model_file_name = "test"
    return converter.quantize_model(model, scale_base=2, scale_exponent=18)


class TestBuildCalibrationModel:
    @pytest.mark.unit
    def test_returns_pre_rescale_outputs_for_conv(self) -> None:
        model, arch_layers = _make_conv_model()
        quantized = _quantize_model(model)

        cal_model, pre_rescale, _range_names = ONNXConverter._build_calibration_model(
            quantized,
            arch_layers,
            rescale_config={},
            alpha=float(2**18),
        )

        assert len(pre_rescale) > 0
        output_names = [o.name for o in cal_model.graph.output]
        for name in pre_rescale:
            assert name in output_names
            assert name.endswith("__pre_rescale")

    @pytest.mark.unit
    def test_returns_range_check_outputs_for_relu(self) -> None:
        model, arch_layers = _make_relu_model()
        quantized = _quantize_model(model)

        _, pre_rescale, range_names = ONNXConverter._build_calibration_model(
            quantized,
            arch_layers,
            rescale_config={},
            alpha=float(2**18),
        )

        assert len(pre_rescale) == 0
        assert len(range_names) > 0

    @pytest.mark.unit
    def test_conv_relu_captures_both_types(self) -> None:
        model, arch_layers = _make_conv_relu_model()
        quantized = _quantize_model(model)

        _, pre_rescale, range_names = ONNXConverter._build_calibration_model(
            quantized,
            arch_layers,
            rescale_config={},
            alpha=float(2**18),
        )

        assert len(pre_rescale) > 0
        assert len(range_names) > 0

    @pytest.mark.unit
    def test_rescale_config_false_skips_rescaling(self) -> None:
        model, arch_layers = _make_conv_model()
        quantized = _quantize_model(model)

        _, pre_rescale, _ = ONNXConverter._build_calibration_model(
            quantized,
            arch_layers,
            rescale_config={"conv0": False},
            alpha=float(2**18),
        )

        assert len(pre_rescale) == 0

    @pytest.mark.unit
    def test_calibration_model_outputs_have_valid_types(self) -> None:
        model, arch_layers = _make_conv_model()
        quantized = _quantize_model(model)

        cal_model, _, _ = ONNXConverter._build_calibration_model(
            quantized,
            arch_layers,
            rescale_config={},
            alpha=float(2**18),
        )

        for output in cal_model.graph.output:
            assert (
                output.type.tensor_type.elem_type != TensorProto.UNDEFINED
            ), f"Output '{output.name}' has UNDEFINED type (data type 0)"


class TestCalibrateNBits:
    @pytest.mark.unit
    def test_returns_dict_with_layer_names(self) -> None:
        model, arch_layers = _make_conv_model()
        quantized = _quantize_model(model)

        converter = ONNXConverter()
        result = converter.calibrate_n_bits(
            quantized_model=quantized,
            architecture_layers=arch_layers,
            rescale_config={},
            scale_base=2,
            scale_exponent=18,
        )

        assert isinstance(result, dict)
        assert "conv0" in result

    @pytest.mark.unit
    def test_n_bits_at_least_minimum(self) -> None:
        model, arch_layers = _make_conv_model()
        quantized = _quantize_model(model)

        converter = ONNXConverter()
        result = converter.calibrate_n_bits(
            quantized_model=quantized,
            architecture_layers=arch_layers,
            rescale_config={},
            scale_base=2,
            scale_exponent=18,
        )

        for layer_name, n_bits in result.items():
            assert (
                n_bits >= _MIN_N_BITS
            ), f"n_bits for {layer_name} is {n_bits}, expected >= {_MIN_N_BITS}"

    @pytest.mark.unit
    def test_conv_relu_model_calibrates_both_layers(self) -> None:
        model, arch_layers = _make_conv_relu_model()
        quantized = _quantize_model(model)

        converter = ONNXConverter()
        result = converter.calibrate_n_bits(
            quantized_model=quantized,
            architecture_layers=arch_layers,
            rescale_config={},
            scale_base=2,
            scale_exponent=18,
        )

        assert "conv0" in result
        assert "relu0" in result

    @pytest.mark.unit
    def test_deterministic_with_same_seed(self) -> None:
        model, arch_layers = _make_conv_model()
        quantized = _quantize_model(model)

        converter = ONNXConverter()
        r1 = converter.calibrate_n_bits(
            quantized_model=quantized,
            architecture_layers=arch_layers,
            rescale_config={},
            scale_base=2,
            scale_exponent=18,
        )
        r2 = converter.calibrate_n_bits(
            quantized_model=quantized,
            architecture_layers=arch_layers,
            rescale_config={},
            scale_base=2,
            scale_exponent=18,
        )

        assert r1 == r2

    @pytest.mark.unit
    def test_empty_architecture_returns_empty_config(self) -> None:
        model, _ = _make_conv_model()
        quantized = _quantize_model(model)

        converter = ONNXConverter()
        result = converter.calibrate_n_bits(
            quantized_model=quantized,
            architecture_layers=[],
            rescale_config={},
            scale_base=2,
            scale_exponent=18,
        )

        assert result == {}

    @pytest.mark.unit
    def test_num_samples_parameter(self) -> None:
        model, arch_layers = _make_conv_model()
        quantized = _quantize_model(model)

        converter = ONNXConverter()
        result = converter.calibrate_n_bits(
            quantized_model=quantized,
            architecture_layers=arch_layers,
            rescale_config={},
            scale_base=2,
            scale_exponent=18,
            num_samples=1,
        )

        assert isinstance(result, dict)
        assert len(result) > 0

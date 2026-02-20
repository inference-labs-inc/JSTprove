from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from python.core.circuit_models.generic_onnx import GenericModelONNX

IN_CHANNELS = 3
OUT_CHANNELS = 4
KERNEL_SIZE = 3
SPATIAL = 5
OUT_SPATIAL = SPATIAL - KERNEL_SIZE + 1

X_SIZE = 1 * IN_CHANNELS * SPATIAL * SPATIAL
W_SIZE = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
B_SIZE = OUT_CHANNELS
OUTPUT_SIZE = 1 * OUT_CHANNELS * OUT_SPATIAL * OUT_SPATIAL


def _build_wai_conv_model() -> onnx.ModelProto:
    rng = np.random.default_rng(42)
    w_data = rng.normal(
        0,
        0.5,
        (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
    ).astype(np.float32)
    b_data = rng.normal(0, 0.1, (OUT_CHANNELS,)).astype(np.float32)

    x_input = helper.make_tensor_value_info(
        "X",
        TensorProto.FLOAT,
        [1, IN_CHANNELS, SPATIAL, SPATIAL],
    )
    w_input = helper.make_tensor_value_info(
        "W",
        TensorProto.FLOAT,
        [OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE],
    )
    b_input = helper.make_tensor_value_info(
        "B",
        TensorProto.FLOAT,
        [OUT_CHANNELS],
    )
    y_output = helper.make_tensor_value_info(
        "Y",
        TensorProto.FLOAT,
        [1, OUT_CHANNELS, OUT_SPATIAL, OUT_SPATIAL],
    )

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[KERNEL_SIZE, KERNEL_SIZE],
    )

    graph = helper.make_graph(
        [conv_node],
        "wai_conv",
        [x_input, w_input, b_input],
        [y_output],
        initializer=[
            numpy_helper.from_array(w_data, name="W"),
            numpy_helper.from_array(b_data, name="B"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    return model


def _mock_get_outputs(
    _self: GenericModelONNX,
    _inputs: np.ndarray | dict[str, np.ndarray],
) -> torch.Tensor:
    return torch.zeros(OUTPUT_SIZE)


@pytest.fixture
def wai_model_path(tmp_path: Path) -> str:
    model = _build_wai_conv_model()
    path = tmp_path / "wai_conv.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def wai_circuit(wai_model_path: str) -> GenericModelONNX:
    circuit = GenericModelONNX("_")
    circuit.load_quantized_model(wai_model_path)
    circuit.weights_as_inputs = True
    return circuit


@pytest.mark.unit
def test_process_for_witness_wai_input_ordering(
    wai_circuit: GenericModelONNX,
) -> None:
    """Verify WAI inputs are flattened in graph input order
    [X, W, B], not in arbitrary set iteration order."""
    rng = np.random.default_rng(99)
    x_data = rng.normal(
        0,
        1,
        (1, IN_CHANNELS, SPATIAL, SPATIAL),
    ).tolist()

    with patch.object(
        GenericModelONNX,
        "get_outputs",
        _mock_get_outputs,
    ):
        circuit_inputs, _ = wai_circuit.process_for_witness(
            {"X": x_data},
        )

    flat = circuit_inputs["input"]
    assert len(flat) == X_SIZE + W_SIZE + B_SIZE

    scale = float(wai_circuit.scale_base**wai_circuit.scale_exponent)
    x_expected = (
        (torch.as_tensor(x_data, dtype=torch.float64) * scale).long().flatten().tolist()
    )
    assert flat[:X_SIZE] == x_expected

    model_inits = {i.name: i for i in wai_circuit.quantized_model.graph.initializer}
    w_arr = numpy_helper.to_array(model_inits["W"])
    b_arr = numpy_helper.to_array(model_inits["B"])

    w_expected = (
        (torch.as_tensor(w_arr.tolist(), dtype=torch.float64) * scale)
        .long()
        .flatten()
        .tolist()
    )
    b_expected = (
        (torch.as_tensor(b_arr.tolist(), dtype=torch.float64) * scale)
        .long()
        .flatten()
        .tolist()
    )

    assert flat[X_SIZE : X_SIZE + W_SIZE] == w_expected
    assert flat[X_SIZE + W_SIZE :] == b_expected


@pytest.mark.unit
def test_process_for_witness_wai_deterministic(
    wai_circuit: GenericModelONNX,
) -> None:
    """Verify process_for_witness produces identical results
    across 20 repeated calls."""
    rng = np.random.default_rng(99)
    x_data = rng.normal(
        0,
        1,
        (1, IN_CHANNELS, SPATIAL, SPATIAL),
    ).tolist()

    with patch.object(
        GenericModelONNX,
        "get_outputs",
        _mock_get_outputs,
    ):
        reference, _ = wai_circuit.process_for_witness(
            {"X": x_data},
        )

        for _ in range(20):
            wai_circuit._cached_input_scales = None
            result, _ = wai_circuit.process_for_witness(
                {"X": x_data},
            )
            assert result["input"] == reference["input"]


@pytest.mark.unit
def test_process_for_witness_wai_per_input_scales(
    tmp_path: Path,
) -> None:
    """Verify per-input scales from model initializers
    are applied correctly."""
    model = _build_wai_conv_model()

    x_scale_val = 262144.0
    w_scale_val = 262144.0
    b_scale_val = 68719476736.0

    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([x_scale_val], dtype=np.float32),
            name="X_scale",
        ),
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([w_scale_val], dtype=np.float32),
            name="W_scale",
        ),
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([b_scale_val], dtype=np.float32),
            name="B_scale",
        ),
    )

    path = tmp_path / "wai_conv_scales.onnx"
    onnx.save(model, str(path))

    circuit = GenericModelONNX("_")
    circuit.load_quantized_model(str(path))
    circuit.weights_as_inputs = True

    rng = np.random.default_rng(99)
    x_data = rng.normal(
        0,
        1,
        (1, IN_CHANNELS, SPATIAL, SPATIAL),
    ).tolist()

    with patch.object(
        GenericModelONNX,
        "get_outputs",
        _mock_get_outputs,
    ):
        circuit_inputs, _ = circuit.process_for_witness(
            {"X": x_data},
        )

    flat = circuit_inputs["input"]

    model_inits = {i.name: i for i in circuit.quantized_model.graph.initializer}
    w_arr = numpy_helper.to_array(model_inits["W"])
    b_arr = numpy_helper.to_array(model_inits["B"])

    w_segment = flat[X_SIZE : X_SIZE + W_SIZE]
    w_expected = (
        (torch.as_tensor(w_arr.tolist(), dtype=torch.float64) * w_scale_val)
        .long()
        .flatten()
        .tolist()
    )
    assert w_segment == w_expected

    b_segment = flat[X_SIZE + W_SIZE :]
    b_expected = (
        (torch.as_tensor(b_arr.tolist(), dtype=torch.float64) * b_scale_val)
        .long()
        .flatten()
        .tolist()
    )
    assert b_segment == b_expected

    x_segment = flat[:X_SIZE]
    x_expected = (
        (torch.as_tensor(x_data, dtype=torch.float64) * x_scale_val)
        .long()
        .flatten()
        .tolist()
    )
    assert x_segment == x_expected

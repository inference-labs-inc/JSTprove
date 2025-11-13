# ruff: noqa: S603
import json
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper


def create_simple_gemm_onnx_model(
    input_size: int,
    output_size: int,
    model_path: Path,
) -> None:
    """Create a simple ONNX model with a single GEMM layer."""
    # Define input
    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, input_size],
    )

    # Define output
    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, output_size],
    )

    # Create random number generator
    rng = np.random.default_rng()

    # Create weight tensor
    weight = rng.standard_normal((output_size, input_size)).astype(np.float32)
    weight_tensor = numpy_helper.from_array(weight, name="weight")

    # Create bias tensor
    bias = rng.standard_normal((output_size,)).astype(np.float32)
    bias_tensor = numpy_helper.from_array(bias, name="bias")

    # Create GEMM node
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        alpha=1.0,
        beta=1.0,
        transB=1,  # Transpose B (weight)
    )

    # Create graph
    graph = helper.make_graph(
        [gemm_node],
        "simple_gemm",
        [input_tensor],
        [output_tensor],
        [weight_tensor, bias_tensor],
    )

    # Create model
    model = helper.make_model(graph, producer_name="simple_gemm_creator")

    # Save model
    onnx.save(model, str(model_path))


@pytest.mark.e2e
def test_parallel_compile_and_witness_two_simple_models(  # noqa: PLR0915
    tmp_path: str,
    capsys: Generator[pytest.CaptureFixture[str], None, None],
) -> None:
    """Test compiling and running witness
    for two different simple ONNX models in parallel.
    """
    # Create two simple ONNX models with different shapes
    model1_path = Path(tmp_path) / "simple_gemm1.onnx"
    model2_path = Path(tmp_path) / "simple_gemm2.onnx"
    model1_input_size = 4
    model1_output_size = 2

    model2_input_size = 6
    model2_output_size = 3

    create_simple_gemm_onnx_model(model1_input_size, model1_output_size, model1_path)
    create_simple_gemm_onnx_model(model2_input_size, model2_output_size, model2_path)

    # Define paths for artifacts
    circuit1_path = Path(tmp_path) / "circuit1.txt"
    circuit2_path = Path(tmp_path) / "circuit2.txt"

    # Compile both models
    compile_cmd1 = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "compile",
        "-m",
        str(model1_path),
        "-c",
        str(circuit1_path),
    ]
    compile_cmd2 = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "compile",
        "-m",
        str(model2_path),
        "-c",
        str(circuit2_path),
    ]

    # Run compile commands
    result1 = subprocess.run(
        compile_cmd1,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result1.returncode == 0, f"Compile failed for model1: {result1.stderr}"

    result2 = subprocess.run(
        compile_cmd2,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result2.returncode == 0, f"Compile failed for model2: {result2.stderr}"

    # Create input files
    input1_data = {"input": [1.0] * model1_input_size}  # 10 inputs
    input2_data = {"input": [1.0] * model2_input_size}  # 20 inputs

    input1_path = Path(tmp_path) / "input1.json"
    input2_path = Path(tmp_path) / "input2.json"

    with Path.open(input1_path, "w") as f:
        json.dump(input1_data, f)
    with Path.open(input2_path, "w") as f:
        json.dump(input2_data, f)

    # Define output and witness paths
    output1_path = Path(tmp_path) / "output1.json"
    witness1_path = Path(tmp_path) / "witness1.bin"
    output2_path = Path(tmp_path) / "output2.json"
    witness2_path = Path(tmp_path) / "witness2.bin"

    # Run witness commands in parallel
    witness_cmd1 = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "witness",
        "-c",
        str(circuit1_path),
        "-i",
        str(input1_path),
        "-o",
        str(output1_path),
        "-w",
        str(witness1_path),
    ]
    witness_cmd2 = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "witness",
        "-c",
        str(circuit2_path),
        "-i",
        str(input2_path),
        "-o",
        str(output2_path),
        "-w",
        str(witness2_path),
    ]

    # Start both processes
    proc1 = subprocess.Popen(witness_cmd1)
    proc2 = subprocess.Popen(witness_cmd2)

    # Wait for both to complete
    proc1.wait()
    proc2.wait()

    # Check return codes
    assert proc1.returncode == 0, "Witness failed for model1"
    assert proc2.returncode == 0, "Witness failed for model2"

    # Verify outputs exist
    assert output1_path.exists(), "Output1 file not generated"
    assert output2_path.exists(), "Output2 file not generated"
    assert witness1_path.exists(), "Witness1 file not generated"
    assert witness2_path.exists(), "Witness2 file not generated"

    # Check output contents (should have the correct shapes)
    with Path.open(output1_path) as f:
        output1 = json.load(f)
    with Path.open(output2_path) as f:
        output2 = json.load(f)

    # Model1: input 10 -> output 5
    assert "output" in output1, "Output1 missing 'output' key"
    assert (
        len(output1["output"]) == model1_output_size
    ), f"Output1 should have {model1_output_size} elements,"
    f" got {len(output1['output'])}"

    # Model2: input 20 -> output 8
    assert "output" in output2, "Output2 missing 'output' key"
    assert (
        len(output2["output"]) == model2_output_size
    ), f"Output2 should have {model2_output_size} elements,"
    f" got {len(output2['output'])}"


def create_multi_input_multi_output_model(model_path: Path) -> None:
    """Create a simple ONNX model with two inputs and two outputs."""
    # Define inputs
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    w = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 4, 4])

    # Define outputs
    y1 = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 1, 4, 4])
    y2 = helper.make_tensor_value_info("pooled", TensorProto.FLOAT, [1, 1, 2, 2])

    # Node 1: Add
    add_node = helper.make_node("Add", inputs=["X", "W"], outputs=["sum"])

    # Node 2: MaxPool
    pool_node = helper.make_node(
        "MaxPool",
        inputs=["sum"],
        outputs=["pooled"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        dilations=[1, 1],
        pads=[0, 0, 0, 0],
        ceil_mode=0,
    )

    # Build the graph
    graph_def = helper.make_graph(
        [add_node, pool_node],
        "TwoOutputGraph",
        [x, w],
        [y1, y2],
    )

    model_def = helper.make_model(graph_def, producer_name="pytest-multi-output-model")
    onnx.save(model_def, model_path)


@pytest.mark.e2e
def test_multi_input_multi_output_model_e2e(tmp_path: Path) -> None:
    """
    E2E test: compile, witness, and verify outputs
    for a multi-input/multi-output ONNX model.
    """
    model_path = tmp_path / "multi_output_no_identity.onnx"
    circuit_path = tmp_path / "circuit.txt"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    witness_path = tmp_path / "witness.bin"
    proof_path = tmp_path / "proof.bin"

    # --- Step 1: Generate model ---
    create_multi_input_multi_output_model(model_path)

    # --- Step 2: Compile model ---
    compile_cmd = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "compile",
        "-m",
        str(model_path),
        "-c",
        str(circuit_path),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    assert (
        result.returncode == 0
    ), f"Compile failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # --- Step 3: Create input JSON ---
    # Simple constant tensors (shape [1,1,4,4])
    x = [
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        ],
    ]
    w = [
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
        ],
    ]

    with Path.open(input_path, "w") as f:
        json.dump({"X": x, "W": w}, f)

    # --- Step 4: Run witness ---
    witness_cmd = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "witness",
        "-c",
        str(circuit_path),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-w",
        str(witness_path),
    ]
    result = subprocess.run(witness_cmd, capture_output=True, text=True, check=False)
    assert (
        result.returncode == 0
    ), f"Witness failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # --- Step 5: Validate output files ---
    assert output_path.exists(), "Output file not generated"
    assert witness_path.exists(), "Witness file not generated"

    with Path.open(output_path) as f:
        outputs = json.load(f)

    output_raw = (
        (torch.as_tensor(x) * 2**18).long() + (torch.as_tensor(w) * 2**18).long()
    ).flatten()

    second_outputs = output_raw.clone().reshape([1, 1, 4, 4])

    outputs_2 = torch.max_pool2d(
        second_outputs,
        kernel_size=2,
        stride=2,
        dilation=1,
        padding=0,
    ).flatten()

    output_raw = torch.cat((output_raw, outputs_2))

    assert torch.allclose(
        torch.as_tensor(outputs["output"]),
        output_raw,
        rtol=1e-3,
        atol=1e-5,
    ), "Outputs do not match"

    # --- Step 5: Prove ---
    prove_cmd = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "prove",
        "-c",
        str(circuit_path),
        "-w",
        str(witness_path),
        "-p",
        str(proof_path),
    ]
    result = subprocess.run(prove_cmd, check=False, capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"Prove failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # --- Step 6: Verify ---
    verify_cmd = [
        sys.executable,
        "-m",
        "python.frontend.cli",
        "verify",
        "-c",
        str(circuit_path),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-w",
        str(witness_path),
        "-p",
        str(proof_path),
    ]
    result = subprocess.run(verify_cmd, check=False, capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"Verify failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # --- Step 7: Validate output ---
    assert output_path.exists(), "Output JSON not generated"
    assert witness_path.exists(), "Witness not generated"
    assert proof_path.exists(), "Proof not generated"

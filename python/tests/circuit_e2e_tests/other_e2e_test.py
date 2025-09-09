import json
import subprocess
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper


def create_simple_gemm_onnx_model(
    input_size: int,
    output_size: int,
    model_path: Path,
) -> None:
    """Create a simple ONNX model with a single GEMM layer."""
    # Define input
    input_tensor = helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, input_size],
    )

    # Define output
    output_tensor = helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.FLOAT,
        [1, output_size],
    )

    # Create weight tensor
    weight = np.random.Generator(output_size, input_size).astype(np.float32)
    weight_tensor = numpy_helper.from_array(weight, name="weight")

    # Create bias tensor
    bias = np.random.Generator(output_size).astype(np.float32)
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


@pytest.mark.e2e()
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
    quantized1_path = Path(tmp_path) / "quantized1.onnx"
    circuit2_path = Path(tmp_path) / "circuit2.txt"
    quantized2_path = Path(tmp_path) / "quantized2.onnx"

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
        "-q",
        str(quantized1_path),
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
        "-q",
        str(quantized2_path),
    ]

    # Run compile commands
    result1 = subprocess.run(
        compile_cmd1,  # noqa: S603
        capture_output=True,
        text=True,
        check=False,
    )
    assert result1.returncode == 0, f"Compile failed for model1: {result1.stderr}"

    result2 = subprocess.run(
        compile_cmd2,  # noqa: S603
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
        "-q",
        str(quantized1_path),
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
        "-q",
        str(quantized2_path),
        "-i",
        str(input2_path),
        "-o",
        str(output2_path),
        "-w",
        str(witness2_path),
    ]

    # Start both processes
    proc1 = subprocess.Popen(witness_cmd1)  # noqa: S603
    proc2 = subprocess.Popen(witness_cmd2)  # noqa: S603

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

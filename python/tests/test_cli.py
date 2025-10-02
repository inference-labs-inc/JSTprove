# python/testing/core/tests/test_cli.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.core.utils.helper_functions import RunType, to_json
from python.frontend.cli import main

# -----------------------
# unit tests: dispatch only
# -----------------------


@pytest.mark.unit()
def test_witness_dispatch(tmp_path: Path) -> None:
    # minimal files so _ensure_exists passes
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    quant = tmp_path / "q.onnx"
    quant.write_bytes(b"\x00")

    inputj = tmp_path / "in.json"
    inputj.write_text('{"input":[0]}')

    outputj = tmp_path / "out.json"  # doesn't need to pre-exist
    witness = tmp_path / "w.bin"  # doesn't need to pre-exist

    fake_circuit = MagicMock()
    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "witness",
                "-c",
                str(circuit),
                "-i",
                str(inputj),
                "-o",
                str(outputj),
                "-w",
                str(witness),
            ],
        )

    assert rc == 0
    call_args = fake_circuit.base_testing.call_args
    config = call_args[0][0]
    assert config.run_type == RunType.GEN_WITNESS
    assert config.circuit_path == str(circuit)
    assert config.input_file == str(inputj)
    assert config.output_file == str(outputj)
    assert config.witness_file == str(witness)


@pytest.mark.unit()
def test_witness_with_nested_output_paths(tmp_path: Path) -> None:
    """Test that witness creates parent directories for nested output paths."""
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    inputj = tmp_path / "in.json"
    inputj.write_text('{"input":[0]}')

    # Use nested paths that don't exist yet
    outputj = tmp_path / "outputs" / "nested" / "out.json"
    witness = tmp_path / "witnesses" / "nested" / "w.bin"

    fake_circuit = MagicMock()
    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "witness",
                "-c",
                str(circuit),
                "-i",
                str(inputj),
                "-o",
                str(outputj),
                "-w",
                str(witness),
            ],
        )

    assert rc == 0
    # Verify parent directories were created
    assert outputj.parent.exists()
    assert witness.parent.exists()
    config = fake_circuit.base_testing.call_args[0][0]
    assert config.output_file == str(outputj)
    assert config.witness_file == str(witness)


@pytest.mark.unit()
def test_prove_dispatch(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    witness = tmp_path / "w.bin"
    witness.write_bytes(b"\x00")

    proof = tmp_path / "p.bin"  # doesn't need to pre-exist

    fake_circuit = MagicMock()
    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "prove",
                "-c",
                str(circuit),
                "-w",
                str(witness),
                "-p",
                str(proof),
            ],
        )

    assert rc == 0
    call_args = fake_circuit.base_testing.call_args
    config = call_args[0][0]
    assert config.run_type == RunType.PROVE_WITNESS
    assert config.circuit_path == str(circuit)
    assert config.witness_file == str(witness)
    assert config.proof_file == str(proof)
    assert config.ecc is False


@pytest.mark.unit()
def test_prove_with_nested_proof_path(tmp_path: Path) -> None:
    """Test that prove creates parent directories for nested proof output."""
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    witness = tmp_path / "w.bin"
    witness.write_bytes(b"\x00")

    # Use nested path that doesn't exist yet
    proof = tmp_path / "proofs" / "experiment1" / "run_1" / "p.bin"

    fake_circuit = MagicMock()
    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "prove",
                "-c",
                str(circuit),
                "-w",
                str(witness),
                "-p",
                str(proof),
            ],
        )

    assert rc == 0
    # Verify parent directories were created
    assert proof.parent.exists()
    config = fake_circuit.base_testing.call_args[0][0]
    assert config.proof_file == str(proof)


# -----------------------
# integration tests: actual file writes
# -----------------------


@pytest.mark.unit()
def test_to_json_creates_nested_directories(tmp_path: Path) -> None:
    """Test that to_json creates parent directories when writing to nested paths."""
    nested_path = tmp_path / "level1" / "level2" / "level3" / "data.json"
    test_data = {"key": "value", "number": 42}

    # Directory doesn't exist yet
    assert not nested_path.parent.exists()

    # Write should succeed and create directories
    to_json(test_data, str(nested_path))

    # Verify directory was created and file was written
    assert nested_path.parent.exists()
    assert nested_path.exists()

    # Verify content
    with nested_path.open() as f:
        loaded = json.load(f)
    assert loaded == test_data


@pytest.mark.unit()
def test_verify_dispatch(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    inputj = tmp_path / "in.json"
    inputj.write_text('{"input":[0]}')

    outputj = tmp_path / "out.json"
    outputj.write_text('{"output":[0]}')  # verify requires it exists

    witness = tmp_path / "w.bin"
    witness.write_bytes(b"\x00")

    proof = tmp_path / "p.bin"
    proof.write_bytes(b"\x00")

    quant = tmp_path / "q.onnx"
    quant.write_bytes(b"\x00")

    fake_circuit = MagicMock()

    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "verify",
                "-c",
                str(circuit),
                "-i",
                str(inputj),
                "-o",
                str(outputj),
                "-w",
                str(witness),
                "-p",
                str(proof),
            ],
        )

    assert rc == 0
    call_args = fake_circuit.base_testing.call_args
    config = call_args[0][0]
    assert config.run_type == RunType.GEN_VERIFY
    assert config.circuit_path == str(circuit)
    assert config.input_file == str(inputj)
    assert config.output_file == str(outputj)
    assert config.witness_file == str(witness)
    assert config.proof_file == str(proof)
    assert config.ecc is False


@pytest.mark.unit()
def test_compile_dispatch(tmp_path: Path) -> None:
    # minimal files so _ensure_exists passes
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    circuit = tmp_path / "circuit.txt"  # doesn't need to pre-exist

    fake_circuit = MagicMock()
    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "compile",
                "-m",
                str(model),
                "-c",
                str(circuit),
            ],
        )

    assert rc == 0
    # Check attributes set on circuit
    assert fake_circuit.model_file_name == str(model)
    assert fake_circuit.onnx_path == str(model)
    assert fake_circuit.model_path == str(model)
    # Check the base_testing call
    call_args = fake_circuit.base_testing.call_args
    config = call_args[0][0]
    assert config.run_type == RunType.COMPILE_CIRCUIT
    assert config.circuit_path == str(circuit)
    assert config.dev_mode is True


@pytest.mark.unit()
def test_compile_with_nested_output_path(tmp_path: Path) -> None:
    """Test that compile creates parent directories for nested output paths."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    # Use nested path that doesn't exist yet
    circuit = tmp_path / "deeply" / "nested" / "path" / "circuit.txt"

    fake_circuit = MagicMock()
    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "compile",
                "-m",
                str(model),
                "-c",
                str(circuit),
            ],
        )

    assert rc == 0
    # Verify parent directories were created
    assert circuit.parent.exists()
    config = fake_circuit.base_testing.call_args[0][0]
    assert config.circuit_path == str(circuit)

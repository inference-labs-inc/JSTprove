# python/testing/core/tests/test_cli.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.core.utils.helper_functions import RunType
from python.frontend.cli import main

# -----------------------
# unit tests: dispatch only
# -----------------------


@pytest.mark.unit
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
    with patch(
        "python.frontend.commands.witness.WitnessCommand._build_circuit",
        return_value=fake_circuit,
    ):
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


@pytest.mark.unit
def test_witness_dispatch_positional(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    quant = tmp_path / "q.onnx"
    quant.write_bytes(b"\x00")

    inputj = tmp_path / "in.json"
    inputj.write_text('{"input":[0]}')

    outputj = tmp_path / "out.json"
    witness = tmp_path / "w.bin"

    fake_circuit = MagicMock()
    with patch(
        "python.frontend.commands.witness.WitnessCommand._build_circuit",
        return_value=fake_circuit,
    ):
        rc = main(
            [
                "--no-banner",
                "witness",
                str(circuit),
                str(inputj),
                str(outputj),
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


@pytest.mark.unit
def test_prove_dispatch(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    witness = tmp_path / "w.bin"
    witness.write_bytes(b"\x00")

    proof = tmp_path / "p.bin"  # doesn't need to pre-exist

    fake_circuit = MagicMock()
    with patch(
        "python.frontend.commands.prove.ProveCommand._build_circuit",
        return_value=fake_circuit,
    ):
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


@pytest.mark.unit
def test_prove_dispatch_positional(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    witness = tmp_path / "w.bin"
    witness.write_bytes(b"\x00")

    proof = tmp_path / "p.bin"

    fake_circuit = MagicMock()
    with patch(
        "python.frontend.commands.prove.ProveCommand._build_circuit",
        return_value=fake_circuit,
    ):
        rc = main(
            [
                "--no-banner",
                "prove",
                str(circuit),
                str(witness),
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


@pytest.mark.unit
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

    with patch(
        "python.frontend.commands.verify.VerifyCommand._build_circuit",
        return_value=fake_circuit,
    ):
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


@pytest.mark.unit
def test_verify_dispatch_positional(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    inputj = tmp_path / "in.json"
    inputj.write_text('{"input":[0]}')

    outputj = tmp_path / "out.json"
    outputj.write_text('{"output":[0]}')

    witness = tmp_path / "w.bin"
    witness.write_bytes(b"\x00")

    proof = tmp_path / "p.bin"
    proof.write_bytes(b"\x00")

    quant = tmp_path / "q.onnx"
    quant.write_bytes(b"\x00")

    fake_circuit = MagicMock()

    with patch(
        "python.frontend.commands.verify.VerifyCommand._build_circuit",
        return_value=fake_circuit,
    ):
        rc = main(
            [
                "--no-banner",
                "verify",
                str(circuit),
                str(inputj),
                str(outputj),
                str(witness),
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


@pytest.mark.unit
def test_compile_dispatch(tmp_path: Path) -> None:
    # minimal files so _ensure_exists passes
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    circuit = tmp_path / "circuit.txt"  # doesn't need to pre-exist

    fake_circuit = MagicMock()
    with patch(
        "python.frontend.commands.compile.CompileCommand._build_circuit",
        return_value=fake_circuit,
    ):
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
    assert fake_circuit.model_file_name == str(model)
    assert fake_circuit.onnx_path == str(model)
    assert fake_circuit.model_path == str(model)
    # Check the base_testing call
    call_args = fake_circuit.base_testing.call_args
    config = call_args[0][0]
    assert config.run_type == RunType.COMPILE_CIRCUIT
    assert config.circuit_path == str(circuit)
    assert config.dev_mode is True


@pytest.mark.unit
def test_compile_dispatch_positional(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    circuit = tmp_path / "circuit.txt"

    fake_circuit = MagicMock()
    with patch(
        "python.frontend.commands.compile.CompileCommand._build_circuit",
        return_value=fake_circuit,
    ):
        rc = main(
            [
                "--no-banner",
                "compile",
                str(model),
                str(circuit),
            ],
        )

    assert rc == 0
    assert fake_circuit.model_file_name == str(model)
    assert fake_circuit.onnx_path == str(model)
    assert fake_circuit.model_path == str(model)
    call_args = fake_circuit.base_testing.call_args
    config = call_args[0][0]
    assert config.run_type == RunType.COMPILE_CIRCUIT
    assert config.circuit_path == str(circuit)
    assert config.dev_mode is True

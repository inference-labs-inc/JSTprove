# python/testing/core/tests/test_cli.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.core.utils.helper_functions import RunType
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
                "-q",
                str(quant),
                "-i",
                str(inputj),
                "-o",
                str(outputj),
                "-w",
                str(witness),
            ],
        )

    assert rc == 0
    fake_circuit.base_testing.assert_called_once_with(
        run_type=RunType.GEN_WITNESS,
        circuit_path=str(circuit),
        quantized_path=str(quant),
        input_file=str(inputj),
        output_file=str(outputj),
        witness_file=str(witness),
    )


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
    kwargs = fake_circuit.base_testing.call_args.kwargs
    assert kwargs["run_type"] == RunType.PROVE_WITNESS
    assert kwargs["circuit_path"] == str(circuit)
    assert kwargs["witness_file"] == str(witness)
    assert kwargs["proof_file"] == str(proof)
    assert kwargs.get("ecc") is False


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
    # verify path calls load_quantized_model() to hydrate input shapes
    fake_circuit.load_quantized_model = MagicMock()

    with patch("python.frontend.cli._build_default_circuit", return_value=fake_circuit):
        rc = main(
            [
                "--no-banner",
                "verify",
                "-c",
                str(circuit),
                "-q",
                str(quant),
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
    fake_circuit.load_quantized_model.assert_called_once_with(str(quant))
    kwargs = fake_circuit.base_testing.call_args.kwargs
    assert kwargs["run_type"] == RunType.GEN_VERIFY
    assert kwargs["circuit_path"] == str(circuit)
    assert kwargs["input_file"] == str(inputj)
    assert kwargs["output_file"] == str(outputj)
    assert kwargs["witness_file"] == str(witness)
    assert kwargs["proof_file"] == str(proof)
    assert kwargs.get("ecc") is False

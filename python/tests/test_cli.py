# python/testing/core/tests/test_cli.py
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from python.core.circuits.errors import CircuitRunError
from python.core.model_processing.onnx_quantizer.exceptions import (
    UnsupportedOpError,
)
from python.core.utils.helper_functions import RunType
from python.frontend.cli import main
from python.frontend.commands.batch import (
    _parse_piped_result,
    _preprocess_manifest,
    _run_prove_chunk_piped,
    _run_verify_chunk_piped,
    _run_witness_chunk_piped,
    _transform_verify_job,
    _transform_witness_job,
    _validate_job_keys,
    batch_prove_piped,
    batch_verify_from_tensors,
    batch_witness_from_tensors,
)

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
    assert config.dev_mode is False


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
    assert config.dev_mode is False


@pytest.mark.unit
def test_compile_missing_model_path() -> None:
    rc = main(["--no-banner", "compile", "-c", "circuit.txt"])
    assert rc == 1


@pytest.mark.unit
def test_compile_missing_circuit_path() -> None:
    rc = main(["--no-banner", "compile", "-m", "model.onnx"])
    assert rc == 1


@pytest.mark.unit
def test_witness_missing_args() -> None:
    rc = main(["--no-banner", "witness", "-c", "circuit.txt"])
    assert rc == 1


@pytest.mark.unit
def test_prove_missing_args() -> None:
    rc = main(["--no-banner", "prove", "-c", "circuit.txt"])
    assert rc == 1


@pytest.mark.unit
def test_verify_missing_args() -> None:
    rc = main(["--no-banner", "verify", "-c", "circuit.txt"])
    assert rc == 1


@pytest.mark.unit
def test_model_check_missing_model_path() -> None:
    rc = main(["--no-banner", "model_check"])
    assert rc == 1


@pytest.mark.unit
def test_compile_file_not_found(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    rc = main(
        [
            "--no-banner",
            "compile",
            "-m",
            "nonexistent.onnx",
            "-c",
            str(circuit),
        ],
    )
    assert rc == 1


@pytest.mark.unit
def test_witness_file_not_found(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    witness = tmp_path / "w.bin"
    rc = main(
        [
            "--no-banner",
            "witness",
            "-c",
            "nonexistent.txt",
            "-i",
            "nonexistent.json",
            "-o",
            str(output),
            "-w",
            str(witness),
        ],
    )
    assert rc == 1


@pytest.mark.unit
def test_prove_file_not_found(tmp_path: Path) -> None:
    proof = tmp_path / "proof.bin"
    rc = main(
        [
            "--no-banner",
            "prove",
            "-c",
            "nonexistent.txt",
            "-w",
            "nonexistent.bin",
            "-p",
            str(proof),
        ],
    )
    assert rc == 1


@pytest.mark.unit
def test_verify_file_not_found(tmp_path: Path) -> None:
    rc = main(
        [
            "--no-banner",
            "verify",
            "-c",
            "nonexistent.txt",
            "-i",
            "nonexistent.json",
            "-o",
            "nonexistent_out.json",
            "-w",
            "nonexistent.bin",
            "-p",
            "nonexistent_proof.bin",
        ],
    )
    assert rc == 1


@pytest.mark.unit
def test_model_check_file_not_found() -> None:
    rc = main(["--no-banner", "model_check", "-m", "nonexistent.onnx"])
    assert rc == 1


@pytest.mark.unit
def test_compile_mixed_positional_and_flag(tmp_path: Path) -> None:
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
                "-c",
                str(circuit),
            ],
        )

    assert rc == 0


@pytest.mark.unit
def test_witness_mixed_positional_and_flag(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

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
                "-i",
                str(inputj),
                "-o",
                str(outputj),
                "-w",
                str(witness),
            ],
        )

    assert rc == 0


@pytest.mark.unit
def test_prove_mixed_positional_and_flag(tmp_path: Path) -> None:
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
                "-p",
                str(proof),
            ],
        )

    assert rc == 0


@pytest.mark.unit
def test_model_check_positional(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    with patch("onnx.load") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        with patch(
            "python.core.model_processing.onnx_quantizer.onnx_op_quantizer.ONNXOpQuantizer",
        ) as mock_quantizer_cls:
            mock_quantizer = MagicMock()
            mock_quantizer_cls.return_value = mock_quantizer

            rc = main(["--no-banner", "model_check", str(model)])

    assert rc == 0
    mock_load.assert_called_once_with(str(model))
    mock_quantizer.check_model.assert_called_once()


@pytest.mark.unit
def test_flag_takes_precedence_over_positional(tmp_path: Path) -> None:
    model_flag = tmp_path / "flag_model.onnx"
    model_flag.write_bytes(b"\x00")
    model_pos = tmp_path / "pos_model.onnx"
    model_pos.write_bytes(b"\x00")
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
                str(model_pos),
                "-m",
                str(model_flag),
                "-c",
                str(circuit),
            ],
        )

    assert rc == 0
    assert fake_circuit.model_path == str(model_flag)


@pytest.mark.unit
def test_parent_dir_creation(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")
    nested_circuit = tmp_path / "nested" / "deep" / "circuit.txt"

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
                str(nested_circuit),
            ],
        )

    assert rc == 0
    assert nested_circuit.parent.exists()


@pytest.mark.unit
def test_verify_mixed_positional_and_flag(tmp_path: Path) -> None:
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
                "-o",
                str(outputj),
                "-w",
                str(witness),
                "-p",
                str(proof),
            ],
        )

    assert rc == 0


@pytest.mark.unit
def test_circuit_run_error_handling(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")
    circuit = tmp_path / "circuit.txt"

    fake_circuit = MagicMock()
    fake_circuit.base_testing.side_effect = CircuitRunError("Test error")

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

    assert rc == 1


@pytest.mark.unit
def test_model_check_unsupported_op_error(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    with patch("onnx.load") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        with patch(
            "python.core.model_processing.onnx_quantizer.onnx_op_quantizer.ONNXOpQuantizer",
        ) as mock_quantizer_cls:
            mock_quantizer = MagicMock()
            mock_quantizer.check_model.side_effect = UnsupportedOpError(["BadOp"])
            mock_quantizer_cls.return_value = mock_quantizer

            rc = main(["--no-banner", "model_check", "-m", str(model)])

    assert rc == 1


@pytest.mark.unit
def test_empty_string_arg() -> None:
    rc = main(["--no-banner", "compile", "-m", "", "-c", "circuit.txt"])
    assert rc == 1


@pytest.mark.unit
def test_flag_empty_string_uses_positional(tmp_path: Path) -> None:
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
                "-m",
                "",
                "-c",
                str(circuit),
            ],
        )

    assert rc == 1


# -----------------------
# bench command tests
# -----------------------


@pytest.mark.unit
def test_bench_list_models() -> None:
    with patch(
        "python.core.utils.model_registry.list_available_models",
        return_value=["onnx: model1", "class: model2"],
    ):
        rc = main(["--no-banner", "bench", "list", "--list-models"])

    assert rc == 0


@pytest.mark.unit
def test_bench_with_model_path(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    with (
        patch(
            "python.frontend.commands.bench.model.ModelCommand._generate_model_input",
        ),
        patch("python.frontend.commands.bench.model.run_subprocess"),
    ):
        rc = main(["--no-banner", "bench", "model", "--model-path", str(model)])

    assert rc == 0


@pytest.mark.unit
def test_bench_with_model_flag() -> None:
    fake_model_entry = MagicMock()
    fake_instance = MagicMock()
    fake_instance.model_file_name = "test_model.onnx"
    fake_model_entry.loader.return_value = fake_instance
    fake_model_entry.name = "test_model"

    with (
        patch(
            "python.core.utils.model_registry.get_models_to_test",
            return_value=[fake_model_entry],
        ),
        patch(
            "python.frontend.commands.bench.model.ModelCommand._generate_model_input",
        ),
        patch("python.frontend.commands.bench.model.run_subprocess"),
    ):
        rc = main(["--no-banner", "bench", "model", "--model", "test_model"])

    assert rc == 0


@pytest.mark.unit
def test_bench_with_source_filter() -> None:
    fake_model_entry = MagicMock()
    fake_instance = MagicMock()
    fake_instance.model_file_name = "test_model.onnx"
    fake_model_entry.loader.return_value = fake_instance
    fake_model_entry.name = "test_model"

    with (
        patch(
            "python.core.utils.model_registry.get_models_to_test",
            return_value=[fake_model_entry],
        ) as mock_get,
        patch(
            "python.frontend.commands.bench.model.ModelCommand._generate_model_input",
        ),
        patch("python.frontend.commands.bench.model.run_subprocess"),
    ):
        rc = main(["--no-banner", "bench", "model", "--source", "onnx"])

    assert rc == 0
    mock_get.assert_called_once_with(None, "onnx")


@pytest.mark.unit
def test_bench_depth_sweep_simple() -> None:
    with patch("python.frontend.commands.bench.sweep.run_subprocess") as mock_run:
        rc = main(["--no-banner", "bench", "sweep", "depth"])

    assert rc == 0
    cmd = mock_run.call_args[0][0]
    assert "python.scripts.gen_and_bench" in cmd[2]
    assert "--sweep" in cmd
    assert "depth" in cmd
    assert "--depth-min" in cmd
    assert "1" in cmd
    assert "--depth-max" in cmd
    assert "16" in cmd


@pytest.mark.unit
def test_bench_breadth_sweep_simple() -> None:
    with patch("python.frontend.commands.bench.sweep.run_subprocess") as mock_run:
        rc = main(["--no-banner", "bench", "sweep", "breadth"])

    assert rc == 0
    cmd = mock_run.call_args[0][0]
    assert "python.scripts.gen_and_bench" in cmd[2]
    assert "--sweep" in cmd
    assert "breadth" in cmd
    assert "--arch-depth" in cmd
    assert "5" in cmd


@pytest.mark.unit
def test_bench_sweep_with_custom_args() -> None:
    with patch("python.frontend.commands.bench.sweep.run_subprocess") as mock_run:
        rc = main(
            [
                "--no-banner",
                "bench",
                "sweep",
                "depth",
                "--depth-min",
                "5",
                "--depth-max",
                "10",
            ],
        )

    assert rc == 0
    cmd = mock_run.call_args[0][0]
    assert "--depth-min" in cmd
    idx_min = cmd.index("--depth-min")
    assert cmd[idx_min + 1] == "5"
    assert "--depth-max" in cmd
    idx_max = cmd.index("--depth-max")
    assert cmd[idx_max + 1] == "10"


@pytest.mark.unit
def test_bench_sweep_with_optional_args() -> None:
    with patch("python.frontend.commands.bench.sweep.run_subprocess") as mock_run:
        rc = main(
            [
                "--no-banner",
                "bench",
                "sweep",
                "depth",
                "--tag",
                "test_tag",
                "--onnx-dir",
                "custom_onnx",
            ],
        )

    assert rc == 0
    cmd = mock_run.call_args[0][0]
    assert "--tag" in cmd
    assert "test_tag" in cmd
    assert "--onnx-dir" in cmd
    assert "custom_onnx" in cmd


@pytest.mark.unit
def test_bench_missing_required_args() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--no-banner", "bench"])
    # argparse exits with code 2 for usage errors
    assert exc_info.value.code == 2  # noqa: PLR2004


@pytest.mark.unit
def test_bench_nonexistent_model_path() -> None:
    rc = main(["--no-banner", "bench", "model", "-m", "nonexistent.onnx"])
    assert rc == 1


@pytest.mark.unit
def test_bench_no_models_found() -> None:
    with patch(
        "python.core.utils.model_registry.get_models_to_test",
        return_value=[],
    ):
        rc = main(["--no-banner", "bench", "model", "--model", "nonexistent_model"])

    assert rc == 1


@pytest.mark.unit
def test_bench_subprocess_failure(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    fake_circuit = MagicMock()
    fake_circuit.get_inputs.return_value = {"input": [0]}
    fake_circuit.format_inputs.return_value = {"input": [0]}

    with (
        patch(
            "python.frontend.commands.bench.model.ModelCommand._build_circuit",
            return_value=fake_circuit,
        ),
        patch(
            "python.frontend.commands.bench.model.run_subprocess",
            side_effect=RuntimeError("Subprocess failed"),
        ),
    ):
        rc = main(["--no-banner", "bench", "model", "-m", str(model)])

    assert rc == 1


@pytest.mark.unit
def test_bench_model_load_failure(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    fake_circuit = MagicMock()
    fake_circuit.load_model.side_effect = RuntimeError("Failed to load model")

    with patch(
        "python.frontend.commands.bench.model.ModelCommand._build_circuit",
        return_value=fake_circuit,
    ):
        rc = main(["--no-banner", "bench", "model", "-m", str(model)])

    assert rc == 1


@pytest.mark.unit
def test_bench_input_generation_failure(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    fake_circuit = MagicMock()
    fake_circuit.load_model.return_value = None
    fake_circuit.get_inputs.side_effect = RuntimeError("Failed to generate input")

    with patch(
        "python.frontend.commands.bench.model.ModelCommand._build_circuit",
        return_value=fake_circuit,
    ):
        rc = main(["--no-banner", "bench", "model", "-m", str(model)])

    assert rc == 1


@pytest.mark.unit
def test_bench_with_iterations(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00")

    with (
        patch(
            "python.frontend.commands.bench.model.ModelCommand._generate_model_input",
        ),
        patch("python.frontend.commands.bench.model.run_subprocess") as mock_run,
    ):
        rc = main(
            [
                "--no-banner",
                "bench",
                "model",
                "--model-path",
                str(model),
                "--iterations",
                "10",
            ],
        )

    assert rc == 0
    cmd = mock_run.call_args[0][0]
    assert "--iterations" in cmd
    idx = cmd.index("--iterations")
    assert cmd[idx + 1] == "10"


@pytest.mark.unit
def test_batch_prove_dispatch(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"jobs": []}')

    metadata = tmp_path / "circuit_metadata.json"
    metadata.write_text("{}")

    fake_circuit = MagicMock()
    fake_circuit.name = "test_circuit"

    with (
        patch(
            "python.frontend.commands.batch.BatchCommand._build_circuit",
            return_value=fake_circuit,
        ),
        patch(
            "python.frontend.commands.batch.run_cargo_command",
        ) as mock_cargo,
    ):
        rc = main(
            [
                "--no-banner",
                "batch",
                "prove",
                "-c",
                str(circuit),
                "-f",
                str(manifest),
            ],
        )

    assert rc == 0
    mock_cargo.assert_called_once()
    call_kwargs = mock_cargo.call_args[1]
    assert call_kwargs["command_type"] == "run_batch_prove"
    assert call_kwargs["args"]["c"] == str(circuit)
    assert call_kwargs["args"]["f"] == str(manifest)


@pytest.mark.unit
def test_batch_verify_dispatch(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"jobs": []}')

    metadata = tmp_path / "circuit_metadata.json"
    metadata.write_text("{}")

    processed = tmp_path / "manifest_processed.json"
    processed.write_text('{"jobs": []}')

    fake_circuit = MagicMock()
    fake_circuit.name = "test_circuit"

    with (
        patch(
            "python.frontend.commands.batch.BatchCommand._build_circuit",
            return_value=fake_circuit,
        ),
        patch(
            "python.frontend.commands.batch._preprocess_manifest",
            return_value=str(processed),
        ) as mock_preprocess,
        patch(
            "python.frontend.commands.batch.run_cargo_command",
        ) as mock_cargo,
    ):
        rc = main(
            [
                "--no-banner",
                "batch",
                "verify",
                "-c",
                str(circuit),
                "-f",
                str(manifest),
            ],
        )

    assert rc == 0
    mock_preprocess.assert_called_once()
    mock_cargo.assert_called_once()
    call_kwargs = mock_cargo.call_args[1]
    assert call_kwargs["command_type"] == "run_batch_verify"
    assert call_kwargs["args"]["f"] == str(processed)


@pytest.mark.unit
def test_batch_witness_dispatch(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"jobs": []}')

    metadata = tmp_path / "circuit_metadata.json"
    metadata.write_text("{}")

    processed = tmp_path / "manifest_processed.json"
    processed.write_text('{"jobs": []}')

    fake_circuit = MagicMock()
    fake_circuit.name = "test_circuit"

    with (
        patch(
            "python.frontend.commands.batch.BatchCommand._build_circuit",
            return_value=fake_circuit,
        ),
        patch(
            "python.frontend.commands.batch._preprocess_manifest",
            return_value=str(processed),
        ) as mock_preprocess,
        patch(
            "python.frontend.commands.batch.run_cargo_command",
        ) as mock_cargo,
    ):
        rc = main(
            [
                "--no-banner",
                "batch",
                "witness",
                "-c",
                str(circuit),
                "-f",
                str(manifest),
            ],
        )

    assert rc == 0
    mock_preprocess.assert_called_once()
    mock_cargo.assert_called_once()
    call_kwargs = mock_cargo.call_args[1]
    assert call_kwargs["command_type"] == "run_batch_witness"
    assert call_kwargs["args"]["f"] == str(processed)


@pytest.mark.unit
def test_batch_missing_circuit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"jobs": []}')

    rc = main(
        [
            "--no-banner",
            "batch",
            "prove",
            "-c",
            str(tmp_path / "nonexistent.txt"),
            "-f",
            str(manifest),
        ],
    )

    assert rc == 1


@pytest.mark.unit
def test_batch_missing_manifest(tmp_path: Path) -> None:
    circuit = tmp_path / "circuit.txt"
    circuit.write_text("ok")

    rc = main(
        [
            "--no-banner",
            "batch",
            "prove",
            "-c",
            str(circuit),
            "-f",
            str(tmp_path / "nonexistent.json"),
        ],
    )

    assert rc == 1


@pytest.mark.unit
@patch("python.frontend.commands.batch.run_cargo_command_piped")
def test_run_witness_chunk_piped_builds_payload(
    mock_piped: MagicMock,
) -> None:
    mock_piped.return_value = MagicMock(
        stdout=b'{"succeeded":1,"failed":0,"errors":[]}',
    )
    jobs = [
        {
            "_circuit_inputs": {"input": [1, 2]},
            "_circuit_outputs": {"output": [3]},
            "witness": "/data/w.bin",
        },
    ]
    result = _run_witness_chunk_piped("circ", "/c.txt", "/m.json", jobs)

    assert result["succeeded"] == 1
    assert result["failed"] == 0
    mock_piped.assert_called_once()
    call_kwargs = mock_piped.call_args[1]
    assert call_kwargs["command_type"] == "run_pipe_witness"
    payload = json.loads(call_kwargs["payload"])
    assert len(payload["jobs"]) == 1
    assert payload["jobs"][0]["witness"] == "/data/w.bin"


@pytest.mark.unit
@patch("python.frontend.commands.batch.run_cargo_command_piped")
def test_run_prove_chunk_piped_builds_payload(
    mock_piped: MagicMock,
) -> None:
    expected_job_count = 2
    mock_piped.return_value = MagicMock(
        stdout=b'{"succeeded":2,"failed":0,"errors":[]}',
    )
    jobs = [
        {"witness": "/data/w1.bin", "proof": "/data/p1.bin"},
        {"witness": "/data/w2.bin", "proof": "/data/p2.bin"},
    ]
    result = _run_prove_chunk_piped("circ", "/c.txt", "/m.json", jobs)

    assert result["succeeded"] == expected_job_count
    mock_piped.assert_called_once()
    payload = json.loads(mock_piped.call_args[1]["payload"])
    assert len(payload["jobs"]) == expected_job_count
    assert payload["jobs"][0]["witness"] == "/data/w1.bin"
    assert payload["jobs"][1]["proof"] == "/data/p2.bin"


@pytest.mark.unit
@patch("python.frontend.commands.batch.run_cargo_command_piped")
def test_run_verify_chunk_piped_builds_payload(
    mock_piped: MagicMock,
) -> None:
    mock_piped.return_value = MagicMock(
        stdout=b'{"succeeded":1,"failed":0,"errors":[]}',
    )
    jobs = [
        {
            "_circuit_inputs": {"input": [1]},
            "_circuit_outputs": {"output": [2]},
            "witness": "/data/w.bin",
            "proof": "/data/p.bin",
        },
    ]
    result = _run_verify_chunk_piped("circ", "/c.txt", "/m.json", jobs)

    assert result["succeeded"] == 1
    payload = json.loads(mock_piped.call_args[1]["payload"])
    assert payload["jobs"][0]["proof"] == "/data/p.bin"


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_witness_chunk_piped")
def test_batch_witness_from_tensors_single_chunk(
    mock_chunk: MagicMock,
) -> None:
    expected_job_count = 2
    mock_chunk.return_value = {
        "succeeded": expected_job_count,
        "failed": 0,
        "errors": [],
    }

    circuit = MagicMock()
    circuit.scale_inputs_only.return_value = {"scaled": True}
    circuit.reshape_inputs_for_inference.return_value = {"inf": True}
    circuit.reshape_inputs_for_circuit.return_value = {"circ": True}
    circuit.get_outputs.return_value = [0.5]
    circuit.format_outputs.return_value = {"output": [0.5]}

    jobs = [
        {"input": {"raw": 1}, "witness": "/w1.bin"},
        {"input": {"raw": 2}, "witness": "/w2.bin"},
    ]
    result = batch_witness_from_tensors(
        circuit,
        jobs,
        "/models/test_circuit.txt",
    )

    assert result["succeeded"] == expected_job_count
    assert result["failed"] == 0
    mock_chunk.assert_called_once()
    chunk_jobs = mock_chunk.call_args[1]["chunk_jobs"]
    assert len(chunk_jobs) == expected_job_count
    assert chunk_jobs[0]["_circuit_inputs"] == {"circ": True}
    assert chunk_jobs[1]["_circuit_outputs"] == {"output": [0.5]}


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_witness_chunk_piped")
def test_batch_witness_from_tensors_chunked(
    mock_chunk: MagicMock,
) -> None:
    expected_total = 2
    mock_chunk.side_effect = [
        {"succeeded": 1, "failed": 0, "errors": []},
        {"succeeded": 1, "failed": 0, "errors": []},
    ]

    circuit = MagicMock()
    circuit.scale_inputs_only.return_value = {}
    circuit.reshape_inputs_for_inference.return_value = {}
    circuit.reshape_inputs_for_circuit.return_value = {}
    circuit.get_outputs.return_value = []
    circuit.format_outputs.return_value = {}

    jobs = [
        {"input": {}, "witness": "/w1.bin"},
        {"input": {}, "witness": "/w2.bin"},
    ]
    result = batch_witness_from_tensors(
        circuit,
        jobs,
        "/models/test_circuit.txt",
        chunk_size=1,
    )

    assert result["succeeded"] == expected_total
    assert mock_chunk.call_count == expected_total


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_witness_chunk_piped")
def test_batch_witness_from_tensors_reads_file_inputs(
    mock_chunk: MagicMock,
) -> None:
    mock_chunk.return_value = {
        "succeeded": 1,
        "failed": 0,
        "errors": [],
    }

    circuit = MagicMock()
    circuit.scale_inputs_only.return_value = {}
    circuit.reshape_inputs_for_inference.return_value = {}
    circuit.reshape_inputs_for_circuit.return_value = {}
    circuit.get_outputs.return_value = []
    circuit.format_outputs.return_value = {}

    jobs = [{"input": "/path/to/input.json", "witness": "/w.bin"}]
    with patch(
        "python.frontend.commands.batch.read_from_json",
        return_value={"data": 1},
    ) as mock_read:
        batch_witness_from_tensors(
            circuit,
            jobs,
            "/models/test_circuit.txt",
        )
    mock_read.assert_called_once_with("/path/to/input.json")


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_prove_chunk_piped")
def test_batch_prove_piped_single_chunk(
    mock_chunk: MagicMock,
) -> None:
    expected_count = 3
    mock_chunk.return_value = {
        "succeeded": expected_count,
        "failed": 0,
        "errors": [],
    }

    jobs = [
        {"witness": f"/w{i}.bin", "proof": f"/p{i}.bin"} for i in range(expected_count)
    ]
    result = batch_prove_piped(
        "circ",
        jobs,
        "/models/circ_circuit.txt",
    )

    assert result["succeeded"] == expected_count
    mock_chunk.assert_called_once()


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_prove_chunk_piped")
def test_batch_prove_piped_chunked(
    mock_chunk: MagicMock,
) -> None:
    expected_total = 3
    expected_chunks = 2
    mock_chunk.side_effect = [
        {"succeeded": 2, "failed": 0, "errors": []},
        {"succeeded": 1, "failed": 0, "errors": []},
    ]

    jobs = [
        {"witness": f"/w{i}.bin", "proof": f"/p{i}.bin"} for i in range(expected_total)
    ]
    result = batch_prove_piped(
        "circ",
        jobs,
        "/models/circ_circuit.txt",
        chunk_size=2,
    )

    assert result["succeeded"] == expected_total
    assert mock_chunk.call_count == expected_chunks


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_verify_chunk_piped")
def test_batch_verify_from_tensors_single_chunk(
    mock_chunk: MagicMock,
) -> None:
    mock_chunk.return_value = {
        "succeeded": 1,
        "failed": 0,
        "errors": [],
    }

    circuit = MagicMock()
    circuit.reshape_inputs_for_circuit.return_value = {"circ": True}

    jobs = [
        {
            "input": {"raw": 1},
            "output": {"out": 2},
            "witness": "/w.bin",
            "proof": "/p.bin",
        },
    ]
    result = batch_verify_from_tensors(
        circuit,
        jobs,
        "/models/test_circuit.txt",
    )

    assert result["succeeded"] == 1
    mock_chunk.assert_called_once()
    chunk_jobs = mock_chunk.call_args[1]["chunk_jobs"]
    assert chunk_jobs[0]["_circuit_inputs"] == {"circ": True}
    assert chunk_jobs[0]["_circuit_outputs"] == {"out": 2}


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_verify_chunk_piped")
def test_batch_verify_from_tensors_reads_files(
    mock_chunk: MagicMock,
) -> None:
    expected_read_count = 2
    mock_chunk.return_value = {
        "succeeded": 1,
        "failed": 0,
        "errors": [],
    }

    circuit = MagicMock()
    circuit.reshape_inputs_for_circuit.return_value = {}

    jobs = [
        {
            "input": "/in.json",
            "output": "/out.json",
            "witness": "/w.bin",
            "proof": "/p.bin",
        },
    ]
    with patch(
        "python.frontend.commands.batch.read_from_json",
        side_effect=[{"i": 1}, {"o": 2}],
    ) as mock_read:
        batch_verify_from_tensors(
            circuit,
            jobs,
            "/models/test_circuit.txt",
        )
    assert mock_read.call_count == expected_read_count


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_witness_chunk_piped")
def test_batch_witness_aggregates_errors(
    mock_chunk: MagicMock,
) -> None:
    mock_chunk.side_effect = [
        {
            "succeeded": 0,
            "failed": 1,
            "errors": [[0, "bad input"]],
        },
        {"succeeded": 1, "failed": 0, "errors": []},
    ]

    circuit = MagicMock()
    circuit.scale_inputs_only.return_value = {}
    circuit.reshape_inputs_for_inference.return_value = {}
    circuit.reshape_inputs_for_circuit.return_value = {}
    circuit.get_outputs.return_value = []
    circuit.format_outputs.return_value = {}

    jobs = [
        {"input": {}, "witness": "/w1.bin"},
        {"input": {}, "witness": "/w2.bin"},
    ]
    result = batch_witness_from_tensors(
        circuit,
        jobs,
        "/models/test_circuit.txt",
        chunk_size=1,
    )

    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert len(result["errors"]) == 1


@pytest.mark.unit
def test_parse_piped_result_single_json_line() -> None:
    stdout = b'{"succeeded":3,"failed":0,"errors":[]}'
    result = _parse_piped_result(stdout)
    assert result == {"succeeded": 3, "failed": 0, "errors": []}


@pytest.mark.unit
def test_parse_piped_result_json_after_log_lines() -> None:
    expected_succeeded = 2
    stdout = (
        b"Loading circuit...\nProcessing 2 jobs.\n"
        b'{"succeeded":2,"failed":0,"errors":[]}\n'
    )
    result = _parse_piped_result(stdout)
    assert result["succeeded"] == expected_succeeded


@pytest.mark.unit
def test_parse_piped_result_no_json() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        _parse_piped_result(b"some log output\nanother line\n")


@pytest.mark.unit
def test_parse_piped_result_empty_stdout() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        _parse_piped_result(b"")


@pytest.mark.unit
def test_parse_piped_result_picks_last_json() -> None:
    stdout = b'{"first":true}\nlog line\n{"second":true}\n'
    result = _parse_piped_result(stdout)
    assert result == {"second": True}


@pytest.mark.unit
def test_validate_job_keys_passes() -> None:
    _validate_job_keys({"input": "a", "output": "b"}, "input", "output")


@pytest.mark.unit
def test_validate_job_keys_missing() -> None:
    with pytest.raises(ValueError, match=r"missing required keys.*output"):
        _validate_job_keys({"input": "a"}, "input", "output")


@pytest.mark.unit
@patch("python.frontend.commands.batch.to_json")
@patch("python.frontend.commands.batch.read_from_json", return_value={"raw": 1})
def test_transform_witness_job(
    mock_read: MagicMock,
    mock_to_json: MagicMock,
) -> None:
    circuit = MagicMock()
    circuit.scale_inputs_only.return_value = {"scaled": True}
    circuit.reshape_inputs_for_inference.return_value = {"inf": True}
    circuit.reshape_inputs_for_circuit.return_value = {"circ": True}
    circuit.get_outputs.return_value = [0.5]
    circuit.format_outputs.return_value = {"out": [0.5]}

    job: dict[str, Any] = {"input": "/data/input.json", "output": "/data/output.json"}
    _transform_witness_job(circuit, job)

    mock_read.assert_called_once_with("/data/input.json")
    expected_to_json_calls = 2
    circuit.scale_inputs_only.assert_called_once_with({"raw": 1})
    assert job["input"].endswith("_adjusted.json")
    assert mock_to_json.call_count == expected_to_json_calls
    adjusted_call, output_call = mock_to_json.call_args_list
    assert adjusted_call[0][0] == {"circ": True}
    assert output_call[0][0] == {"out": [0.5]}
    assert output_call[0][1] == "/data/output.json"


@pytest.mark.unit
def test_transform_witness_job_missing_keys() -> None:
    circuit = MagicMock()
    with pytest.raises(ValueError, match="missing required keys"):
        _transform_witness_job(circuit, {"input": "/data/input.json"})


@pytest.mark.unit
@patch("python.frontend.commands.batch.to_json")
@patch("python.frontend.commands.batch.read_from_json", return_value={"raw": 1})
def test_transform_verify_job(
    mock_read: MagicMock,
    mock_to_json: MagicMock,
) -> None:
    circuit = MagicMock()
    circuit.reshape_inputs_for_circuit.return_value = {"circ": True}

    job: dict[str, Any] = {"input": "/data/input.json"}
    _transform_verify_job(circuit, job)

    mock_read.assert_called_once_with("/data/input.json")
    assert job["input"].endswith("_veri.json")
    mock_to_json.assert_called_once_with({"circ": True}, job["input"])


@pytest.mark.unit
def test_transform_verify_job_missing_keys() -> None:
    circuit = MagicMock()
    with pytest.raises(ValueError, match="missing required keys"):
        _transform_verify_job(circuit, {"output": "x"})


@pytest.mark.unit
@patch("python.frontend.commands.batch.to_json")
@patch(
    "python.frontend.commands.batch.read_from_json",
    return_value={"jobs": [{"input": "/a.json", "output": "/b.json"}]},
)
def test_preprocess_manifest(
    _mock_read: MagicMock,
    mock_to_json: MagicMock,
    tmp_path: Path,
) -> None:
    manifest_path = str(tmp_path / "manifest.json")
    circuit_path = str(tmp_path / "model_circuit.txt")

    circuit = MagicMock()
    transform = MagicMock()

    result = _preprocess_manifest(circuit, manifest_path, circuit_path, transform)

    circuit.load_quantized_model.assert_called_once()
    transform.assert_called_once()
    assert "_processed" in result
    mock_to_json.assert_called_once()


@pytest.mark.unit
@patch(
    "python.frontend.commands.batch.read_from_json",
    return_value="not a dict",
)
def test_preprocess_manifest_invalid_format(
    _mock_read: MagicMock,
    tmp_path: Path,
) -> None:
    circuit = MagicMock()
    with pytest.raises(TypeError, match="Invalid manifest"):
        _preprocess_manifest(
            circuit,
            str(tmp_path / "bad.json"),
            str(tmp_path / "circuit.txt"),
            MagicMock(),
        )


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_prove_chunk_piped")
def test_batch_prove_piped_aggregates_errors(
    mock_chunk: MagicMock,
) -> None:
    mock_chunk.side_effect = [
        {"succeeded": 0, "failed": 1, "errors": [[0, "witness corrupt"]]},
        {"succeeded": 1, "failed": 0, "errors": []},
    ]

    jobs = [{"witness": f"/w{i}.bin", "proof": f"/p{i}.bin"} for i in range(2)]
    result = batch_prove_piped("circ", jobs, "/models/circ.txt", chunk_size=1)

    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert len(result["errors"]) == 1


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_verify_chunk_piped")
def test_batch_verify_from_tensors_aggregates_errors(
    mock_chunk: MagicMock,
) -> None:
    mock_chunk.side_effect = [
        {"succeeded": 0, "failed": 1, "errors": [[0, "mismatch"]]},
        {"succeeded": 1, "failed": 0, "errors": []},
    ]

    circuit = MagicMock()
    circuit.reshape_inputs_for_circuit.return_value = {}

    jobs = [
        {"input": {}, "output": {}, "witness": f"/w{i}.bin", "proof": f"/p{i}.bin"}
        for i in range(2)
    ]
    result = batch_verify_from_tensors(
        circuit,
        jobs,
        "/models/test_circuit.txt",
        chunk_size=1,
    )

    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert len(result["errors"]) == 1


@pytest.mark.unit
@patch("python.frontend.commands.batch._run_verify_chunk_piped")
def test_batch_verify_from_tensors_chunked(
    mock_chunk: MagicMock,
) -> None:
    expected_total = 3
    expected_chunks = 2
    mock_chunk.side_effect = [
        {"succeeded": 2, "failed": 0, "errors": []},
        {"succeeded": 1, "failed": 0, "errors": []},
    ]

    circuit = MagicMock()
    circuit.reshape_inputs_for_circuit.return_value = {}

    jobs = [
        {"input": {}, "output": {}, "witness": f"/w{i}.bin", "proof": f"/p{i}.bin"}
        for i in range(expected_total)
    ]
    result = batch_verify_from_tensors(
        circuit,
        jobs,
        "/models/test_circuit.txt",
        chunk_size=2,
    )

    assert result["succeeded"] == expected_total
    assert mock_chunk.call_count == expected_chunks

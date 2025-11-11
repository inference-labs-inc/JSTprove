# python/testing/core/tests/test_cli.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.core.circuits.errors import CircuitRunError
from python.core.model_processing.onnx_quantizer.exceptions import (
    UnsupportedOpError,
)
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

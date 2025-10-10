from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from python.core.utils.errors import ProofBackendError, ProofSystemNotImplementedError
from python.core.utils.helper_functions import (
    CircuitExecutionConfig,
    ExpanderMode,
    RunType,
    ZKProofSystems,
    compile_circuit,
    compute_and_store_output,
    generate_proof,
    generate_verification,
    generate_witness,
    get_expander_file_paths,
    get_files,
    prepare_io_files,
    read_from_json,
    run_cargo_command,
    run_end_to_end,
    to_json,
)


# ---------- compute_and_store_output ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.Path.mkdir")
@patch("python.core.utils.helper_functions.Path.exists", return_value=False)
@patch("python.core.utils.helper_functions.json.dump")
@patch("python.core.utils.helper_functions.Path.open")
def test_compute_and_store_output_saves(
    mock_file: MagicMock,
    mock_dump: MagicMock,
    mock_exists: MagicMock,
    mock_mkdir: MagicMock,
) -> None:
    mock_file.return_value.__enter__.return_value = MagicMock()
    mock_file.return_value.__exit__.return_value = None

    class Dummy:
        name = "test"
        temp_folder = "temp_test"

        @compute_and_store_output
        def get_outputs(self: Dummy) -> dict[str, int]:
            return {"out": 123}

    d = Dummy()
    result = d.get_outputs()

    mock_mkdir.assert_called_once()
    mock_dump.assert_called_once()
    assert result == {"out": 123}


@pytest.mark.unit
@patch("python.core.utils.helper_functions.Path")
@patch("python.core.utils.helper_functions.json.load", return_value={"out": 456})
def test_compute_and_store_output_loads_from_cache(
    mock_load: MagicMock,
    mock_path_class: MagicMock,
) -> None:
    # Arrange the Path mock
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.open.return_value = mock_open(
        read_data='{"out": 456}',
    ).return_value
    mock_path_class.return_value = mock_path_instance

    class Dummy:
        name = "test"
        temp_folder = "temp_test"

        @compute_and_store_output
        def get_outputs(self: Dummy) -> dict[str, int]:
            return {"out": 999}  # should not run

    # Act
    d = Dummy()
    output = d.get_outputs()

    # Assert
    assert output == {"out": 456}


@pytest.mark.unit
@patch("python.core.utils.helper_functions.os.path.exists", return_value=True)
@patch("python.core.utils.helper_functions.open", new_callable=mock_open)
@patch(
    "python.core.utils.helper_functions.json.load",
    side_effect=json.JSONDecodeError("msg", "doc", 0),
)
@patch("python.core.utils.helper_functions.json.dump")
def test_compute_and_store_output_on_json_error(
    mock_dump: MagicMock,
    mock_load: MagicMock,
    mock_file: MagicMock,
    mock_exists: MagicMock,
) -> None:

    class Dummy:
        name = "bad"
        temp_folder = "temp_test"

        @compute_and_store_output
        def get_outputs(self: Dummy) -> dict[str, bool]:
            return {"fallback": True}

    d = Dummy()
    output = d.get_outputs()
    assert output == {"fallback": True}


# ---------- prepare_io_files ----------
@pytest.mark.unit
@patch(
    "python.core.utils.helper_functions.get_files",
    return_value={
        "witness_file": "witness.wtns",
        "input_file": "input.json",
        "proof_path": "proof.json",
        "public_path": "public.json",
        "verification_key": "vk.key",
        "circuit_name": "test_circuit",
        "metadata_path": "metadata.json",
        "architecture_path": "architecture.json",
        "w_and_b_path": "w_and_b.json",
        "output_file": "out.json",
    },
)
@patch("python.core.utils.helper_functions.to_json")
@patch(
    "python.core.utils.helper_functions.os.path.splitext",
    return_value=("model", ".onnx"),
)
@patch("python.core.utils.helper_functions.open", new_callable=mock_open)
def test_prepare_io_files_runs_func(
    mock_file: MagicMock,
    mock_splitext: MagicMock,
    mock_json: MagicMock,
    mock_get_files: MagicMock,
) -> None:

    class Dummy:
        name = "model"
        input_shape = (1, 4)
        scale_base = 10
        scale_exponent = 1

        def __init__(self: Dummy) -> None:
            self.get_inputs = lambda: 1
            self.get_outputs = lambda _x=None: 2
            self.get_inputs_from_file = lambda _file_name, _is_scaled=True: 3
            self.format_inputs = lambda x: {"input": x}
            self.format_outputs = lambda x: {"output": x}
            self.get_weights = lambda: {"weights": [1, 2]}
            self.save_quantized_model = MagicMock()
            self.get_model_and_quantize = MagicMock()

        @prepare_io_files
        def base_testing(
            self: Dummy,
            exec_config: CircuitExecutionConfig,
        ) -> dict[str, bool]:
            assert exec_config.run_type == RunType.GEN_WITNESS
            return {"test": True}

    d = Dummy()
    print(CircuitExecutionConfig(run_type=RunType.GEN_WITNESS, write_json=True))
    result = d.base_testing(
        CircuitExecutionConfig(run_type=RunType.GEN_WITNESS, write_json=True),
    )
    assert result == {"test": True}
    assert d._file_info["output_file"] == "out.json"
    assert d._file_info["w_and_b_path"] == "w_and_b.json"


# ---------- to_json ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.Path")
@patch("python.core.utils.helper_functions.json.dump")
def test_to_json_saves_json(mock_dump: MagicMock, mock_path: MagicMock) -> None:
    mock_path.return_value.open.return_value.__enter__.return_value = MagicMock()

    data = {"a": 1}
    to_json(data, "output.json")

    mock_path.assert_called_with("output.json")  # verify the filename was used
    mock_path.return_value.open.assert_called_once_with(
        "w",
    )  # verify open was called with write mode
    mock_dump.assert_called_once_with(
        data,
        mock_path.return_value.open.return_value.__enter__.return_value,
    )


# ---------- read_from_json ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.Path")
@patch("python.core.utils.helper_functions.json.load", return_value={"x": 42})
def test_read_from_json_loads_json(mock_load: MagicMock, mock_path: MagicMock) -> None:
    mock_path.return_value.open.return_value.__enter__.return_value = MagicMock()
    result = read_from_json("input.json")
    mock_path.assert_called_once_with("input.json")  # check filename used
    mock_path.return_value.open.assert_called_once_with()  # open called with no args
    mock_load.assert_called_once_with(
        mock_path.return_value.open.return_value.__enter__.return_value,
    )
    assert result == {"x": 42}


# ---------- run_cargo_command ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.Path")
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_cargo_command_normal(
    mock_run: MagicMock,
    mock_path_class: MagicMock,
) -> None:
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_class.return_value = mock_path_instance

    mock_run.return_value = MagicMock(returncode=0, stdout="ok")
    code = run_cargo_command("zkbinary", "run", {"i": "input.json"}, dev_mode=False)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "./target/release/zkbinary" in args[0]
    assert "run" in args
    assert "-i" in args
    assert "input.json" in args
    assert code.returncode == 0


@pytest.mark.unit
@patch("python.core.utils.helper_functions.subprocess.run")
@patch("python.core.utils.helper_functions.Path.read_text")
def test_run_cargo_command_dev_mode(
    mock_read_text: MagicMock,
    mock_run: MagicMock,
) -> None:
    # Mock pyproject.toml contents so version parsing succeeds
    mock_read_text.return_value = """
    [project]
    version = "1.2.3"
    """

    # Mock subprocess.run success
    mock_run.return_value = MagicMock(returncode=0)

    # Run the function under test

    run_cargo_command("testbin", "compile", dev_mode=True)

    # Extract the actual cargo command used
    args = mock_run.call_args[0][0]

    # Build the expected binary name based on pyproject.toml version
    expected_bin = "testbin_None"

    # Assertions
    assert args[:5] == ["cargo", "run", "--bin", expected_bin, "--release"]
    assert "compile" in args


@pytest.mark.unit
@patch("python.core.utils.helper_functions.Path")
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_cargo_command_fallback_to_cargo_run(
    mock_run: MagicMock,
    mock_path_class: MagicMock,
) -> None:
    """Test that when binary doesn't exist
    and dev_mode=False, it falls back to cargo run."""
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_path_class.return_value = mock_path_instance

    mock_run.return_value = MagicMock(returncode=0)
    run_cargo_command("missingbin", "compile", dev_mode=False)

    args = mock_run.call_args[0][0]
    assert args[:3] == ["cargo", "run", "--bin"]
    assert "missingbin" in args[3]
    assert args[4] == "--release"
    assert "compile" in args


@pytest.mark.unit
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_cargo_command_bool_args(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    run_cargo_command("zkproof", "verify", {"v": True, "json": False, "i": "in.json"})

    args = mock_run.call_args[0][0]
    assert "-v" in args
    assert "-i" in args
    assert "in.json" in args
    assert "-json" in args  # Even though False, it's added


@pytest.mark.unit
@patch(
    "python.core.utils.helper_functions.subprocess.run",
    side_effect=Exception("subprocess failed"),
)
def test_run_cargo_command_raises_on_failure(mock_run: MagicMock) -> None:
    with pytest.raises(Exception, match="subprocess failed"):
        run_cargo_command("failbin", "fail_cmd", {"x": 1})


@pytest.mark.unit
@patch("python.core.utils.helper_functions.Path")
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_command_failure(mock_run: MagicMock, mock_path_class: MagicMock) -> None:
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_class.return_value = mock_path_instance

    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["fakecmd"],
        stderr="boom!",
    )

    with pytest.raises(ProofBackendError) as excinfo:
        run_cargo_command("fakecmd", "type")

    assert "Rust backend error" in str(excinfo.value)
    assert "fakecmd" in str(excinfo.value)
    assert "type" in str(excinfo.value)


# ---------- get_expander_file_paths ----------
@pytest.mark.unit
def test_get_expander_file_paths() -> None:
    name = "model"
    paths = get_expander_file_paths(name)
    assert paths["circuit_file"] == "model_circuit.txt"
    assert paths["witness_file"] == "model_witness.txt"
    assert paths["proof_file"] == "model_proof.txt"


# ---------- compile_circuit ----------
@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_compile_circuit_expander(mock_run: MagicMock) -> None:
    compile_circuit(
        "model",
        "path/to/circuit",
        "path/to/metadata.json",
        "path/to/architecture.json",
        "path/to/w_and_b.json",
        ZKProofSystems.Expander,
    )
    _, kwargs = mock_run.call_args
    assert kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model"
    assert kwargs["args"]["c"] == "path/to/circuit"
    assert kwargs["args"]["m"] == "path/to/metadata.json"
    assert kwargs["args"]["a"] == "path/to/architecture.json"
    assert kwargs["args"]["b"] == "path/to/w_and_b.json"
    mock_run.assert_called_once()


@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_compile_circuit_expander_dev_mode_true(mock_run: MagicMock) -> None:
    compile_circuit(
        "model2",
        "path/to/circuit2",
        "path/to/metadata2.json",
        "path/to/architecture2.json",
        "path/to/w_and_b2.json",
        ZKProofSystems.Expander,
        dev_mode=True,
    )
    _, kwargs = mock_run.call_args
    assert kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model2"
    assert kwargs["args"]["c"] == "path/to/circuit2"
    assert kwargs["args"]["m"] == "path/to/metadata2.json"
    assert kwargs["args"]["a"] == "path/to/architecture2.json"
    assert kwargs["args"]["b"] == "path/to/w_and_b2.json"
    mock_run.assert_called_once()


@pytest.mark.integration
@patch(
    "python.core.utils.helper_functions.run_cargo_command",
    side_effect=ProofBackendError("TEST"),
)
def test_compile_circuit_expander_rust_error(
    mock_run: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(Exception, match="TEST"):
        compile_circuit(
            "model2",
            "path/to/circuit2",
            "path/to/metadata2.json",
            "path/to/architecture2.json",
            "path/to/w_and_b2.json",
            ZKProofSystems.Expander,
            dev_mode=True,
        )
    assert "Warning: Compile operation failed: TEST" in caplog.text
    assert "Using binary: model2" in caplog.text


@pytest.mark.integration
def test_compile_circuit_unknown_raises() -> None:
    with pytest.raises(ProofSystemNotImplementedError):
        compile_circuit("m", "p", "meta.json", "arch.json", "w.json", "unsupported")


# # ---------- generate_witness ----------
@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_witness_expander(mock_run: MagicMock) -> None:
    generate_witness(
        "model",
        "path/to/circuit",
        "witness",
        "input",
        "output",
        "metadata.json",
        ZKProofSystems.Expander,
    )
    _, kwargs = mock_run.call_args
    assert not kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model"
    assert kwargs["args"]["c"] == "path/to/circuit"
    assert kwargs["args"]["w"] == "witness"
    assert kwargs["args"]["i"] == "input"
    assert kwargs["args"]["o"] == "output"
    assert kwargs["args"]["m"] == "metadata.json"
    mock_run.assert_called_once()


@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_witness_expander_dev_mode_true(mock_run: MagicMock) -> None:
    generate_witness(
        "model",
        "path/to/circuit",
        "witness",
        "input",
        "output",
        "metadata.json",
        ZKProofSystems.Expander,
        dev_mode=True,
    )
    _, kwargs = mock_run.call_args
    assert kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model"
    assert kwargs["args"]["c"] == "path/to/circuit"
    assert kwargs["args"]["w"] == "witness"
    assert kwargs["args"]["i"] == "input"
    assert kwargs["args"]["o"] == "output"
    assert kwargs["args"]["m"] == "metadata.json"
    mock_run.assert_called_once()


@pytest.mark.integration
@patch(
    "python.core.utils.helper_functions.run_cargo_command",
    side_effect=ProofBackendError("TEST"),
)
def test_generate_witness_expander_rust_error(
    mock_run: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(Exception, match="TEST"):
        generate_witness(
            "model2",
            "path/to/circuit2",
            "witness",
            "input",
            "output",
            "metadata.json",
            ZKProofSystems.Expander,
            dev_mode=True,
        )
    assert "Warning: Witness generation failed: TEST" in caplog.text


@pytest.mark.unit
def test_generate_witness_unknown_raises() -> None:
    with pytest.raises(ProofSystemNotImplementedError):
        generate_witness(
            "m",
            "p",
            "witness",
            "input",
            "output",
            "metadata.json",
            "unsupported",
        )


# ---------- generate_proof ----------


@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_expander_raw")
@patch(
    "python.core.utils.helper_functions.get_expander_file_paths",
    return_value={"circuit_file": "c", "witness_file": "w", "proof_file": "p"},
)
def test_generate_proof_expander_no_ecc(
    mock_paths: MagicMock,
    mock_exec: MagicMock,
) -> None:
    generate_proof("model", "c", "w", "p", ZKProofSystems.Expander, ecc=False)
    assert mock_exec.call_args[1]["mode"] == ExpanderMode.PROVE
    assert mock_exec.call_args[1]["circuit_file"] == "c"
    assert mock_exec.call_args[1]["witness_file"] == "w"
    assert mock_exec.call_args[1]["proof_file"] == "p"

    assert mock_exec.call_count == 1


@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_proof_expander_with_ecc(mock_run: MagicMock) -> None:
    generate_proof(
        "model",
        "c",
        "w",
        "p",
        "metadata.json",
        ZKProofSystems.Expander,
        ecc=True,
    )
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["binary_name"] == "model"
    assert kwargs["command_type"] == "run_prove_witness"
    assert not kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model"
    assert kwargs["args"]["c"] == "c"
    assert kwargs["args"]["w"] == "w"
    assert kwargs["args"]["p"] == "p"
    assert kwargs["args"]["m"] == "metadata.json"


@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_proof_expander_with_ecc_dev_mode_true(mock_run: MagicMock) -> None:
    generate_proof(
        "model",
        "c",
        "w",
        "p",
        "metadata.json",
        ZKProofSystems.Expander,
        ecc=True,
        dev_mode=True,
    )
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["binary_name"] == "model"
    assert kwargs["command_type"] == "run_prove_witness"
    assert kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model"
    assert kwargs["args"]["c"] == "c"
    assert kwargs["args"]["w"] == "w"
    assert kwargs["args"]["p"] == "p"
    assert kwargs["args"]["m"] == "metadata.json"


@pytest.mark.unit
def test_generate_proof_unknown_raises() -> None:
    with pytest.raises(ProofSystemNotImplementedError):
        generate_proof("m", "p", "w", "proof", "metadata.json", "unsupported")


@pytest.mark.unit
@patch(
    "python.core.utils.helper_functions.run_cargo_command",
    side_effect=ProofBackendError("TEST"),
)
def test_generate_proof_expander_rust_error(
    mock_run: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(Exception, match="TEST"):
        generate_proof(
            "model2",
            "path/to/circuit2",
            "w",
            "p",
            "metadata.json",
            ZKProofSystems.Expander,
            dev_mode=True,
        )
    assert "Warning: Proof generation failed: TEST" in caplog.text


# # ---------- generate_verification ----------
@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_expander_raw")
@patch(
    "python.core.utils.helper_functions.get_expander_file_paths",
    return_value={"circuit_file": "c", "witness_file": "w", "proof_file": "p"},
)
def test_generate_verify_expander_no_ecc(
    mock_paths: MagicMock,
    mock_exec: MagicMock,
) -> None:
    generate_verification(
        "model",
        "c",
        "i",
        "o",
        "w",
        "p",
        ZKProofSystems.Expander,
        ecc=False,
    )
    assert mock_exec.call_args[1]["mode"] == ExpanderMode.VERIFY
    assert mock_exec.call_args[1]["circuit_file"] == "c"
    assert mock_exec.call_args[1]["witness_file"] == "w"
    assert mock_exec.call_args[1]["proof_file"] == "p"

    assert mock_exec.call_count == 1


@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_verify_expander_with_ecc(mock_run: MagicMock) -> None:
    generate_verification(
        "model",
        "c",
        "i",
        "o",
        "w",
        "p",
        "metadata.json",
        ZKProofSystems.Expander,
        ecc=True,
    )
    mock_run.assert_called_once()

    _, kwargs = mock_run.call_args
    assert kwargs["binary_name"] == "model"
    assert kwargs["command_type"] == "run_gen_verify"
    assert not kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model"
    assert kwargs["args"]["c"] == "c"
    assert kwargs["args"]["w"] == "w"
    assert kwargs["args"]["p"] == "p"
    assert kwargs["args"]["i"] == "i"
    assert kwargs["args"]["o"] == "o"
    assert kwargs["args"]["m"] == "metadata.json"
    mock_run.assert_called_once()


@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_verify_expander_with_ecc_dev_mode_true(mock_run: MagicMock) -> None:
    generate_verification(
        "model",
        "c",
        "i",
        "o",
        "w",
        "p",
        "metadata.json",
        ZKProofSystems.Expander,
        ecc=True,
        dev_mode=True,
    )
    _, kwargs = mock_run.call_args
    assert kwargs["binary_name"] == "model"
    assert kwargs["command_type"] == "run_gen_verify"
    assert kwargs["dev_mode"]
    assert kwargs["args"]["n"] == "model"
    assert kwargs["args"]["c"] == "c"
    assert kwargs["args"]["w"] == "w"
    assert kwargs["args"]["p"] == "p"
    assert kwargs["args"]["i"] == "i"
    assert kwargs["args"]["o"] == "o"
    assert kwargs["args"]["m"] == "metadata.json"
    mock_run.assert_called_once()


@pytest.mark.unit
def test_generate_verify_unknown_raises() -> None:
    with pytest.raises(ProofSystemNotImplementedError):
        generate_verification(
            "model",
            "c",
            "i",
            "o",
            "w",
            "p",
            "metadata.json",
            "unsupported",
        )


@pytest.mark.unit
def test_proof_system_not_implemented_full_process() -> None:
    with pytest.raises(
        ProofSystemNotImplementedError,
        match="Proof system UnknownProofSystem not implemented",
    ):
        generate_verification(
            "model",
            "c",
            "i",
            "o",
            "w",
            "p",
            "metadata.json",
            "UnknownProofSystem",
        )
    with pytest.raises(
        ProofSystemNotImplementedError,
        match="Proof system UnknownProofSystem not implemented",
    ):
        generate_proof("m", "p", "w", "proof", "metadata.json", "UnknownProofSystem")
    with pytest.raises(
        ProofSystemNotImplementedError,
        match="Proof system UnknownProofSystem not implemented",
    ):
        generate_witness(
            "m",
            "p",
            "witness",
            "input",
            "output",
            "metadata.json",
            "UnknownProofSystem",
        )
    with pytest.raises(
        ProofSystemNotImplementedError,
        match="Proof system UnknownProofSystem not implemented",
    ):
        compile_circuit(
            "model",
            "path/to/circuit",
            "metadata.json",
            "architecture.json",
            "w_and_b.json",
            "UnknownProofSystem",
        )


@pytest.mark.unit
@patch(
    "python.core.utils.helper_functions.run_cargo_command",
    side_effect=ProofBackendError("TEST"),
)
def test_generate_verify_expander_rust_error(
    mock_run: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(Exception, match="TEST"):
        generate_verification(
            "model",
            "c",
            "i",
            "o",
            "w",
            "p",
            "metadata.json",
            ZKProofSystems.Expander,
            dev_mode=True,
        )
    assert "Warning: Verification generation failed: TEST" in caplog.text


# # ---------- run_end_to_end ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.generate_verification")
@patch("python.core.utils.helper_functions.generate_proof")
@patch("python.core.utils.helper_functions.generate_witness")
@patch("python.core.utils.helper_functions.compile_circuit")
def test_run_end_to_end_calls_all(
    mock_compile: MagicMock,
    mock_witness: MagicMock,
    mock_proof: MagicMock,
    mock_verify: MagicMock,
) -> None:
    run_end_to_end("m", "m_circuit.txt", "i.json", "o.json")
    mock_compile.assert_called_once_with(
        "m",
        "m_circuit.txt",
        "m_circuit_metadata.json",
        "m_circuit_architecture.json",
        "m_circuit_wandb.json",
        ZKProofSystems.Expander,
        False,  # noqa: FBT003
    )
    mock_witness.assert_called_once_with(
        "m",
        "m_circuit.txt",
        "m_circuit_witness.txt",
        "i.json",
        "o.json",
        "m_circuit_metadata.json",
        ZKProofSystems.Expander,
        False,  # noqa: FBT003
    )
    mock_proof.assert_called_once_with(
        "m",
        "m_circuit.txt",
        "m_circuit_witness.txt",
        "m_circuit_proof.bin",
        "m_circuit_metadata.json",
        ZKProofSystems.Expander,
        False,  # noqa: FBT003
        True,  # noqa: FBT003
    )
    mock_verify.assert_called_once_with(
        "m",
        "m_circuit.txt",
        "i.json",
        "o.json",
        "m_circuit_witness.txt",
        "m_circuit_proof.bin",
        "m_circuit_metadata.json",
        ZKProofSystems.Expander,
        False,  # noqa: FBT003
        True,  # noqa: FBT003
    )


@pytest.mark.unit
@patch("python.core.utils.helper_functions.generate_verification")
@patch("python.core.utils.helper_functions.generate_proof")
@patch("python.core.utils.helper_functions.generate_witness")
@patch("python.core.utils.helper_functions.compile_circuit")
def test_unknown_proof_system_errors_end_to_end(
    mock_compile: MagicMock,
    mock_witness: MagicMock,
    mock_proof: MagicMock,
    mock_verify: MagicMock,
) -> None:
    with pytest.raises(
        ProofSystemNotImplementedError,
        match="Proof system UnknownProofSystem not implemented",
    ):
        run_end_to_end("m", "m_circuit.txt", "i.json", "o.json", "UnknownProofSystem")


# # ---------- get_files ----------
@pytest.mark.unit
def test_get_files() -> None:
    folders = {
        "input": "inputs",
        "proof": "proofs",
        "temp": "tmp",
        "circuit": "circuits",
        "weights": "weights",
        "output": "out",
        "quantized_model": "quantized_models",
    }
    paths = get_files("model", ZKProofSystems.Expander, folders)
    assert paths["input_file"].endswith("model_input.json")


@pytest.mark.unit
def test_get_files_non_proof_system() -> None:

    folders = {
        "input": "inputs",
        "proof": "proofs",
        "temp": "tmp",
        "circuit": "circuits",
        "weights": "weights",
        "output": "out",
        "quantized_model": "quantized_models",
    }
    fake_proof_system = "unknown"
    with pytest.raises(
        ProofSystemNotImplementedError,
        match=f"Proof system {fake_proof_system} not implemented",
    ):
        get_files("model", fake_proof_system, folders)

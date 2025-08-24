import subprocess
import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from python.core.utils.helper_functions import compile_circuit, compute_and_store_output, create_folder, generate_proof, generate_verification, generate_witness, get_expander_file_paths, get_files, prepare_io_files, run_end_to_end, to_json, read_from_json, run_cargo_command
from python.core.utils.helper_functions import RunType, ZKProofSystems


# ---------- compute_and_store_output ----------
@pytest.mark .unit
@patch("python.core.utils.helper_functions.os.makedirs")
@patch("python.core.utils.helper_functions.os.path.exists", return_value=False)
@patch("python.core.utils.helper_functions.json.dump")
@patch("python.core.utils.helper_functions.open", new_callable=mock_open)
def test_compute_and_store_output_saves(mock_file, mock_dump, mock_exists, mock_mkdirs):
    class Dummy:
        name = "test"
        temp_folder = "temp_test"

        @compute_and_store_output
        def get_outputs(self):
            return {"out": 123}

    d = Dummy()
    result = d.get_outputs()

    mock_mkdirs.assert_called_once()
    mock_dump.assert_called_once()
    assert result == {"out": 123}

@pytest.mark.unit
@patch("python.core.utils.helper_functions.os.path.exists", return_value=True)
@patch("python.core.utils.helper_functions.open", new_callable=mock_open, read_data='{"out": 456}')
@patch("python.core.utils.helper_functions.json.load", return_value={"out": 456})
def test_compute_and_store_output_loads_from_cache(mock_load, mock_file, mock_exists):
    class Dummy:
        name = "test"
        temp_folder = "temp_test"

        @compute_and_store_output
        def get_outputs(self):
            return {"out": 999}  # should not run

    d = Dummy()
    output = d.get_outputs()
    assert output == {"out": 456}

@pytest.mark.unit
@patch("python.core.utils.helper_functions.os.path.exists", return_value=True)
@patch("python.core.utils.helper_functions.open", new_callable=mock_open)
@patch("python.core.utils.helper_functions.json.load", side_effect=json.JSONDecodeError("msg", "doc", 0))
@patch("python.core.utils.helper_functions.json.dump")
def test_compute_and_store_output_on_json_error(mock_dump, mock_load, mock_file, mock_exists):
    class Dummy:
        name = "bad"
        temp_folder = "temp_test"

        @compute_and_store_output
        def get_outputs(self):
            return {"fallback": True}

    d = Dummy()
    output = d.get_outputs()
    assert output == {"fallback": True}


# ---------- prepare_io_files ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.get_files", return_value={
    "witness_file": "witness.wtns",
    "input_file": "input.json",
    "proof_path": "proof.json",
    "public_path": "public.json",
    "verification_key": "vk.key",
    "circuit_name": "test_circuit",
    "weights_path": "weights.json",
    "output_file": "out.json"
})
@patch("python.core.utils.helper_functions.to_json")
@patch("python.core.utils.helper_functions.os.path.splitext", return_value=("model", ".onnx"))
@patch("python.core.utils.helper_functions.open", new_callable=mock_open)
def test_prepare_io_files_runs_func(mock_file, mock_splitext, mock_json, mock_get_files):
    class Dummy:
        name = "model"
        input_shape = (1, 4)
        scale_base = 10
        scale_exponent = 1
        def __init__(self):
            self.get_inputs = lambda: 1
            self.get_outputs = lambda x=None: 2
            self.get_inputs_from_file = lambda file_name, is_scaled=True: 3
            self.format_inputs = lambda x: {"input": x}
            self.format_outputs = lambda x: {"output": x}
            self.load_quantized_model = MagicMock()
            self.get_weights = lambda: {"weights": [1, 2]}
            self.save_quantized_model = MagicMock()
            self.get_model_and_quantize = MagicMock()

        @prepare_io_files
        def base_testing(self, run_type, witness_file, input_file, proof_path, public_path,
                         verification_key, circuit_name, weights_path, output_file,
                         proof_system, **kwargs):
            assert run_type == RunType.GEN_WITNESS
            return {"test": True}

    d = Dummy()
    result = d.base_testing(run_type=RunType.GEN_WITNESS, write_json=True)
    assert result == {"test": True}
    assert d._file_info["output_file"] == "out.json"
    assert d._file_info["weights_path"] == "weights.json"



# ---------- to_json ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.open", new_callable=mock_open)
@patch("python.core.utils.helper_functions.json.dump")
def test_to_json_saves_json(mock_dump, mock_file):
    data = {"a": 1}
    to_json(data, "output.json")
    mock_file.assert_called_once_with("output.json", "w")
    mock_dump.assert_called_once_with(data, mock_file())


# ---------- read_from_json ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.open", new_callable=mock_open, read_data='{"x": 42}')
@patch("python.core.utils.helper_functions.json.load", return_value={"x": 42})
def test_read_from_json_loads_json(mock_load, mock_file):
    result = read_from_json("input.json")
    mock_file.assert_called_once_with("input.json")
    mock_load.assert_called_once()
    assert result == {"x": 42}


# ---------- run_cargo_command ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_cargo_command_normal(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="ok")
    code = run_cargo_command("zkbinary", "run", {"i": "input.json"}, dev_mode=False)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "./target/release/zkbinary"
    assert "run" in args
    assert "-i" in args and "input.json" in args
    assert code.returncode == 0

@pytest.mark.unit
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_cargo_command_dev_mode(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    run_cargo_command("testbin", "compile", dev_mode=True)

    args = mock_run.call_args[0][0]
    assert args[:5] == ["cargo", "run", "--bin", "testbin", "--release"]
    assert "compile" in args

@pytest.mark.unit
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_cargo_command_bool_args(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    run_cargo_command("zkproof", "verify", {"v": True, "json": False, "i": "in.json"})

    args = mock_run.call_args[0][0]
    assert "-v" in args
    assert "-i" in args
    assert "in.json" in args
    assert "-json" in args  # Even though False, it's added

@pytest.mark.unit
@patch("python.core.utils.helper_functions.subprocess.run", side_effect=Exception("subprocess failed"))
def test_run_cargo_command_raises_on_failure(mock_run):
    with pytest.raises(Exception, match="subprocess failed"):
        run_cargo_command("failbin", "fail_cmd", {"x": 1})

@pytest.mark.unit
@patch("python.core.utils.helper_functions.subprocess.run")
def test_run_command_failure(mock_run):
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["fakecmd"],
        stderr="boom!"
    )

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_cargo_command("fakecmd", "type")

    assert excinfo.value.returncode == 1

# ---------- get_expander_file_paths ----------
@pytest.mark.unit
def test_get_expander_file_paths():
    name = "model"
    paths = get_expander_file_paths(name)
    assert paths["circuit_file"] == "model_circuit.txt"
    assert paths["witness_file"] == "model_witness.txt"
    assert paths["proof_file"] == "model_proof.txt"


# ---------- compile_circuit ----------
@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_compile_circuit_expander(mock_run):
    compile_circuit("model", "path/to/circuit", ZKProofSystems.Expander)
    args = mock_run.call_args[0][2]
    assert args["n"] == "model"
    assert args["c"] == "path/to/circuit"
    assert mock_run.call_args[0][3] == True

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_compile_circuit_expander_dev_mode_true(mock_run):
    compile_circuit("model2", "path/to/circuit2", ZKProofSystems.Expander, True)
    args = mock_run.call_args[0][2]
    assert args["n"] == "model2"
    assert args["c"] == "path/to/circuit2"
    assert mock_run.call_args[0][3] == True

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command", side_effect = Exception("TEST"))
def test_compile_circuit_expander_rust_error(mock_run, capfd):
    compile_circuit("model2", "path/to/circuit2", ZKProofSystems.Expander, True)
    out, err = capfd.readouterr()
    assert "Warning: Compile operation failed: TEST" in out
    assert "Using binary: model2" in out


@pytest.mark.integration
def test_compile_circuit_unknown_raises():
    with pytest.raises(NotImplementedError):
        compile_circuit("m", "p", "unsupported")


# # ---------- generate_witness ----------
@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_witness_expander(mock_run):
    generate_witness("model", "path/to/circuit", "witness", "input", "output", ZKProofSystems.Expander)
    args = mock_run.call_args[0][2]
    assert args["n"] == "model"
    assert args["c"] == "path/to/circuit"
    assert args["w"] == "witness"
    assert args["i"] == "input"
    assert args["o"] == "output"
    assert mock_run.call_args[0][3] == False

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_witness_expander_dev_mode_true(mock_run):
    generate_witness("model", "path/to/circuit", "witness", "input", "output", ZKProofSystems.Expander, True)
    args = mock_run.call_args[0][2]
    assert args["n"] == "model"
    assert args["c"] == "path/to/circuit"
    assert args["w"] == "witness"
    assert args["i"] == "input"
    assert args["o"] == "output"
    assert mock_run.call_args[0][3] == True

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command", side_effect = Exception("TEST"))
def test_generate_witness_expander_rust_error(mock_run, capfd):
    generate_witness("model2", "path/to/circuit2", "witness", "input", "output", ZKProofSystems.Expander, True)
    out, err = capfd.readouterr()
    assert "Warning: Witness generation failed: TEST" in out


@pytest.mark.unit
def test_generate_witness_unknown_raises():
    with pytest.raises(NotImplementedError):
        generate_witness("m", "p","witness", "input", "output",  "unsupported")


# ---------- generate_proof ----------

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_expander_raw")
@patch("python.core.utils.helper_functions.get_expander_file_paths", return_value={
    "circuit_file": "c", "witness_file": "w", "proof_file": "p"
})
def test_generate_proof_expander_no_ecc(mock_paths, mock_exec):
    generate_proof("model", "cp", "w", "p", ZKProofSystems.Expander, ecc=False)
    # print(mock_exec.call_args[1])
    assert mock_exec.call_args[1]['mode'] == 'prove'
    assert mock_exec.call_args[1]['circuit_file'] == 'cp'
    assert mock_exec.call_args[1]['witness_file'] == 'w'
    assert mock_exec.call_args[1]['proof_file'] == 'p'

    assert mock_exec.call_count == 1

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_proof_expander_with_ecc(mock_run):
    generate_proof("model", "c", "w", "p", ZKProofSystems.Expander, ecc=True)
    mock_run.assert_called_once()
    args = mock_run.call_args[0]
    assert args[0] == 'model'
    assert args[1] == 'run_prove_witness'
    assert args[3] == False
    assert args[2]['n'] == 'model'
    assert args[2]['c'] == 'c'
    assert args[2]['w'] == 'w'
    assert args[2]['p'] == 'p'

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_proof_expander_with_ecc_dev_mode_true(mock_run):
    generate_proof("model", "c", "w", "p", ZKProofSystems.Expander, ecc=True, dev_mode=True)
    mock_run.assert_called_once()
    args = mock_run.call_args[0]
    assert args[0] == 'model'
    assert args[1] == 'run_prove_witness'
    assert args[3] == True
    assert args[2]['n'] == 'model'
    assert args[2]['c'] == 'c'
    assert args[2]['w'] == 'w'
    assert args[2]['p'] == 'p'

@pytest.mark.unit
def test_generate_proof_unknown_raises():
    with pytest.raises(NotImplementedError):
        generate_proof("m", "p", "w", "proof", "unsupported")

@pytest.mark.unit
@patch("python.core.utils.helper_functions.run_cargo_command", side_effect = Exception("TEST"))
def test_generate_proof_expander_rust_error(mock_run, capfd):
    generate_proof("model2", "path/to/circuit2", "w", 'p', ZKProofSystems.Expander, True)
    out, err = capfd.readouterr()
    assert "Warning: Proof generation failed: TEST" in out




# # ---------- generate_verification ----------
@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_expander_raw")
@patch("python.core.utils.helper_functions.get_expander_file_paths", return_value={
    "circuit_file": "c", "witness_file": "w", "proof_file": "p"
})
def test_generate_verify_expander_no_ecc(mock_paths, mock_exec):
    generate_verification("model", "cp", "i", "o", "w", "p", ZKProofSystems.Expander, ecc=False)
    assert mock_exec.call_args[1]['mode'] == 'verify'
    assert mock_exec.call_args[1]['circuit_file'] == 'cp'
    assert mock_exec.call_args[1]['witness_file'] == 'w'
    assert mock_exec.call_args[1]['proof_file'] == 'p'

    assert mock_exec.call_count == 1

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_verify_expander_with_ecc(mock_run):
    generate_verification("model", "cp", "i", "o", "w", "p", ZKProofSystems.Expander, ecc=True)
    mock_run.assert_called_once()
    args = mock_run.call_args[0]
    assert args[0] == 'model'
    assert args[1] == 'run_gen_verify'
    assert args[3] == False
    assert args[2]['n'] == 'model'
    assert args[2]['c'] == 'cp'
    assert args[2]['w'] == 'w'
    assert args[2]['p'] == 'p'
    assert args[2]['i'] == 'i'
    assert args[2]['o'] == 'o'

@pytest.mark.integration
@patch("python.core.utils.helper_functions.run_cargo_command")
def test_generate_verify_expander_with_ecc_dev_mode_true(mock_run):
    generate_verification("model", "cp", "i", "o", "w", "p",  ZKProofSystems.Expander, ecc=True, dev_mode=True)
    mock_run.assert_called_once()
    args = mock_run.call_args[0]
    assert args[0] == 'model'
    assert args[1] == 'run_gen_verify'
    assert args[3] == True
    assert args[2]['n'] == 'model'
    assert args[2]['c'] == 'cp'
    assert args[2]['w'] == 'w'
    assert args[2]['p'] == 'p'
    assert args[2]['i'] == 'i'
    assert args[2]['o'] == 'o'

@pytest.mark.unit
def test_generate_verify_unknown_raises():
    with pytest.raises(NotImplementedError):
        generate_verification("model", "cp", "i", "o", "w", "p",  "unsupported")

@pytest.mark.unit
def test_proof_system_not_implemented_full_process():
    with pytest.raises(NotImplementedError, match="Proof system UnknownProofSystem not implemented"):
        generate_verification("model", "cp", "i", "o", "w", "p", "UnknownProofSystem")
    with pytest.raises(NotImplementedError, match="Proof system UnknownProofSystem not implemented"):
        generate_proof("m", "p", "w", "proof", "UnknownProofSystem")
    with pytest.raises(NotImplementedError, match="Proof system UnknownProofSystem not implemented"):
        generate_witness("m", "p","witness", "input", "output",  "UnknownProofSystem")
    with pytest.raises(NotImplementedError, match="Proof system UnknownProofSystem not implemented"):
        compile_circuit("model", "path/to/circuit", "UnknownProofSystem")

@pytest.mark.unit
@patch("python.core.utils.helper_functions.run_cargo_command", side_effect = Exception("TEST"))
def test_generate_verify_expander_rust_error(mock_run, capfd):
    generate_verification("model", "cp", "i", "o", "w", "p", ZKProofSystems.Expander, True)
    out, err = capfd.readouterr()
    assert "Warning: Verification generation failed: TEST" in out



# # ---------- run_end_to_end ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.generate_verification")
@patch("python.core.utils.helper_functions.generate_proof")
@patch("python.core.utils.helper_functions.generate_witness")
@patch("python.core.utils.helper_functions.compile_circuit")
def test_run_end_to_end_calls_all(mock_compile, mock_witness, mock_proof, mock_verify):
    run_end_to_end("m", "m_circuit.txt", "i.json", "o.json")
    mock_compile.assert_called_once()
    mock_witness.assert_called_once()
    mock_proof.assert_called_once()
    mock_verify.assert_called_once()

@pytest.mark.unit
@patch("python.core.utils.helper_functions.generate_verification")
@patch("python.core.utils.helper_functions.generate_proof")
@patch("python.core.utils.helper_functions.generate_witness")
@patch("python.core.utils.helper_functions.compile_circuit")
def test_circom_proof_system_errors_end_to_end(mock_compile, mock_witness, mock_proof, mock_verify):
    with pytest.raises(NotImplementedError, match="Proof system UnknownProofSystem not implemented"):
        run_end_to_end("m", "m_circuit.txt", "i.json", "o.json", "UnknownProofSystem")
    
    


# # ---------- get_files / create_folder ----------
@pytest.mark.unit
@patch("python.core.utils.helper_functions.create_folder")
def test_get_files_and_create(mock_create):
    folders = {
        "input": "inputs",
        "proof": "proofs",
        "temp": "tmp",
        "circuit": "circuits",
        "weights": "weights",
        "output": "out",
        "quantized_model": "quantized_models"
    }
    paths = get_files("model", ZKProofSystems.Expander, folders)
    assert paths["input_file"].endswith("model_input.json")
    assert mock_create.call_count == len(folders)

@pytest.mark.unit
@patch("python.core.utils.helper_functions.create_folder")
def test_get_files_non_proof_system(mock_create):
    folders = {
        "input": "inputs",
        "proof": "proofs",
        "temp": "tmp",
        "circuit": "circuits",
        "weights": "weights",
        "output": "out",
        "quantized_model": "quantized_models"
    }
    fake_proof_system = "unknown"
    with pytest.raises(NotImplementedError, match=f"Proof system {fake_proof_system} not implemented"):
        get_files("model", fake_proof_system, folders)

@pytest.mark.unit
@patch("python.core.utils.helper_functions.os.makedirs")
@patch("python.core.utils.helper_functions.os.path.exists", return_value=False)
def test_create_folder_creates(mock_exists, mock_mkdir):
    create_folder("new_folder")
    mock_mkdir.assert_called_once_with("new_folder")

@pytest.mark.unit
@patch("python.core.utils.helper_functions.os.makedirs")
@patch("python.core.utils.helper_functions.os.path.exists", return_value=True)
def test_create_folder_skips_existing(mock_exists, mock_mkdir):
    create_folder("existing")
    mock_mkdir.assert_not_called()

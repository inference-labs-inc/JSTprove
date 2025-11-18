import re
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from python.core.utils.errors import ShapeMismatchError

sys.modules.pop("python.core.circuits.base", None)


with (
    patch(
        "python.core.utils.helper_functions.compute_and_store_output",
        lambda x: x,
    ),
    patch(
        "python.core.utils.helper_functions.prepare_io_files",
        lambda f: f,
    ),
):  # MUST BE BEFORE THE UUT GETS IMPORTED ANYWHERE!
    from python.core.circuits.base import (
        Circuit,
        CircuitExecutionConfig,
        RunType,
        ZKProofSystems,
    )
    from python.core.circuits.errors import (
        CircuitConfigurationError,
        CircuitFileError,
        CircuitInputError,
        CircuitProcessingError,
        CircuitRunError,
        WitnessMatchError,
    )


# ---------- Test __init__ ----------
@pytest.mark.unit
def test_circuit_init_defaults() -> None:
    c = Circuit()
    assert c.input_folder == "inputs"
    assert c.proof_folder == "analysis"
    assert c.temp_folder == "temp"
    assert c.circuit_folder == ""
    assert c.weights_folder == "weights"
    assert c.output_folder == "output"
    assert c.proof_system == ZKProofSystems.Expander
    assert c._file_info is None
    assert c.required_keys is None


@pytest.mark.unit
def test_circuit_execution_config_with_new_paths() -> None:
    config = CircuitExecutionConfig(
        circuit_name="test_circuit",
        metadata_path="meta.json",
        architecture_path="arch.json",
        w_and_b_path="weights.json",
    )
    assert config.circuit_name == "test_circuit"
    assert config.metadata_path == "meta.json"
    assert config.architecture_path == "arch.json"
    assert config.w_and_b_path == "weights.json"


# ---------- Test parse_inputs ----------
@pytest.mark.unit
def test_parse_inputs_missing_required_keys() -> None:
    c = Circuit()
    c.required_keys = ["x", "y"]
    with pytest.raises(CircuitInputError, match="Missing required parameter: 'x'"):
        c.parse_inputs(y=5)


@pytest.mark.unit
def test_parse_inputs_type_check() -> None:
    c = Circuit()
    c.required_keys = ["x"]
    with pytest.raises(
        CircuitInputError,
        match="Parameter 'x' must be an int or list of ints",
    ):
        c.parse_inputs(x="not-an-int")


@pytest.mark.unit
def test_parse_inputs_success_int() -> None:
    c = Circuit()
    c.required_keys = ["x", "y"]
    x = 10
    y = 20

    c.parse_inputs(x=x, y=y)

    assert c.x == x
    assert c.y == y


@pytest.mark.unit
def test_parse_inputs_success_list() -> None:
    c = Circuit()
    c.required_keys = ["arr"]
    c.parse_inputs(arr=[1, 2, 3])
    assert c.arr == [1, 2, 3]


@pytest.mark.unit
def test_parse_inputs_required_keys_none() -> None:
    c = Circuit()
    with pytest.raises(CircuitConfigurationError):
        c.parse_inputs()


# ---------- Test Not Implemented --------------
@pytest.mark.unit
def test_get_inputs_not_implemented() -> None:
    c = Circuit()
    with pytest.raises(NotImplementedError, match="get_inputs must be implemented"):
        c.get_inputs()


@pytest.mark.unit
def test_get_outputs_not_implemented() -> None:
    c = Circuit()
    with pytest.raises(NotImplementedError, match="get_outputs must be implemented"):
        c.get_outputs()


# ---------- Test parse_proof_run_type ----------


@pytest.mark.unit
@patch("python.core.circuits.base.compile_circuit")
@patch("python.core.circuits.base.generate_witness")
@patch("python.core.circuits.base.generate_proof")
@patch("python.core.circuits.base.generate_verification")
@patch("python.core.circuits.base.run_end_to_end")
def test_parse_proof_dispatch_logic(
    mock_end_to_end: MagicMock,
    mock_verify: MagicMock,
    mock_proof: MagicMock,
    mock_witness: MagicMock,
    mock_compile: MagicMock,
) -> None:
    c = Circuit()

    # Mock internal preprocessing methods
    c._compile_preprocessing = MagicMock()
    c._gen_witness_preprocessing = MagicMock(return_value="i")
    c.adjust_inputs = MagicMock(return_value="i")
    c.rename_inputs = MagicMock(return_value="i")
    c.prepare_inputs_for_verification = MagicMock(return_value="i")

    c.load_and_compare_witness_to_io = MagicMock(return_value="True")

    # COMPILE_CIRCUIT
    config_compile = CircuitExecutionConfig(
        witness_file="w",
        input_file="i",
        proof_file="p",
        public_path="pub",
        verification_key="vk",
        circuit_name="circuit",
        circuit_path="path",
        proof_system=ZKProofSystems.Expander,
        output_file="out",
        metadata_path="metadata",
        architecture_path="architecture",
        w_and_b_path="w_and_b",
        quantized_path="q",
        run_type=RunType.COMPILE_CIRCUIT,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )
    c.parse_proof_run_type(config_compile)
    mock_compile.assert_called_once()
    c._compile_preprocessing.assert_called_once_with(
        metadata_path="metadata",
        architecture_path="architecture",
        w_and_b_path="w_and_b",
        quantized_path="q",
    )
    _, kwargs = mock_compile.call_args
    assert kwargs == {
        "circuit_name": "circuit",
        "circuit_path": "path",
        "proof_system": ZKProofSystems.Expander,
        "dev_mode": False,
        "bench": False,
        "architecture_path": "architecture",
        "metadata_path": "metadata",
        "w_and_b_path": "w_and_b",
    }

    # GEN_WITNESS
    config_witness = CircuitExecutionConfig(
        witness_file="w",
        input_file="i",
        proof_file="p",
        public_path="pub",
        verification_key="vk",
        circuit_name="circuit",
        circuit_path="path",
        proof_system=ZKProofSystems.Expander,
        output_file="out",
        metadata_path="metadata",
        architecture_path="architecture",
        w_and_b_path="w_and_b",
        quantized_path="q",
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )
    c.parse_proof_run_type(config_witness)
    mock_witness.assert_called_once()
    c._gen_witness_preprocessing.assert_called()
    _, kwargs = mock_witness.call_args
    assert kwargs == {
        "circuit_name": "circuit",
        "circuit_path": "path",
        "witness_file": "w",
        "input_file": "i",
        "output_file": "out",
        "proof_system": ZKProofSystems.Expander,
        "dev_mode": False,
        "bench": False,
        "metadata_path": "metadata",
    }

    # PROVE_WITNESS
    config_prove = CircuitExecutionConfig(
        witness_file="w",
        input_file="i",
        proof_file="p",
        public_path="pub",
        verification_key="vk",
        circuit_name="circuit",
        circuit_path="path",
        proof_system=ZKProofSystems.Expander,
        output_file="out",
        metadata_path="metadata",
        architecture_path="architecture",
        w_and_b_path="w_and_b",
        quantized_path="q",
        run_type=RunType.PROVE_WITNESS,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )
    c.parse_proof_run_type(config_prove)
    mock_proof.assert_called_once()
    _, kwargs = mock_proof.call_args

    assert kwargs == {
        "circuit_name": "circuit",
        "circuit_path": "path",
        "witness_file": "w",
        "proof_file": "p",
        "proof_system": ZKProofSystems.Expander,
        "dev_mode": False,
        "ecc": True,
        "bench": False,
        "metadata_path": "metadata",
    }

    # GEN_VERIFY
    config_verify = CircuitExecutionConfig(
        witness_file="w",
        input_file="i",
        proof_file="p",
        public_path="pub",
        verification_key="vk",
        circuit_name="circuit",
        circuit_path="path",
        proof_system=ZKProofSystems.Expander,
        output_file="out",
        metadata_path="metadata",
        architecture_path="architecture",
        w_and_b_path="w_and_b",
        quantized_path="q",
        run_type=RunType.GEN_VERIFY,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )
    c.parse_proof_run_type(config_verify)
    mock_verify.assert_called_once()
    _, kwargs = mock_verify.call_args
    assert kwargs == {
        "circuit_name": "circuit",
        "circuit_path": "path",
        "input_file": "i",
        "output_file": "out",
        "witness_file": "w",
        "proof_file": "p",
        "proof_system": ZKProofSystems.Expander,
        "dev_mode": False,
        "ecc": True,
        "bench": False,
        "metadata_path": "metadata",
    }

    # END_TO_END
    config_end_to_end = CircuitExecutionConfig(
        witness_file="w",
        input_file="i",
        proof_file="p",
        public_path="pub",
        verification_key="vk",
        circuit_name="circuit",
        circuit_path="path",
        proof_system=ZKProofSystems.Expander,
        output_file="out",
        metadata_path="metadata",
        architecture_path="architecture",
        w_and_b_path="w_and_b",
        quantized_path="q",
        run_type=RunType.END_TO_END,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )
    c.parse_proof_run_type(config_end_to_end)

    preprocess_call_count = 2

    mock_end_to_end.assert_called_once()
    assert c._compile_preprocessing.call_count >= preprocess_call_count
    assert c._gen_witness_preprocessing.call_count >= preprocess_call_count


# ---------- Test new methods for metadata, architecture, w_and_b ----------
@pytest.mark.unit
def test_get_metadata_default() -> None:
    c = Circuit()
    assert c.get_metadata() == {}


@pytest.mark.unit
def test_get_architecture_default() -> None:
    c = Circuit()
    assert c.get_architecture() == {}


@pytest.mark.unit
def test_get_w_and_b_default() -> None:
    c = Circuit()
    assert c.get_w_and_b() == {}


# ---------- Optional: test get_weights ----------
@pytest.mark.unit
def test_get_weights_default() -> None:
    c = Circuit()
    assert c.get_weights() == {}


@pytest.mark.unit
def test_get_inputs_from_file() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 2
    with patch(
        "python.core.circuits.base.read_from_json",
        return_value={"input": [1, 2, 3, 4]},
    ):
        x = c.get_inputs_from_file("", is_scaled=True)
        assert x == {"input": [1, 2, 3, 4]}

        y = c.get_inputs_from_file("", is_scaled=False)
        assert y == {"input": [4, 8, 12, 16]}


@pytest.mark.unit
def test_get_inputs_from_file_multiple_inputs() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 2
    with patch(
        "python.core.circuits.base.read_from_json",
        return_value={"input": [1, 2, 3, 4], "nonce": 25},
    ):
        x = c.get_inputs_from_file("", is_scaled=True)
        assert x == {"input": [1, 2, 3, 4], "nonce": 25}

        y = c.get_inputs_from_file("", is_scaled=False)
        assert y == {"input": [4, 8, 12, 16], "nonce": 100}


@pytest.mark.unit
def test_get_inputs_from_file_dne() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 2
    with pytest.raises(CircuitFileError, match="Failed to read input file"):
        c.get_inputs_from_file("this_file_should_not_exist_12345.json", is_scaled=True)


@pytest.mark.unit
def test_format_outputs() -> None:
    c = Circuit()
    out = c.format_outputs([10, 15, 20])
    assert out == {"output": [10, 15, 20]}


# ---------- _gen_witness_preprocessing ----------
@pytest.mark.unit
@patch("python.core.circuits.base.to_json")
def test_gen_witness_preprocessing_write_json_true(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "quant.pt"}
    c.load_quantized_model = MagicMock()
    c.get_inputs = MagicMock(return_value="inputs")
    c.get_outputs = MagicMock(return_value="outputs")
    c.format_inputs = MagicMock(return_value={"input": 1})
    c.format_outputs = MagicMock(return_value={"output": 2})

    c._gen_witness_preprocessing(
        "in.json",
        "out.json",
        None,
        write_json=True,
        is_scaled=True,
    )

    c.load_quantized_model.assert_called_once_with("quant.pt")
    c.get_inputs.assert_called_once()
    c.get_outputs.assert_called_once_with("inputs")
    mock_to_json.assert_any_call({"input": 1}, "in.json")
    mock_to_json.assert_any_call({"output": 2}, "out.json")


@pytest.mark.unit
def test_gen_witness_preprocessing_write_json_false(tmp_path: Path) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "quant.pt"}

    # Mock all method calls used by _gen_witness_preprocessing
    c.load_quantized_model = MagicMock()
    c._read_from_json_safely = MagicMock(return_value={"mock": "inputs"})
    c.scale_inputs_only = MagicMock(return_value={"scaled": "inputs"})
    c.reshape_inputs_for_inference = MagicMock(return_value={"inference": "inputs"})
    c.reshape_inputs_for_circuit = MagicMock(return_value={"input": [1, 2, 3]})
    c._to_json_safely = MagicMock()
    c.get_outputs = MagicMock(return_value={"raw_output": 123})
    c.format_outputs = MagicMock(return_value={"formatted_output": 999})

    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"

    result = c._gen_witness_preprocessing(
        str(input_path),
        str(output_path),
        None,
        write_json=False,
        is_scaled=False,
    )

    # --- Assertions ---
    c.load_quantized_model.assert_called_once_with("quant.pt")
    c._read_from_json_safely.assert_called_once_with(str(input_path))
    c.scale_inputs_only.assert_called_once_with({"mock": "inputs"})
    c.reshape_inputs_for_inference.assert_called_once_with({"scaled": "inputs"})
    c.reshape_inputs_for_circuit.assert_called_once_with({"scaled": "inputs"})

    # Verify safe JSON writes
    new_input_file = str(input_path.with_name("in_adjusted.json"))
    c._to_json_safely.assert_any_call({"input": [1, 2, 3]}, new_input_file, "input")
    c._to_json_safely.assert_any_call(
        {"formatted_output": 999},
        str(output_path),
        "output",
    )

    # Verify output generation
    c.get_outputs.assert_called_once_with({"inference": "inputs"})
    c.format_outputs.assert_called_once_with({"raw_output": 123})

    # Function should return the adjusted input file path
    assert result == new_input_file


# ---------- _compile_preprocessing ----------
@pytest.mark.unit
@patch("python.core.circuits.base.to_json")
def test_compile_preprocessing_saves_all_files(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_metadata = MagicMock(return_value={"version": "1.0"})
    c.get_architecture = MagicMock(return_value={"layers": ["conv", "relu"]})
    c.get_w_and_b = MagicMock(return_value={"weights": [1, 2, 3]})
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("metadata.json", "architecture.json", "w_and_b.json", None)

    c.get_model_and_quantize.assert_called_once()
    c.get_metadata.assert_called_once()
    c.get_architecture.assert_called_once()
    c.get_w_and_b.assert_called_once()
    c.save_quantized_model.assert_called_once_with("model.pth")
    mock_to_json.assert_any_call({"version": "1.0"}, "metadata.json")
    mock_to_json.assert_any_call({"layers": ["conv", "relu"]}, "architecture.json")
    mock_to_json.assert_any_call({"weights": [1, 2, 3]}, "w_and_b.json")


@pytest.mark.unit
@patch("python.core.circuits.base.to_json")
def test_compile_preprocessing_saves_all_files(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_metadata = MagicMock(return_value={"version": "1.0"})
    c.get_architecture = MagicMock(return_value={"layers": ["conv", "relu"]})
    c.get_w_and_b = MagicMock(return_value={"weights": [1, 2, 3]})
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("metadata.json", "architecture.json", "w_and_b.json", None)

    c.get_model_and_quantize.assert_called_once()
    c.get_metadata.assert_called_once()
    c.get_architecture.assert_called_once()
    c.get_w_and_b.assert_called_once()
    c.save_quantized_model.assert_called_once_with("model.pth")
    mock_to_json.assert_any_call({"version": "1.0"}, "metadata.json")
    mock_to_json.assert_any_call({"layers": ["conv", "relu"]}, "architecture.json")
    mock_to_json.assert_any_call({"weights": [1, 2, 3]}, "w_and_b.json")


@pytest.mark.unit
@patch("python.core.circuits.base.to_json")
def test_compile_preprocessing_weights_dict(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_metadata = MagicMock(return_value={"TEST": "2"})
    c.get_architecture = MagicMock(return_value={"TEST": "1"})
    c.get_w_and_b = MagicMock(return_value={"a": 1})
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("metadata.json", "architecture.json", "w_and_b.json", None)

    c.get_model_and_quantize.assert_called_once()
    c.get_w_and_b.assert_called_once()
    c.save_quantized_model.assert_called_once_with("model.pth")
    mock_to_json.assert_any_call({"TEST": "2"}, "metadata.json")
    mock_to_json.assert_any_call({"TEST": "1"}, "architecture.json")
    mock_to_json.assert_any_call({"a": 1}, "w_and_b.json")


@pytest.mark.unit
@patch("python.core.circuits.base.to_json")
def test_compile_preprocessing_weights_list(
    mock_to_json: MagicMock,
) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_metadata = MagicMock(return_value={"TEST": "1"})
    c.get_architecture = MagicMock(return_value={"TEST": "2"})
    c.get_w_and_b = MagicMock(return_value=[{"w1": 1}, {"w2": 2}, {"w3": 3}])
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("metadata.json", "architecture.json", "w_and_b.json", None)

    call_count = 5  # 2 for metadata/architecture + 3 for weights

    assert mock_to_json.call_count == call_count
    mock_to_json.assert_any_call({"TEST": "1"}, "metadata.json")
    mock_to_json.assert_any_call({"TEST": "2"}, "architecture.json")
    mock_to_json.assert_any_call({"w1": 1}, Path("w_and_b.json"))
    mock_to_json.assert_any_call({"w2": 2}, Path("w_and_b2.json"))
    mock_to_json.assert_any_call({"w3": 3}, Path("w_and_b3.json"))


@pytest.mark.unit
@patch("python.core.circuits.base.to_json")
def test_compile_preprocessing_weights_list_single_call(
    mock_to_json: MagicMock,
) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_metadata = MagicMock(return_value={})
    c.get_architecture = MagicMock(return_value={})
    c.get_weights = MagicMock(return_value=[{"w1": 1}, {"w2": 2}, {"w3": 3}])
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("metadata.json", "architecture.json", "w_and_b.json", None)

    call_count = 3

    assert mock_to_json.call_count == call_count  # +2 for metadata and architecture
    mock_to_json.assert_any_call({"w1": 1}, Path("w_and_b.json"))
    mock_to_json.assert_any_call({"w2": 2}, Path("w_and_b2.json"))
    mock_to_json.assert_any_call({"w3": 3}, Path("w_and_b3.json"))


@pytest.mark.unit
def test_compile_preprocessing_raises_on_bad_weights() -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_metadata = MagicMock(return_value={})
    c.get_architecture = MagicMock(return_value={})
    c.get_w_and_b = MagicMock(return_value="bad_type")
    c.save_quantized_model = MagicMock()

    with pytest.raises(CircuitConfigurationError, match="Unsupported w_and_b type"):
        c._compile_preprocessing(
            "metadata.json",
            "architecture.json",
            "w_and_b.json",
            None,
        )


# ---------- Test check attributes --------------
@pytest.mark.unit
def test_check_attributes_true() -> None:
    c = Circuit()
    c.required_keys = ["input"]
    c.name = "test"
    c.scale_exponent = 2
    c.scale_base = 2
    c.check_attributes()


@pytest.mark.unit
def test_check_attributes_no_scaling() -> None:
    c = Circuit()
    c.required_keys = ["input"]
    c.name = "test"
    c.scale_base = 2
    with pytest.raises(CircuitConfigurationError) as exc_info:
        c.check_attributes()

    msg = str(exc_info.value)
    assert "Circuit class (python) is misconfigured" in msg
    assert "scale_exponent" in msg


@pytest.mark.unit
def test_check_attributes_no_scalebase() -> None:
    c = Circuit()
    c.required_keys = ["input"]
    c.name = "test"
    c.scale_exponent = 2

    with pytest.raises(CircuitConfigurationError) as exc_info:
        c.check_attributes()

    msg = str(exc_info.value)
    assert "Circuit class (python) is misconfigured" in msg
    assert "scale_base" in msg


@pytest.mark.unit
def test_check_attributes_no_name() -> None:
    c = Circuit()
    c.required_keys = ["input"]
    c.scale_base = 2
    c.scale_exponent = 2

    with pytest.raises(CircuitConfigurationError) as exc_info:
        c.check_attributes()

    msg = str(exc_info.value)
    assert "Circuit class (python) is misconfigured" in msg
    assert "name" in msg


# ---------- base_testing ------------
@pytest.mark.unit
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_calls_parse_proof_run_type_correctly(
    mock_parse: MagicMock,
) -> None:
    c = Circuit()
    c.name = "test"

    c._file_info = {}
    c._file_info["metadata_path"] = "metadata.json"
    c._file_info["architecture_path"] = "architecture.json"
    c._file_info["w_and_b_path"] = "w_and_b.json"
    c.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            witness_file="w.wtns",
            input_file="i.json",
            proof_file="p.json",
            public_path="pub.json",
            verification_key="vk.key",
            circuit_name="circuit_model",
            output_file="o.json",
            circuit_path="circuit_path.txt",
            quantized_path="quantized_path.pt",
            write_json=True,
            proof_system=ZKProofSystems.Expander,
        ),
    )

    mock_parse.assert_called_once()
    expected_config = CircuitExecutionConfig(
        witness_file="w.wtns",
        input_file="i.json",
        proof_file="p.json",
        public_path="pub.json",
        verification_key="vk.key",
        circuit_name="circuit_model",
        circuit_path="circuit_path.txt",
        proof_system=ZKProofSystems.Expander,
        output_file="o.json",
        metadata_path="metadata.json",
        architecture_path="architecture.json",
        w_and_b_path="w_and_b.json",
        quantized_path="quantized_path.pt",
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        ecc=True,
        write_json=True,
        bench=False,
    )
    mock_parse.assert_called_once_with(expected_config)


@pytest.mark.unit
def test_prepare_io_files_sets_new_file_paths() -> None:
    """Test that prepare_io_files decorator sets the new file paths correctly."""
    from python.core.utils.helper_functions import prepare_io_files  # noqa: PLC0415

    class TestCircuit(Circuit):
        def __init__(self: Circuit) -> None:
            super().__init__()
            self.name = "test_circuit"

        @prepare_io_files
        def test_method(self: Circuit, exec_config: str) -> str:
            _ = exec_config
            return self._file_info

    c = TestCircuit()

    with patch("python.core.utils.helper_functions.get_files") as mock_get_files:
        mock_get_files.return_value = {
            "witness_file": "witness.wtns",
            "input_file": "input.json",
            "proof_path": "proof.json",
            "public_path": "public.json",
            "circuit_name": "test_circuit",
            "metadata_path": "metadata.json",
            "architecture_path": "architecture.json",
            "w_and_b_path": "w_and_b.json",
            "output_file": "output.json",
        }

        config = CircuitExecutionConfig(run_type=RunType.COMPILE_CIRCUIT)
        file_info = c.test_method(config)

        assert file_info["metadata_path"] == "metadata.json"
        assert file_info["architecture_path"] == "architecture.json"
        assert file_info["w_and_b_path"] == "w_and_b.json"
        assert config.metadata_path == "metadata.json"
        assert config.architecture_path == "architecture.json"
        assert config.w_and_b_path == "w_and_b.json"


@pytest.mark.unit
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_uses_default_circuit_path(mock_parse: MagicMock) -> None:
    class MyCircuit(Circuit):
        def __init__(self: "MyCircuit") -> None:
            super().__init__()
            self._file_info = {
                "metadata_path": "metadata.json",
                "architecture_path": "architecture.json",
                "w_and_b_path": "w_and_b.json",
            }

    c = MyCircuit()
    c.base_testing(CircuitExecutionConfig(circuit_name="test_model"))

    mock_parse.assert_called_once()
    config = mock_parse.call_args[0][0]

    assert config.circuit_name == "test_model"
    assert config.circuit_path == "test_model.txt"
    assert config.metadata_path == "metadata.json"
    assert config.architecture_path == "architecture.json"
    assert config.w_and_b_path == "w_and_b.json"


@pytest.mark.unit
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_returns_none(mock_parse: MagicMock) -> None:
    class MyCircuit(Circuit):
        def __init__(self: "MyCircuit") -> None:
            super().__init__()
            self._file_info = {
                "metadata_path": "metadata.json",
                "architecture_path": "architecture.json",
                "w_and_b_path": "w_and_b.json",
            }

    c = MyCircuit()
    result = c.base_testing(CircuitExecutionConfig(circuit_name="abc"))
    assert result is None
    mock_parse.assert_called_once()


@pytest.mark.unit
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_weights_exists(mock_parse: MagicMock) -> None:
    _ = mock_parse

    class MyCircuit(Circuit):
        def __init__(self: "MyCircuit") -> None:
            super().__init__()

    c = MyCircuit()
    with pytest.raises(CircuitConfigurationError, match="Circuit file information"):
        c.base_testing(CircuitExecutionConfig(circuit_name="abc"))


@pytest.mark.unit
def test_parse_proof_run_type_invalid_run_type(
    caplog: Generator[pytest.LogCaptureFixture, None, None],
) -> None:
    c = Circuit()
    config_invalid = CircuitExecutionConfig(
        witness_file="w.wtns",
        input_file="i.json",
        proof_file="p.json",
        public_path="pub.json",
        verification_key="vk.key",
        circuit_name="model",
        circuit_path="path.txt",
        proof_system=None,
        output_file="out.json",
        metadata_path="metadata.json",
        architecture_path="architecture.json",
        w_and_b_path="w_and_b.json",
        quantized_path="quantized_model.pt",
        run_type="NOT_A_REAL_RUN_TYPE",  # Invalid run type
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )

    with pytest.raises(CircuitRunError, match="Unsupported run type"):
        c.parse_proof_run_type(config_invalid)

    # Check that the error messages are logged
    assert "Unknown run type: NOT_A_REAL_RUN_TYPE" in caplog.text
    assert "Operation NOT_A_REAL_RUN_TYPE failed" in caplog.text


@pytest.mark.unit
@patch(
    "python.core.circuits.base.compile_circuit",
    side_effect=Exception("Boom goes the dynamite!"),
)
@patch.object(Circuit, "_compile_preprocessing")
def test_parse_proof_run_type_catches_internal_exception(
    mock_compile_preprocessing: MagicMock,
    mock_compile: MagicMock,
    caplog: Generator[pytest.LogCaptureFixture, None, None],
) -> None:
    c = Circuit()

    config_exception = CircuitExecutionConfig(
        witness_file="w.wtns",
        input_file="i.json",
        proof_file="p.json",
        public_path="pub.json",
        verification_key="vk.key",
        circuit_name="model",
        circuit_path="path.txt",
        proof_system=None,
        output_file="out.json",
        metadata_path="metadata.json",
        architecture_path="architecture.json",
        w_and_b_path="w_and_b.json",
        quantized_path="quantized_path.pt",
        run_type=RunType.COMPILE_CIRCUIT,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )

    # This will raise inside `compile_circuit`, which is patched to raise
    with pytest.raises(CircuitRunError, match="Circuit operation 'Compile' failed"):

        c.parse_proof_run_type(config_exception)

    # Check that the error message is logged
    assert "Operation RunType.COMPILE_CIRCUIT failed" in caplog.text
    assert mock_compile.called
    assert mock_compile_preprocessing.called


@pytest.mark.unit
def test_save_and_load_model_not_implemented() -> None:
    c = Circuit()
    assert hasattr(c, "save_model")
    assert hasattr(c, "load_model")
    assert hasattr(c, "save_quantized_model")
    assert hasattr(c, "load_quantized_model")


@pytest.mark.unit
def test_adjust_inputs_processing_error() -> None:
    c = Circuit()
    c.input_variables = ["input"]
    c.input_shape = [2, 2]
    c.scale_base = 2
    c.scale_exponent = 1

    with patch(
        "python.core.circuits.base.read_from_json",
        return_value={"input": [1, 2, 3, 4]},
    ):
        _ = c
        with patch("torch.tensor") as mock_tensor:
            mock_tensor.side_effect = RuntimeError("Invalid tensor shape")

            with pytest.raises(
                CircuitProcessingError,
                match="Failed to reshape input data",
            ):
                c.adjust_inputs({"input": [1, 2, 3, 4]}, "dummy.json")


@pytest.mark.unit
def test_get_inputs_from_file_file_error() -> None:
    c = Circuit()
    with patch.object(
        c,
        "_read_from_json_safely",
        side_effect=CircuitFileError("Failed to read input file: protected.json"),
    ):
        _ = c
        with pytest.raises(CircuitFileError, match="Failed to read input file"):
            c.get_inputs_from_file("protected.json")


@pytest.mark.unit
def test_get_inputs_from_file_processing_error() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 1

    with patch.object(
        c,
        "_read_from_json_safely",
        return_value={"input": "invalid_data"},
    ):
        _ = c
        with pytest.raises(
            CircuitProcessingError,
            match="Failed to scale input data",
        ):
            c.get_inputs_from_file("dummy.json", is_scaled=False)


# ---------- Test _raise_unknown_run_type ----------
@pytest.mark.unit
def test_raise_unknown_run_type() -> None:
    c = Circuit()

    with pytest.raises(CircuitRunError, match="Unsupported run type: INVALID_TYPE"):
        c._raise_unknown_run_type("INVALID_TYPE")


# ---------- Test contains_float ----------
@pytest.mark.unit
def test_contains_float_with_float() -> None:
    c = Circuit()
    assert c.contains_float(3.14) is True
    assert c.contains_float(2.0) is True
    assert c.contains_float(1.5) is True


@pytest.mark.unit
def test_contains_float_with_int() -> None:
    c = Circuit()
    assert c.contains_float(1) is False
    assert c.contains_float(0) is False
    assert c.contains_float(-5) is False


@pytest.mark.unit
def test_contains_float_with_list() -> None:
    c = Circuit()
    assert c.contains_float([1, 2, 3]) is False
    assert c.contains_float([1.0, 2, 3]) is True
    assert c.contains_float([1, 2.5, 3]) is True
    assert c.contains_float([]) is False


@pytest.mark.unit
def test_contains_float_with_dict() -> None:
    c = Circuit()
    assert c.contains_float({"a": 1, "b": 2}) is False
    assert c.contains_float({"a": 1.0, "b": 2}) is True
    assert c.contains_float({"a": 1, "b": 2.5}) is True
    assert c.contains_float({}) is False


@pytest.mark.unit
def test_contains_float_nested_structures() -> None:
    c = Circuit()
    nested_with_float = {"a": [1, 2.0, 3], "b": {"c": 4.5}}
    nested_without_float = {"a": [1, 2, 3], "b": {"c": 4}}

    assert c.contains_float(nested_with_float) is True
    assert c.contains_float(nested_without_float) is False


# ---------- Test adjust_shape ----------
@pytest.mark.unit
def test_adjust_shape_list() -> None:
    c = Circuit()
    assert c.adjust_shape([1, 2, 3]) == [1, 2, 3]
    assert c.adjust_shape([0, -1, 5]) == [1, 1, 5]
    assert c.adjust_shape([-5, 0, 3]) == [1, 1, 3]


@pytest.mark.unit
def test_adjust_shape_dict_single_value() -> None:
    c = Circuit()
    result = c.adjust_shape({"key": [2, 3, 4]})
    assert result == [2, 3, 4]
    assert result == [2, 3, 4]


@pytest.mark.unit
def test_adjust_shape_dict_multiple_values() -> None:
    c = Circuit()
    input_dict = {"input": [2, 3, 4], "weight": [1, -1, 5], "bias": [0, 0, 3]}
    expected = {"input": [2, 3, 4], "weight": [1, 1, 5], "bias": [1, 1, 3]}
    assert c.adjust_shape(input_dict) == expected


@pytest.mark.unit
def test_adjust_shape_invalid_type() -> None:
    c = Circuit()
    with pytest.raises(CircuitInputError, match="Expected list or dict for 'shape'"):
        c.adjust_shape("invalid")


@pytest.mark.unit
def test_adjust_shape_dict_invalid_value() -> None:
    c = Circuit()
    with pytest.raises(
        CircuitInputError,
        match="Expected shape list for input, got str",
    ):
        c.adjust_shape({"bad": "not_a_list"})


# ---------- Test scale_and_round ----------
@pytest.mark.unit
def test_scale_and_round_with_floats() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 2

    with patch(
        "python.core.model_processing.onnx_quantizer.layers.base.BaseOpQuantizer.get_scaling",
        return_value=4.0,
    ):
        result = c.scale_and_round([1.5, 2.5], 2, 2)
        assert result == [6, 10]  # rounded(1.5 * 4) = 6, rounded(2.5 * 4) = 10


@pytest.mark.unit
def test_scale_and_round_with_ints() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 2

    with patch(
        "python.core.model_processing.onnx_quantizer.layers.base.BaseOpQuantizer.get_scaling",
        return_value=4.0,
    ):
        result = c.scale_and_round([1, 2, 3], 2, 2)
        assert result == [1, 2, 3]  # No change for integers


@pytest.mark.unit
def test_scale_and_round_with_tensors() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 2

    with patch(
        "python.core.model_processing.onnx_quantizer.layers.base.BaseOpQuantizer.get_scaling",
        return_value=4.0,
    ):

        tensor_input = [1.5, 2.5]
        result = c.scale_and_round(tensor_input, 2, 2)
        assert result == [6, 10]


# ---------- Test _to_json_safely and _read_from_json_safely ----------
@pytest.mark.unit
@patch("python.core.circuits.base.to_json")
def test_to_json_safely_success(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._to_json_safely({"key": "value"}, "file.json", "test var")
    mock_to_json.assert_called_once_with({"key": "value"}, "file.json")


@pytest.mark.unit
@patch("python.core.circuits.base.to_json", side_effect=Exception("Write failed"))
def test_to_json_safely_failure(mock_to_json: MagicMock) -> None:
    c = Circuit()
    with pytest.raises(
        CircuitFileError,
        match=re.escape("Failed to write test var file: file.json"),
    ):
        c._to_json_safely({"key": "value"}, "file.json", "test var")


@pytest.mark.unit
@patch("python.core.circuits.base.read_from_json", return_value={"key": "value"})
def test_read_from_json_safely_success(mock_read: MagicMock) -> None:
    c = Circuit()
    result = c._read_from_json_safely("file.json")
    mock_read.assert_called_once_with("file.json")
    assert result == {"key": "value"}


@pytest.mark.unit
@patch("python.core.circuits.base.read_from_json", side_effect=Exception("Read failed"))
def test_read_from_json_safely_failure(mock_read: MagicMock) -> None:
    c = Circuit()
    with pytest.raises(
        CircuitFileError,
        match=re.escape("Failed to read input file: file.json"),
    ):
        c._read_from_json_safely("file.json")


# ---------- Test _adjust_single_input ----------
@pytest.mark.unit
def test_adjust_single_input_success() -> None:
    c = Circuit()
    c.input_shape = [2, 2]
    c.scale_base = 2
    c.scale_exponent = 1
    five = 5

    inputs = {"input": [1, 2, 3, 4], "extra": 5}
    result = c._adjust_single_input(inputs)

    assert "input" in result
    assert "extra" in result
    assert result["extra"] == five


# ---------- Test _adjust_multiple_inputs ----------
@pytest.mark.unit
def test_adjust_multiple_inputs_success() -> None:
    c = Circuit()
    c.x_shape = [2]
    c.y_shape = [2]
    c.scale_base = 2
    c.scale_exponent = 1
    five = 5

    inputs = {"x": [1, 2], "y": [3, 4], "z": 5}
    input_variables = ["x", "y"]
    result = c._adjust_multiple_inputs(inputs, input_variables)

    assert "x" in result
    assert "y" in result
    assert "z" in result
    assert result["z"] == five


# ---------- Test _reshape_input_value ----------
@pytest.mark.unit
def test_reshape_input_value_success() -> None:
    c = Circuit()
    c.input_shape = [2, 2]

    result = c._reshape_input_value([1, 2, 3, 4], "input_shape", "input")
    assert result == [[1, 2], [3, 4]]


@pytest.mark.unit
def test_reshape_input_value_missing_shape_attr() -> None:
    c = Circuit()

    with pytest.raises(
        CircuitConfigurationError,
        match="Required shape attribute 'missing_shape'",
    ):
        c._reshape_input_value([1, 2, 3, 4], "missing_shape", "input")


@pytest.mark.unit
def test_reshape_input_value_invalid_shape() -> None:
    c = Circuit()
    c.input_shape = [2, 3]  # 6 elements needed

    with pytest.raises(CircuitProcessingError, match="Failed to reshape input data"):
        c._reshape_input_value([1, 2, 3, 4], "input_shape", "input")


# ---------- Test scale_inputs_only ----------
@pytest.mark.unit
def test_scale_inputs_only_success() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 1

    inputs = {"x": [1, 2], "y": [3, 4]}
    with patch.object(
        c,
        "scale_and_round",
        side_effect=lambda v, _sb, _se: [v[0] * 2, v[1] * 2],
    ):
        result = c.scale_inputs_only(inputs)
        assert result == {"x": [2, 4], "y": [6, 8]}


# ---------- Test rename_inputs ----------
@pytest.mark.unit
def test_rename_inputs_single_input() -> None:
    c = Circuit()
    c.input_variables = ["input"]

    inputs = {"input_data": [1, 2, 3], "extra": 4}
    result = c.rename_inputs(inputs)

    assert result == {"input": [1, 2, 3], "extra": 4}


@pytest.mark.unit
def test_rename_inputs_multiple_inputs() -> None:
    c = Circuit()
    c.input_variables = ["x", "y"]

    inputs = {"x": [1, 2], "y": [3, 4], "z": 5}
    result = c.rename_inputs(inputs)

    assert result == inputs  # Should remain unchanged


# ---------- Test _rename_single_input ----------
@pytest.mark.unit
def test_rename_single_input_success() -> None:
    c = Circuit()
    inputs = {"input_vec": [1, 2, 3], "extra": 4}
    result = c._rename_single_input(inputs)

    assert result == {"input": [1, 2, 3], "extra": 4}


@pytest.mark.unit
def test_rename_single_input_multiple_keys_error() -> None:
    c = Circuit()
    inputs = {"input1": [1, 2], "input2": [3, 4]}

    with pytest.raises(
        CircuitInputError,
        match="Multiple inputs found containing 'input'",
    ):
        c._rename_single_input(inputs)


# ---------- Test reshape_inputs_for_inference ----------
@pytest.mark.unit
def test_reshape_inputs_for_inference_single_input() -> None:
    c = Circuit()
    c.input_shape = [2, 2]

    inputs = {"data": [1, 2, 3, 4]}
    result = c.reshape_inputs_for_inference(inputs)

    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_reshape_inputs_for_inference_dict_shapes() -> None:

    c = Circuit()
    c.input_shape = {"x": [2], "y": [2]}

    inputs = {"x": [1, 2], "y": [3, 4]}
    result = c.reshape_inputs_for_inference(inputs)

    _ = {"x": np.array([1, 2]), "y": np.array([3, 4])}
    assert list(result.keys()) == ["x", "y"]


@pytest.mark.unit
def test_reshape_inputs_for_inference_missing_shape() -> None:
    c = Circuit()
    inputs = {"data": [1, 2, 3, 4]}

    with pytest.raises(CircuitConfigurationError, match="input_shape"):
        c.reshape_inputs_for_inference(inputs)


@pytest.mark.unit
def test_reshape_inputs_for_inference_shape_mismatch() -> None:
    c = Circuit()
    c.input_shape = [2, 3]  # Needs 6 elements

    inputs = {"data": [1, 2, 3, 4]}  # Only 4 elements

    with pytest.raises(ShapeMismatchError):
        c.reshape_inputs_for_inference(inputs)


# ---------- Test _reshape_dict_inputs ----------
@pytest.mark.unit
def test_reshape_dict_inputs_success() -> None:
    c = Circuit()
    shape_dict = {"x": [2], "y": [2, 1]}

    inputs = {"x": [1, 2], "y": [3, 4]}
    result = c._reshape_dict_inputs(inputs, shape_dict)

    np.testing.assert_array_equal(result["x"], np.array([1, 2]))
    np.testing.assert_array_equal(result["y"], np.array([[3], [4]]))


@pytest.mark.unit
def test_reshape_dict_inputs_non_dict_shape() -> None:
    c = Circuit()
    shape_list = [2, 2]

    with pytest.raises(
        CircuitInputError,
        match="_reshape_dict_inputs requires dict shape",
    ):
        c._reshape_dict_inputs({"x": [1, 2]}, shape_list)


@pytest.mark.unit
def test_reshape_dict_inputs_shape_mismatch() -> None:
    c = Circuit()
    shape_dict = {"x": [2, 2]}  # Needs 4 elements

    inputs = {"x": [1, 2]}  # Only 2 elements

    with pytest.raises(ShapeMismatchError):
        c._reshape_dict_inputs(inputs, shape_dict)


# ---------- Test reshape_inputs_for_circuit ----------
@pytest.mark.unit
def test_reshape_inputs_for_circuit_success() -> None:
    c = Circuit()
    inputs = {"x": [1, 2], "y": [3, 4]}

    result = c.reshape_inputs_for_circuit(inputs)

    assert result == {"input": [1, 2, 3, 4]}


@pytest.mark.unit
def test_reshape_inputs_for_circuit_with_input_shapes() -> None:
    c = Circuit()
    c.input_shapes = {"y": [2], "x": [2]}  # Ordered differently

    inputs = {"x": [1, 2], "y": [3, 4]}

    result = c.reshape_inputs_for_circuit(inputs)

    assert result == {"input": [3, 4, 1, 2]}  # Respects order from input_shapes


@pytest.mark.unit
def test_reshape_inputs_for_circuit_invalid_type() -> None:
    c = Circuit()

    with pytest.raises(CircuitConfigurationError, match="Expected a dict, got list"):
        c.reshape_inputs_for_circuit([1, 2, 3, 4])


@pytest.mark.unit
def test_reshape_inputs_for_circuit_missing_key() -> None:
    c = Circuit()
    c.input_shapes = {"x": [2], "y": [2]}

    inputs = {"x": [1, 2]}  # Missing "y"

    with pytest.raises(CircuitProcessingError, match="Missing expected input key 'y'"):
        c.reshape_inputs_for_circuit(inputs)


@pytest.mark.unit
def test_reshape_inputs_for_circuit_unsupported_type() -> None:
    c = Circuit()

    inputs = {"x": "invalid_type"}

    with pytest.raises(
        CircuitProcessingError,
        match="Unsupported input type for key 'x'",
    ):
        c.reshape_inputs_for_circuit(inputs)


# ---------- Test load_and_compare_witness_to_io ----------
@pytest.mark.unit
@patch("python.core.circuits.base.load_witness")
@patch("python.core.circuits.base.compare_witness_to_io")
def test_load_and_compare_witness_to_io_success(
    mock_compare: MagicMock,
    mock_load: MagicMock,
) -> None:
    c = Circuit()
    c._read_from_json_safely = MagicMock
    mock_load.return_value = {"modulus": 10, "public_inputs": [1, 2, 3]}
    mock_compare.return_value = True

    _ = c.load_and_compare_witness_to_io(
        "witness.bin",
        "inputs.json",
        "outputs.json",
        ZKProofSystems.Expander,
    )

    mock_load.assert_called_once_with("witness.bin", ZKProofSystems.Expander)
    mock_compare.assert_called_once()


@pytest.mark.unit
@patch("python.core.circuits.base.load_witness")
def test_load_and_compare_witness_to_io_missing_modulus(mock_load: MagicMock) -> None:
    c = Circuit()
    c._read_from_json_safely = MagicMock
    mock_load.return_value = {"public_inputs": [1, 2, 3]}  # No modulus

    with pytest.raises(
        WitnessMatchError,
        match=r"Witness not correctly formed\. Missing modulus\.",
    ):
        c.load_and_compare_witness_to_io(
            "witness.bin",
            "inputs.json",
            "outputs.json",
            ZKProofSystems.Expander,
        )


# ---------- Test prepare_inputs_for_verification ----------
@pytest.mark.unit
def test_prepare_inputs_for_verification_success(tmp_path: Path) -> None:
    c = Circuit()
    c._read_from_json_safely = MagicMock(return_value={"input": [1, 2, 3, 4]})
    c.reshape_inputs_for_circuit = MagicMock(return_value={"input": [1, 2, 3, 4]})
    c._to_json_safely = MagicMock()

    input_file = tmp_path / "input.json"
    exec_config = MagicMock()
    exec_config.input_file = str(input_file)

    result = c.prepare_inputs_for_verification(exec_config)

    expected_file = str(tmp_path / "input_veri.json")
    assert result == expected_file

    c._read_from_json_safely.assert_called_once_with(str(input_file))
    c.reshape_inputs_for_circuit.assert_called_once_with({"input": [1, 2, 3, 4]})
    c._to_json_safely.assert_called_once_with(
        {"input": [1, 2, 3, 4]},
        expected_file,
        "renamed input",
    )

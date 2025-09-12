import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

sys.modules.pop("python.core.circuits.base", None)


with patch(
    "python.core.utils.helper_functions.compute_and_store_output",
    lambda x: x,
), patch(
    "python.core.utils.helper_functions.prepare_io_files",
    lambda f: f,
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
    )


# ---------- Test __init__ ----------
@pytest.mark.unit()
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


# ---------- Test parse_inputs ----------
@pytest.mark.unit()
def test_parse_inputs_missing_required_keys() -> None:
    c = Circuit()
    c.required_keys = ["x", "y"]
    with pytest.raises(CircuitInputError, match="Missing required parameter: 'x'"):
        c.parse_inputs(y=5)


@pytest.mark.unit()
def test_parse_inputs_type_check() -> None:
    c = Circuit()
    c.required_keys = ["x"]
    with pytest.raises(
        CircuitInputError,
        match="Parameter 'x' must be an int or list of ints",
    ):
        c.parse_inputs(x="not-an-int")


@pytest.mark.unit()
def test_parse_inputs_success_int() -> None:
    c = Circuit()
    c.required_keys = ["x", "y"]
    x = 10
    y = 20

    c.parse_inputs(x=x, y=y)

    assert c.x == x
    assert c.y == y


@pytest.mark.unit()
def test_parse_inputs_success_list() -> None:
    c = Circuit()
    c.required_keys = ["arr"]
    c.parse_inputs(arr=[1, 2, 3])
    assert c.arr == [1, 2, 3]


@pytest.mark.unit()
def test_parse_inputs_required_keys_none() -> None:
    c = Circuit()
    with pytest.raises(CircuitConfigurationError):
        c.parse_inputs()


# ---------- Test Not Implemented --------------
@pytest.mark.unit()
def test_get_inputs_not_implemented() -> None:
    c = Circuit()
    with pytest.raises(NotImplementedError, match="get_inputs must be implemented"):
        c.get_inputs()


@pytest.mark.unit()
def test_get_outputs_not_implemented() -> None:
    c = Circuit()
    with pytest.raises(NotImplementedError, match="get_outputs must be implemented"):
        c.get_outputs()


# ---------- Test parse_proof_run_type ----------


@pytest.mark.unit()
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
        weights_path="weights",
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
        weights_path="weights",
        quantized_path="q",
    )
    args, kwargs = mock_compile.call_args
    assert kwargs == {
        "circuit_name": "circuit",
        "circuit_path": "path",
        "proof_system": ZKProofSystems.Expander,
        "dev_mode": False,
        "bench": False,
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
        weights_path="weights",
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
    args, kwargs = mock_witness.call_args
    assert kwargs == {
        "circuit_name": "circuit",
        "circuit_path": "path",
        "witness_file": "w",
        "input_file": "i",
        "output_file": "out",
        "proof_system": ZKProofSystems.Expander,
        "dev_mode": False,
        "bench": False,
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
        weights_path="weights",
        quantized_path="q",
        run_type=RunType.PROVE_WITNESS,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )
    c.parse_proof_run_type(config_prove)
    mock_proof.assert_called_once()
    args, kwargs = mock_proof.call_args

    assert kwargs == {
        "circuit_name": "circuit",
        "circuit_path": "path",
        "witness_file": "w",
        "proof_file": "p",
        "proof_system": ZKProofSystems.Expander,
        "dev_mode": False,
        "ecc": True,
        "bench": False,
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
        weights_path="weights",
        quantized_path="q",
        run_type=RunType.GEN_VERIFY,
        dev_mode=False,
        ecc=True,
        write_json=False,
        bench=False,
    )
    c.parse_proof_run_type(config_verify)
    mock_verify.assert_called_once()
    args, kwargs = mock_verify.call_args
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
        weights_path="weights",
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


# ---------- Optional: test get_weights ----------
@pytest.mark.unit()
def test_get_weights_default() -> None:
    c = Circuit()
    assert c.get_weights() == {}


@pytest.mark.unit()
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


@pytest.mark.unit()
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


@pytest.mark.unit()
def test_get_inputs_from_file_dne() -> None:
    c = Circuit()
    c.scale_base = 2
    c.scale_exponent = 2
    with pytest.raises(CircuitFileError, match="Failed to read input file"):
        c.get_inputs_from_file("this_file_should_not_exist_12345.json", is_scaled=True)


@pytest.mark.unit()
def test_format_outputs() -> None:
    c = Circuit()
    out = c.format_outputs([10, 15, 20])
    assert out == {"output": [10, 15, 20]}


# ---------- _gen_witness_preprocessing ----------
@pytest.mark.unit()
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


@pytest.mark.unit()
@patch("python.core.circuits.base.to_json")
def test_gen_witness_preprocessing_write_json_false(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "quant.pt"}
    c.load_quantized_model = MagicMock()
    c.get_inputs_from_file = MagicMock(return_value="mock_inputs")
    c.reshape_inputs = MagicMock(return_value="in.json")
    c.rescale_inputs = MagicMock(return_value="in.json")
    c.rename_inputs = MagicMock(return_value="in.json")
    c.rescale_and_reshape_inputs = MagicMock(return_value="in.json")
    c.adjust_inputs = MagicMock(return_value="in.json")

    c.get_outputs = MagicMock(return_value="mock_outputs")
    c.format_outputs = MagicMock(return_value={"output": 99})

    c._gen_witness_preprocessing(
        "in.json",
        "out.json",
        None,
        write_json=False,
        is_scaled=False,
    )

    c.load_quantized_model.assert_called_once_with("quant.pt")
    c.get_inputs_from_file.assert_called_once_with("in.json", is_scaled=False)
    c.get_outputs.assert_called_once_with("mock_inputs")
    c.format_outputs.assert_called_once_with("mock_outputs")
    mock_to_json.assert_called_once_with({"output": 99}, "out.json")


# ---------- _compile_preprocessing ----------
@pytest.mark.unit()
@patch("python.core.circuits.base.to_json")
def test_compile_preprocessing_weights_dict(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_weights = MagicMock(return_value={"a": 1})
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("weights.json", None)

    c.get_model_and_quantize.assert_called_once()
    c.get_weights.assert_called_once()
    c.save_quantized_model.assert_called_once_with("model.pth")
    mock_to_json.assert_called_once_with({"a": 1}, "weights.json")


@pytest.mark.unit()
@patch("python.core.circuits.base.to_json")
def test_compile_preprocessing_weights_list(mock_to_json: MagicMock) -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_weights = MagicMock(return_value=[{"w1": 1}, {"w2": 2}, {"w3": 3}])
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("weights.json", None)

    call_count = 3

    assert mock_to_json.call_count == call_count
    mock_to_json.assert_any_call({"w1": 1}, "weights.json")
    mock_to_json.assert_any_call({"w2": 2}, "weights2.json")
    mock_to_json.assert_any_call({"w3": 3}, "weights3.json")


@pytest.mark.unit()
def test_compile_preprocessing_raises_on_bad_weights() -> None:
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_weights = MagicMock(return_value="bad_type")
    c.save_quantized_model = MagicMock()

    with pytest.raises(CircuitConfigurationError, match="Unsupported weights type"):
        c._compile_preprocessing("weights.json", None)


# ---------- Test check attributes --------------
@pytest.mark.unit()
def test_check_attributes_true() -> None:
    c = Circuit()
    c.required_keys = ["input"]
    c.name = "test"
    c.scale_exponent = 2
    c.scale_base = 2
    c.check_attributes()


@pytest.mark.unit()
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


@pytest.mark.unit()
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


@pytest.mark.unit()
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
@pytest.mark.unit()
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_calls_parse_proof_run_type_correctly(
    mock_parse: MagicMock,
) -> None:
    c = Circuit()
    c.name = "test"

    c._file_info = {}
    c._file_info["weights"] = "weights/model_weights.json"
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
        weights_path="weights/model_weights.json",
        quantized_path="quantized_path.pt",
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        ecc=True,
        write_json=True,
        bench=False,
    )
    mock_parse.assert_called_once_with(expected_config)


@pytest.mark.unit()
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_uses_default_circuit_path(mock_parse: MagicMock) -> None:
    class MyCircuit(Circuit):
        def __init__(self: "MyCircuit") -> None:
            super().__init__()
            self._file_info = {"weights": "weights.json"}

    c = MyCircuit()
    c.base_testing(CircuitExecutionConfig(circuit_name="test_model"))

    mock_parse.assert_called_once()
    config = mock_parse.call_args[0][0]

    assert config.circuit_name == "test_model"
    assert config.circuit_path == "test_model.txt"
    assert config.weights_path == "weights.json"


@pytest.mark.unit()
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_returns_none(mock_parse: MagicMock) -> None:
    class MyCircuit(Circuit):
        def __init__(self: "MyCircuit") -> None:
            super().__init__()
            self._file_info = {"weights": "some_weights.json"}

    c = MyCircuit()
    result = c.base_testing(CircuitExecutionConfig(circuit_name="abc"))
    assert result is None
    mock_parse.assert_called_once()


@pytest.mark.unit()
@patch.object(Circuit, "parse_proof_run_type")
def test_base_testing_weights_exists(mock_parse: MagicMock) -> None:
    _ = mock_parse

    class MyCircuit(Circuit):
        def __init__(self: "MyCircuit") -> None:
            super().__init__()

    c = MyCircuit()
    with pytest.raises(CircuitConfigurationError, match="Circuit file information"):
        c.base_testing(CircuitExecutionConfig(circuit_name="abc"))


@pytest.mark.unit()
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
        weights_path="weights.json",
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


@pytest.mark.unit()
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
        weights_path="weights.json",
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


@pytest.mark.unit()
def test_save_and_load_model_not_implemented() -> None:
    c = Circuit()
    assert hasattr(c, "save_model")
    assert hasattr(c, "load_model")
    assert hasattr(c, "save_quantized_model")
    assert hasattr(c, "load_quantized_model")


# ---------- New error handling tests ----------
@pytest.mark.unit()
def test_adjust_inputs_file_error() -> None:
    c = Circuit()
    c.input_variables = ["input"]
    c.input_shape = [2, 2]
    c.scale_base = 2
    c.scale_exponent = 1

    with patch(
        "python.core.circuits.base.read_from_json",
        side_effect=FileNotFoundError("File not found"),
    ):
        _ = c
        with pytest.raises(CircuitFileError, match="Failed to read input file"):
            c.adjust_inputs("nonexistent.json")


@pytest.mark.unit()
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
                c.adjust_inputs("dummy.json")


@pytest.mark.unit()
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


@pytest.mark.unit()
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

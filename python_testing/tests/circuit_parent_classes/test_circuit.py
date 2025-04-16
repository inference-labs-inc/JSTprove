import pytest
from unittest.mock import patch, MagicMock


with patch('python_testing.utils.helper_functions.compute_and_store_output', lambda x: x):  # MUST BE BEFORE THE UUT GETS IMPORTED ANYWHERE!
    from python_testing.circuit_components.circuit_helpers import ZKProofSystems, RunType, Circuit



# ---------- Test __init__ ----------

def test_circuit_init_defaults():
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

def test_parse_inputs_missing_required_keys():
    c = Circuit()
    c.required_keys = ["x", "y"]
    with pytest.raises(KeyError, match="Missing required parameter: x"):
        c.parse_inputs(y=5)

def test_parse_inputs_type_check():
    c = Circuit()
    c.required_keys = ["x"]
    with pytest.raises(ValueError, match="Expected an integer for x"):
        c.parse_inputs(x="not-an-int")

def test_parse_inputs_success_int():
    c = Circuit()
    c.required_keys = ["x", "y"]
    c.parse_inputs(x=10, y=20)
    assert c.x == 10
    assert c.y == 20

def test_parse_inputs_success_list():
    c = Circuit()
    c.required_keys = ["arr"]
    c.parse_inputs(arr=[1, 2, 3])
    assert c.arr == [1, 2, 3]

def test_parse_inputs_required_keys_none():
    c = Circuit()
    with pytest.raises(NotImplementedError):
        c.parse_inputs()

# ---------- Test Not Implemented --------------
def test_get_inputs_not_implemented():
    c = Circuit()
    with pytest.raises(NotImplementedError, match="get_inputs must be implemented"):
        c.get_inputs()


def test_get_outputs_not_implemented():
    c = Circuit()
    with pytest.raises(NotImplementedError, match="get_outputs must be implemented"):
        c.get_outputs()


# ---------- Test parse_proof_run_type ----------

# @patch()

@patch("python_testing.circuit_components.circuit_helpers.prove_and_verify")
@patch("python_testing.circuit_components.circuit_helpers.compile_circuit")
@patch("python_testing.circuit_components.circuit_helpers.generate_witness")
@patch("python_testing.circuit_components.circuit_helpers.generate_proof")
@patch("python_testing.circuit_components.circuit_helpers.generate_verification")
@patch("python_testing.circuit_components.circuit_helpers.run_end_to_end")
def test_parse_proof_dispatch_logic(
    mock_end_to_end,
    mock_verify,
    mock_proof,
    mock_witness,
    mock_compile,
    mock_prove_and_verify
):
    c = Circuit()

    # Mock internal preprocessing methods
    c._compile_preprocessing = MagicMock()
    c._gen_witness_preprocessing = MagicMock()

    # COMPILE_CIRCUIT
    c.parse_proof_run_type(
        "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
        "out", "weights", RunType.COMPILE_CIRCUIT
    )
    mock_compile.assert_called_once()
    c._compile_preprocessing.assert_called_once_with("weights")

    # GEN_WITNESS
    c.parse_proof_run_type(
        "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
        "out", "weights", RunType.GEN_WITNESS
    )
    mock_witness.assert_called_once()
    c._gen_witness_preprocessing.assert_called()

    # PROVE_WITNESS
    c.parse_proof_run_type(
        "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
        "out", "weights", RunType.PROVE_WITNESS
    )
    mock_proof.assert_called_once()

    # GEN_VERIFY
    c.parse_proof_run_type(
        "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
        "out", "weights", RunType.GEN_VERIFY
    )
    mock_verify.assert_called_once()

    # END_TO_END
    c.parse_proof_run_type(
        "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
        "out", "weights", RunType.END_TO_END
    )
    mock_end_to_end.assert_called_once()
    assert c._compile_preprocessing.call_count >= 2
    assert c._gen_witness_preprocessing.call_count >= 2

    # BASE_TESTING
    c.parse_proof_run_type(
        "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
        "out", "weights", RunType.BASE_TESTING
    )
    mock_prove_and_verify.assert_called_once()



# ---------- Optional: test get_weights ----------

def test_get_weights_default():
    c = Circuit()
    assert c.get_weights() == {}

def test_get_inputs_from_file():
    c = Circuit()
    c.scale_base = 2
    c.scaling = 2
    with patch('python_testing.circuit_components.circuit_helpers.read_from_json', return_value = {"input":[1,2,3,4]}):
        x = c.get_inputs_from_file("", is_scaled=True)
        assert x == {"input":[1,2,3,4]}

        y = c.get_inputs_from_file("", is_scaled=False)
        assert y == {"input":[4,8,12,16]}

def test_get_inputs_from_file_multiple_inputs():
    c = Circuit()
    c.scale_base = 2
    c.scaling = 2
    with patch('python_testing.circuit_components.circuit_helpers.read_from_json', return_value = {"input":[1,2,3,4], "nonce": 25}):
        x = c.get_inputs_from_file("", is_scaled=True)
        assert x == {"input":[1,2,3,4], "nonce": 25}

        y = c.get_inputs_from_file("", is_scaled=False)
        assert y == {"input":[4,8,12,16], "nonce": 100}

def test_get_inputs_from_file_dne():
    c = Circuit()
    c.scale_base = 2
    c.scaling = 2
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        c.get_inputs_from_file("", is_scaled=True)
    
def test_format_outputs():
    c = Circuit()
    out = c.format_outputs([10,15,20])
    assert out == {"output":[10,15,20]}


# ---------- _gen_witness_preprocessing ----------

@patch("python_testing.circuit_components.circuit_helpers.to_json")
def test_gen_witness_preprocessing_write_json_true(mock_to_json):
    c = Circuit()
    c._file_info = {"quantized_model_path": "quant.pt"}
    c.load_quantized_model = MagicMock()
    c.get_inputs = MagicMock(return_value="inputs")
    c.get_outputs = MagicMock(return_value="outputs")
    c.format_inputs = MagicMock(return_value={"input": 1})
    c.format_outputs = MagicMock(return_value={"output": 2})

    c._gen_witness_preprocessing("in.json", "out.json", write_json=True, is_scaled=True)

    c.load_quantized_model.assert_called_once_with("quant.pt")
    c.get_inputs.assert_called_once()
    c.get_outputs.assert_called_once_with("inputs")
    mock_to_json.assert_any_call({"input": 1}, "in.json")
    mock_to_json.assert_any_call({"output": 2}, "out.json")


@patch("python_testing.circuit_components.circuit_helpers.to_json")
def test_gen_witness_preprocessing_write_json_false(mock_to_json):
    c = Circuit()
    c._file_info = {"quantized_model_path": "quant.pt"}
    c.load_quantized_model = MagicMock()
    c.get_inputs_from_file = MagicMock(return_value="mock_inputs")
    c.get_outputs = MagicMock(return_value="mock_outputs")
    c.format_outputs = MagicMock(return_value={"output": 99})

    c._gen_witness_preprocessing("in.json", "out.json", write_json=False, is_scaled=False)

    c.load_quantized_model.assert_called_once_with("quant.pt")
    c.get_inputs_from_file.assert_called_once_with("in.json", is_scaled=False)
    c.get_outputs.assert_called_once_with("mock_inputs")
    c.format_outputs.assert_called_once_with("mock_outputs")
    mock_to_json.assert_called_once_with({"output": 99}, "out.json")


# ---------- _compile_preprocessing ----------

@patch("python_testing.circuit_components.circuit_helpers.to_json")
def test_compile_preprocessing_weights_dict(mock_to_json):
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_weights = MagicMock(return_value={"a": 1})
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("weights.json")

    c.get_model_and_quantize.assert_called_once()
    c.get_weights.assert_called_once()
    c.save_quantized_model.assert_called_once_with("model.pth")
    mock_to_json.assert_called_once_with({"a": 1}, "weights.json")


@patch("python_testing.circuit_components.circuit_helpers.to_json")
def test_compile_preprocessing_weights_list(mock_to_json):
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_weights = MagicMock(return_value=[{"w1": 1}, {"w2": 2}, {"w3": 3}])
    c.save_quantized_model = MagicMock()

    c._compile_preprocessing("weights.json")

    assert mock_to_json.call_count == 3
    mock_to_json.assert_any_call({"w1": 1}, "weights.json")
    mock_to_json.assert_any_call({"w2": 2}, "weights2.json")
    mock_to_json.assert_any_call({"w3": 3}, "weights3.json")


def test_compile_preprocessing_raises_on_bad_weights():
    c = Circuit()
    c._file_info = {"quantized_model_path": "model.pth"}
    c.get_model_and_quantize = MagicMock()
    c.get_weights = MagicMock(return_value="bad_type")
    c.save_quantized_model = MagicMock()

    with pytest.raises(NotImplementedError, match="Weights type is incorrect"):
        c._compile_preprocessing("weights.json")
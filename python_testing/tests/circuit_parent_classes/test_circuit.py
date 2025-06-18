# import pytest
# from unittest.mock import patch, MagicMock
# import sys
# sys.modules.pop("python_testing.circuit_components.circuit_helpers", None)



# with patch('python_testing.utils.helper_functions.compute_and_store_output', lambda x: x):  # MUST BE BEFORE THE UUT GETS IMPORTED ANYWHERE!
#     with patch('python_testing.utils.helper_functions.prepare_io_files', lambda f: f):  # MUST BE BEFORE THE UUT GETS IMPORTED ANYWHERE!
#         from python_testing.circuit_components.circuit_helpers import ZKProofSystems, RunType, Circuit



# # ---------- Test __init__ ----------

# def test_circuit_init_defaults():
#     c = Circuit()
#     assert c.input_folder == "inputs"
#     assert c.proof_folder == "analysis"
#     assert c.temp_folder == "temp"
#     assert c.circuit_folder == ""
#     assert c.weights_folder == "weights"
#     assert c.output_folder == "output"
#     assert c.proof_system == ZKProofSystems.Expander
#     assert c._file_info is None
#     assert c.required_keys is None


# # ---------- Test parse_inputs ----------

# def test_parse_inputs_missing_required_keys():
#     c = Circuit()
#     c.required_keys = ["x", "y"]
#     with pytest.raises(KeyError, match="Missing required parameter: x"):
#         c.parse_inputs(y=5)

# def test_parse_inputs_type_check():
#     c = Circuit()
#     c.required_keys = ["x"]
#     with pytest.raises(ValueError, match="Expected an integer for x"):
#         c.parse_inputs(x="not-an-int")

# def test_parse_inputs_success_int():
#     c = Circuit()
#     c.required_keys = ["x", "y"]
#     c.parse_inputs(x=10, y=20)
#     assert c.x == 10
#     assert c.y == 20

# def test_parse_inputs_success_list():
#     c = Circuit()
#     c.required_keys = ["arr"]
#     c.parse_inputs(arr=[1, 2, 3])
#     assert c.arr == [1, 2, 3]

# def test_parse_inputs_required_keys_none():
#     c = Circuit()
#     with pytest.raises(NotImplementedError):
#         c.parse_inputs()

# # ---------- Test Not Implemented --------------
# def test_get_inputs_not_implemented():
#     c = Circuit()
#     with pytest.raises(NotImplementedError, match="get_inputs must be implemented"):
#         c.get_inputs()


# def test_get_outputs_not_implemented():
#     c = Circuit()
#     # c.name = "test"
#     with pytest.raises(NotImplementedError, match="get_outputs must be implemented"):
#         c.get_outputs()


# # ---------- Test parse_proof_run_type ----------

# # @patch()

# @patch("python_testing.circuit_components.circuit_helpers.prove_and_verify")
# @patch("python_testing.circuit_components.circuit_helpers.compile_circuit")
# @patch("python_testing.circuit_components.circuit_helpers.generate_witness")
# @patch("python_testing.circuit_components.circuit_helpers.generate_proof")
# @patch("python_testing.circuit_components.circuit_helpers.generate_verification")
# @patch("python_testing.circuit_components.circuit_helpers.run_end_to_end")
# def test_parse_proof_dispatch_logic(
#     mock_end_to_end,
#     mock_verify,
#     mock_proof,
#     mock_witness,
#     mock_compile,
#     mock_prove_and_verify
# ):
#     c = Circuit()

#     # Mock internal preprocessing methods
#     c._compile_preprocessing = MagicMock()
#     c._gen_witness_preprocessing = MagicMock(return_value = "i")
#     # c.rescale_inputs = MagicMock(return_value = "i")
#     # c.reshape_inputs = MagicMock(return_value = "i")
#     c.adjust_inputs = MagicMock(return_value = "i")



#     # COMPILE_CIRCUIT
#     c.parse_proof_run_type( 
#         "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
#         "out", "weights", "q", RunType.COMPILE_CIRCUIT
#     )
#     mock_compile.assert_called_once()
#     c._compile_preprocessing.assert_called_once_with("weights", "q")
#     args = mock_compile.call_args[0]
#     assert args == ('circuit', "path", ZKProofSystems.Expander, False, False)

#     # GEN_WITNESS
#     c.parse_proof_run_type(
#         "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
#         "out", "weights", "q", RunType.GEN_WITNESS
#     )
#     mock_witness.assert_called_once()
#     c._gen_witness_preprocessing.assert_called()
#     args = mock_witness.call_args[0]
#     assert args == ('circuit', "path", "w","i", "out", ZKProofSystems.Expander, False, False)

#     # PROVE_WITNESS
#     c.parse_proof_run_type(
#         "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
#         "out", "weights", "q", RunType.PROVE_WITNESS
#     )
#     mock_proof.assert_called_once()
#     args = mock_proof.call_args[0]
#     kwargs = mock_proof.call_args[1]



#     assert args == ('circuit', "path", "w","p", ZKProofSystems.Expander, False)
#     assert kwargs == {'ecc': True, 'bench': False}


#     # GEN_VERIFY
#     c.parse_proof_run_type(
#         "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
#         "out", "weights", "q", RunType.GEN_VERIFY
#     )
#     mock_verify.assert_called_once()
#     args = mock_verify.call_args[0]
#     kwargs = mock_proof.call_args[1]
#     assert args == ('circuit', "path", "i","out", "w", "p", ZKProofSystems.Expander, False)
#     assert kwargs == {'ecc': True, 'bench': False}



#     # END_TO_END
#     c.parse_proof_run_type(
#         "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
#         "out", "weights", "q", RunType.END_TO_END
#     )
#     mock_end_to_end.assert_called_once()
#     assert c._compile_preprocessing.call_count >= 2
#     assert c._gen_witness_preprocessing.call_count >= 2

#     # BASE_TESTING
#     c.parse_proof_run_type(
#         "w", "i", "p", "pub", "vk", "circuit", "path", ZKProofSystems.Expander,
#         "out", "weights", "q", RunType.BASE_TESTING
#     )
#     mock_prove_and_verify.assert_called_once()


# # ---------- Optional: test get_weights ----------

# def test_get_weights_default():
#     c = Circuit()
#     assert c.get_weights() == {}

# def test_get_inputs_from_file():
#     c = Circuit()
#     c.scale_base = 2
#     c.scaling = 2
#     with patch('python_testing.circuit_components.circuit_helpers.read_from_json', return_value = {"input":[1,2,3,4]}):
#         x = c.get_inputs_from_file("", is_scaled=True)
#         assert x == {"input":[1,2,3,4]}

#         y = c.get_inputs_from_file("", is_scaled=False)
#         assert y == {"input":[4,8,12,16]}

# def test_get_inputs_from_file_multiple_inputs():
#     c = Circuit()
#     c.scale_base = 2
#     c.scaling = 2
#     with patch('python_testing.circuit_components.circuit_helpers.read_from_json', return_value = {"input":[1,2,3,4], "nonce": 25}):
#         x = c.get_inputs_from_file("", is_scaled=True)
#         assert x == {"input":[1,2,3,4], "nonce": 25}

#         y = c.get_inputs_from_file("", is_scaled=False)
#         assert y == {"input":[4,8,12,16], "nonce": 100}

# def test_get_inputs_from_file_dne():
#     c = Circuit()
#     c.scale_base = 2
#     c.scaling = 2
#     with pytest.raises(FileNotFoundError, match="No such file or directory"):
#         c.get_inputs_from_file("", is_scaled=True)
    
# def test_format_outputs():
#     c = Circuit()
#     out = c.format_outputs([10,15,20])
#     assert out == {"output":[10,15,20]}


# # ---------- _gen_witness_preprocessing ----------

# @patch("python_testing.circuit_components.circuit_helpers.to_json")
# def test_gen_witness_preprocessing_write_json_true(mock_to_json):
#     c = Circuit()
#     c._file_info = {"quantized_model_path": "quant.pt"}
#     c.load_quantized_model = MagicMock()
#     c.get_inputs = MagicMock(return_value="inputs")
#     c.get_outputs = MagicMock(return_value="outputs")
#     c.format_inputs = MagicMock(return_value={"input": 1})
#     c.format_outputs = MagicMock(return_value={"output": 2})

#     c._gen_witness_preprocessing("in.json", "out.json", None, write_json=True, is_scaled=True)

#     c.load_quantized_model.assert_called_once_with("quant.pt")
#     c.get_inputs.assert_called_once()
#     c.get_outputs.assert_called_once_with("inputs")
#     mock_to_json.assert_any_call({"input": 1}, "in.json")
#     mock_to_json.assert_any_call({"output": 2}, "out.json")


# @patch("python_testing.circuit_components.circuit_helpers.to_json")
# def test_gen_witness_preprocessing_write_json_false(mock_to_json):
#     c = Circuit()
#     c._file_info = {"quantized_model_path": "quant.pt"}
#     c.load_quantized_model = MagicMock()
#     c.get_inputs_from_file = MagicMock(return_value="mock_inputs")
#     c.reshape_inputs = MagicMock(return_value="in.json")
#     c.rescale_inputs = MagicMock(return_value="in.json")
#     c.rename_inputs = MagicMock(return_value="in.json")
#     c.rescale_and_reshape_inputs = MagicMock(return_value="in.json")
#     c.adjust_inputs = MagicMock(return_value="in.json")



#     c.get_outputs = MagicMock(return_value="mock_outputs")
#     c.format_outputs = MagicMock(return_value={"output": 99})

#     c._gen_witness_preprocessing("in.json", "out.json", None, write_json=False, is_scaled=False)

#     c.load_quantized_model.assert_called_once_with("quant.pt")
#     c.get_inputs_from_file.assert_called_once_with("in.json", is_scaled=False)
#     c.get_outputs.assert_called_once_with("mock_inputs")
#     c.format_outputs.assert_called_once_with("mock_outputs")
#     mock_to_json.assert_called_once_with({"output": 99}, "out.json")


# # ---------- _compile_preprocessing ----------

# @patch("python_testing.circuit_components.circuit_helpers.to_json")
# def test_compile_preprocessing_weights_dict(mock_to_json):
#     c = Circuit()
#     c._file_info = {"quantized_model_path": "model.pth"}
#     c.get_model_and_quantize = MagicMock()
#     c.get_weights = MagicMock(return_value={"a": 1})
#     c.save_quantized_model = MagicMock()

#     c._compile_preprocessing("weights.json", None)

#     c.get_model_and_quantize.assert_called_once()
#     c.get_weights.assert_called_once()
#     c.save_quantized_model.assert_called_once_with("model.pth")
#     mock_to_json.assert_called_once_with({"a": 1}, "weights.json")


# @patch("python_testing.circuit_components.circuit_helpers.to_json")
# def test_compile_preprocessing_weights_list(mock_to_json):
#     c = Circuit()
#     c._file_info = {"quantized_model_path": "model.pth"}
#     c.get_model_and_quantize = MagicMock()
#     c.get_weights = MagicMock(return_value=[{"w1": 1}, {"w2": 2}, {"w3": 3}])
#     c.save_quantized_model = MagicMock()

#     c._compile_preprocessing("weights.json", None)

#     assert mock_to_json.call_count == 3
#     mock_to_json.assert_any_call({"w1": 1}, "weights.json")
#     mock_to_json.assert_any_call({"w2": 2}, "weights2.json")
#     mock_to_json.assert_any_call({"w3": 3}, "weights3.json")


# def test_compile_preprocessing_raises_on_bad_weights():
#     c = Circuit()
#     c._file_info = {"quantized_model_path": "model.pth"}
#     c.get_model_and_quantize = MagicMock()
#     c.get_weights = MagicMock(return_value="bad_type")
#     c.save_quantized_model = MagicMock()

#     with pytest.raises(NotImplementedError, match="Weights type is incorrect"):
#         c._compile_preprocessing("weights.json", None)

# # ---------- Test check attributes --------------
# def test_check_attributes_true():
#     c = Circuit()
#     c.required_keys = ["input"]
#     c.name = "test"
#     c.scaling = 2
#     c.scale_base = 2
#     c.check_attributes()

# def test_check_attributes_no_scaling():
#     c = Circuit()
#     c.required_keys = ["input"]
#     c.name = "test"
#     c.scale_base = 2
#     with pytest.raises(NotImplementedError) as exc_info:
#         c.check_attributes()

#     msg = str(exc_info.value)
#     assert "Subclasses must define" in msg
#     assert "'scaling'" in msg


# def test_check_attributes_no_scalebase():
#     c = Circuit()
#     c.required_keys = ["input"]
#     c.name = "test"
#     c.scaling = 2

#     with pytest.raises(NotImplementedError) as exc_info:
#         c.check_attributes()

#     msg = str(exc_info.value)
#     assert "Subclasses must define" in msg
#     assert "'scale_base'" in msg

# def test_check_attributes_no_name():
#     c = Circuit()
#     c.required_keys = ["input"]
#     c.scale_base = 2
#     c.scaling = 2

#     with pytest.raises(NotImplementedError) as exc_info:
#         c.check_attributes()

#     msg = str(exc_info.value)
#     assert "Subclasses must define" in msg
#     assert "'name'" in msg


# # ---------- base_testing ------------

# @patch.object(Circuit, "parse_proof_run_type")
# def test_base_testing_calls_parse_proof_run_type_correctly(mock_parse):
#     c = Circuit()
#     c.name = "test"

#     c._file_info = {}
#     c._file_info["weights"] = "weights/model_weights.json"
#     c.base_testing(
#         run_type=RunType.GEN_WITNESS,
#         witness_file="w.wtns",
#         input_file="i.json",
#         proof_file="p.json",
#         public_path="pub.json",
#         verification_key="vk.key",
#         circuit_name="circuit_model",
#         output_file="o.json",
#         circuit_path="circuit_path.txt",
#         quantized_path="quantized_path.pt",
#         write_json=True
#     )

#     mock_parse.assert_called_once()
#     args = mock_parse.call_args[0]
#     expected_args = ("w.wtns", 
#                     "i.json", 
#                     "p.json", 
#                     "pub.json",
#                     "vk.key", 
#                     "circuit_model", 
#                     "circuit_path.txt",
#                     None, 
#                     "o.json", 
#                     "weights/model_weights.json", 
#                     "quantized_path.pt", 
#                     RunType.GEN_WITNESS,
#                     False, 
#                     True, 
#                     True,
#                     False
#     )
    
#     assert args == expected_args
#     assert args[0] == "w.wtns"
#     assert args[1] == "i.json"
#     assert args[2] == "p.json"
#     assert args[3] == "pub.json"
#     assert args[4] == "vk.key"
#     assert args[5] == "circuit_model"
#     assert args[6] == "circuit_path.txt"
#     assert args[7] is None  # proof_system not specified
#     assert args[8] == "o.json"
#     assert args[9] == "weights/model_weights.json"
#     assert args[10] == "quantized_path.pt"
#     assert args[11] == RunType.GEN_WITNESS
#     assert args[12] is False
#     assert args[13] is True
#     assert args[14] is True

# @patch.object(Circuit, "parse_proof_run_type")
# def test_base_testing_uses_default_circuit_path(mock_parse):
#     class MyCircuit(Circuit):
#         def __init__(self):
#             super().__init__()
#             self._file_info = {"weights": "weights.json"}
            

#     c = MyCircuit()
#     c.base_testing(circuit_name="test_model")

#     mock_parse.assert_called_once()
#     args = mock_parse.call_args[0]
#     assert args[6] == "test_model.txt"  # default circuit_path


# @patch.object(Circuit, "parse_proof_run_type")
# def test_base_testing_returns_none(mock_parse):
#     class MyCircuit(Circuit):
#         def __init__(self):
#             super().__init__()
#             self._file_info = {"weights": "some_weights.json"}

#     c = MyCircuit()
#     result = c.base_testing(circuit_name="abc")
#     assert result is None
#     mock_parse.assert_called_once()

# @patch.object(Circuit, "parse_proof_run_type")
# def test_base_testing_weights_exists(mock_parse):
#     class MyCircuit(Circuit):
#         def __init__(self):
#             super().__init__()

#     c = MyCircuit()
#     with pytest.raises(KeyError, match="_file_info"):
#         result = c.base_testing(circuit_name="abc")


# def test_parse_proof_run_type_invalid_run_type(capsys):
#     c = Circuit()

#     c.parse_proof_run_type(
#         "w.wtns", "i.json", "p.json", "pub.json",
#         "vk.key", "model", "path.txt", None, "out.json",
#         "weights.json", "quantized_model.pt","NOT_A_REAL_RUN_TYPE"
#     )
#     captured = capsys.readouterr()
#     assert "Unknown entry: NOT_A_REAL_RUN_TYPE" in captured.out
#     assert "Warning: Operation NOT_A_REAL_RUN_TYPE failed: Unknown run type: NOT_A_REAL_RUN_TYPE" in captured.out
#     assert "Input and output files have still been created correctly." in captured.out


# # @patch.object(Circuit, "parse_proof_run_type", side_effect = Exception("Boom!"))
# @patch("python_testing.circuit_components.circuit_helpers.compile_circuit", side_effect=Exception("Boom goes the dynamite!"))
# @patch.object(Circuit, "_compile_preprocessing")
# def test_parse_proof_run_type_catches_internal_exception(mock_compile_preprocessing, mock_compile, capsys):
#     c = Circuit()

#     # This will raise inside `compile_circuit`, which is patched to raise
#     c.parse_proof_run_type(
#         "w.wtns", "i.json", "p.json", "pub.json",
#         "vk.key", "model", "path.txt", None, "out.json",
#         "weights.json", "quantized_path.pt", RunType.COMPILE_CIRCUIT
#     )

#     captured = capsys.readouterr()
#     print(captured.out)
#     assert "Warning: Operation RunType.COMPILE_CIRCUIT failed: Boom goes the dynamite!" in captured.out
#     assert "Input and output files have still been created correctly." in captured.out


# def test_save_and_load_model_not_implemented():
#     c = Circuit()
#     assert hasattr(c, "save_model")
#     assert hasattr(c, "load_model")
#     assert hasattr(c, "save_quantized_model")
#     assert hasattr(c, "load_quantized_model")

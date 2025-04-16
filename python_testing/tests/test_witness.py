# import subprocess
# from unittest import mock
# import pytest
# import torch
# from python_testing.utils.run_proofs import ZKProofSystems
# from python_testing.circuit_models.simple_circuit import SimpleCircuit, RunType


# proof_system = ZKProofSystems.Expander
# proof_folder = "analysis"
# output_folder = "output"
# temp_folder = "temp"
# input_folder = "inputs"
# weights_folder = "weights"
# circuit_folder = ""

# def test_simple_circuit_run():
#     SimpleCircuit().base_testing(run_type=RunType.COMPILE_CIRCUIT)

#     SimpleCircuit().base_testing(run_type=RunType.GEN_WITNESS) # This should specify the model_params file


#     SimpleCircuit().base_testing(run_type=RunType.PROVE_WITNESS) # This should specify the model_params file
    
#     SimpleCircuit().base_testing(run_type=RunType.GEN_VERIFY)

# def test_simple_circuit_incorrect_output(capsys):
#     SimpleCircuit().base_testing(run_type=RunType.COMPILE_CIRCUIT)

#     SimpleCircuit().base_testing(run_type=RunType.GEN_WITNESS)


#     SimpleCircuit().base_testing(run_type=RunType.PROVE_WITNESS) 
    
#     SimpleCircuit().base_testing(run_type=RunType.GEN_VERIFY)

#     SimpleCircuit().base_testing(run_type=RunType.GEN_WITNESS) 

#     # with pytest.raises(subprocess.CalledProcessError) as exc_info:

#     # This should be an error
#     SimpleCircuit().base_testing(run_type=RunType.GEN_VERIFY) 
#     error_message = "assertion failed: "
#     # assert error_message in exc_info.value.stderr, f"Expected to find '{error_message}' in stderr, but got: {exc_info.value.stderr}" 
#     captured = capsys.readouterr()
#     assert error_message in captured.out
#     assert "Warning: Verification generation failed:" in captured.out


#         # Check stderr for the panic message or error message
    
import subprocess
from unittest import mock
import pytest
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.circuit_components.convolution import Convolution, QuantizedConv


proof_system = ZKProofSystems.Expander
proof_folder = "analysis"
output_folder = "output"
temp_folder = "temp"
input_folder = "inputs"
weights_folder = "weights"
circuit_folder = ""


def convolution_incorrect_output(self):
    x =  torch.from_numpy(Convolution.conv_run(self, self.input_arr, self.weights, self.bias, "NOTSET",self.dilation, self.group, self.kernel_shape,self.pads, self.strides))
    x[0][0][0][0] = x[0][0][0][0] + 1
    return x

def test_convolution_base_run():
    test_circuit = Convolution()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

def test_convolution_base_quantize_run():
    test_circuit = QuantizedConv()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)



def test_convolution_incorrect_output():
    test_circuit = Convolution()
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        with mock.patch.object(Convolution, 'get_outputs', side_effect=convolution_incorrect_output, autospec=True) as mock_get_output:
            test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    assert exc_info is not None, "Expected subprocess.CalledProcessError to be raised, but it was not."
    # Check stderr for the panic message or error message
    error_message = "assertion `left == right` failed"
    assert error_message in exc_info.value.stderr, f"Expected to find '{error_message}' in stderr, but got: {exc_info.value.stderr}"

    # Optionally check the stdout if necessary
    expected_stdout = "built layered circuit"
    assert expected_stdout in exc_info.value.stdout, f"Expected to find '{expected_stdout}' in stdout, but got: {exc_info.value.stdout}"


def test_quantized_convolution_incorrect_output():
    test_circuit = QuantizedConv()
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        with mock.patch.object(QuantizedConv, 'get_outputs', side_effect=convolution_incorrect_output, autospec=True) as mock_get_output:
            test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    assert exc_info is not None, "Expected subprocess.CalledProcessError to be raised, but it was not."
    # Check stderr for the panic message or error message
    error_message = "assertion `left == right` failed"
    assert error_message in exc_info.value.stderr, f"Expected to find '{error_message}' in stderr, but got: {exc_info.value.stderr}"

    # Optionally check the stdout if necessary
    expected_stdout = "built layered circuit"
    assert expected_stdout in exc_info.value.stdout, f"Expected to find '{expected_stdout}' in stdout, but got: {exc_info.value.stdout}"
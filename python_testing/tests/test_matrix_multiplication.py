import subprocess
from unittest import mock
import pytest
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.circuit_components.matrix_multiplication import MatrixMultiplication, QuantizedMatrixMultiplication, QuantizedMatrixMultiplicationReLU

proof_system = ZKProofSystems.Expander
proof_folder = "analysis"
output_folder = "output"
temp_folder = "temp"
input_folder = "inputs"
weights_folder = "weights"
circuit_folder = ""


def mat_mult_incorrect_output(self: MatrixMultiplication):
    x = torch.matmul(self.matrix_a, self.matrix_b)
        # return mat_mult
    x[0][0] = x[0][0] + 1
    return (self.matrix_a, self.matrix_b, x)

def mat_mult_incorrect_output(self: MatrixMultiplication):

    matrix_a = torch.mul(self.matrix_a, 2**self.scaling).long()
    matrix_b = torch.mul(self.matrix_b, 2**self.scaling).long()
    matrix_product_ab = torch.matmul(matrix_a, matrix_b)
    matrix_product_ab = torch.div(matrix_product_ab, 2**self.scaling, rounding_mode="floor").long()

    matrix_product_ab[0][0] = matrix_product_ab[0][0] + 1

    return (matrix_a, matrix_b, matrix_product_ab)

def test_mat_mult_base_run():
    test_circuit = MatrixMultiplication()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

def test_mat_mult_base_quantize_run():
    test_circuit = QuantizedMatrixMultiplication()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

def test_mat_mult_base_quantize_run_ones():
    test_circuit = QuantizedMatrixMultiplication()
    test_circuit.matrix_a = torch.ones(test_circuit.matrix_a.shape)
    test_circuit.matrix_b = torch.ones(test_circuit.matrix_b.shape)
    # With input of all ones, and this size circuit, we need to reduce the scaling factor to work with the built in 32 v value
    test_circuit.scaling = 20

    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

def test_mat_mult_base_quantize_run_negative_ones():
    test_circuit = QuantizedMatrixMultiplication()
    test_circuit.matrix_a = torch.full(test_circuit.matrix_a.shape, -1)
    test_circuit.matrix_b = torch.full(test_circuit.matrix_b.shape, -1)
    # With input of all ones, and this size circuit, we need to reduce the scaling factor to work with the built in 32 v value
    test_circuit.scaling = 20

    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)



def test_mat_mult_incorrect_output():
    test_circuit = MatrixMultiplication()
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        with mock.patch.object(MatrixMultiplication, 'get_outputs', side_effect=mat_mult_incorrect_output, autospec=True) as mock_get_output:
            test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    assert exc_info is not None, "Expected subprocess.CalledProcessError to be raised, but it was not."
    # Check stderr for the panic message or error message
    error_message = "assertion `left == right` failed"
    assert error_message in exc_info.value.stderr, f"Expected to find '{error_message}' in stderr, but got: {exc_info.value.stderr}"

    # Optionally check the stdout if necessary
    expected_stdout = "built layered circuit"
    assert expected_stdout in exc_info.value.stdout, f"Expected to find '{expected_stdout}' in stdout, but got: {exc_info.value.stdout}"


def test_quantized_mat_mult_incorrect_output():
    test_circuit = QuantizedMatrixMultiplication()
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        with mock.patch.object(QuantizedMatrixMultiplication, 'get_outputs', side_effect=mat_mult_incorrect_output, autospec=True) as mock_get_output:
            test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    assert exc_info is not None, "Expected subprocess.CalledProcessError to be raised, but it was not."
    # Check stderr for the panic message or error message
    error_message = "assertion `left == right` failed"
    assert error_message in exc_info.value.stderr, f"Expected to find '{error_message}' in stderr, but got: {exc_info.value.stderr}"

    # Optionally check the stdout if necessary
    expected_stdout = "built layered circuit"
    assert expected_stdout in exc_info.value.stdout, f"Expected to find '{expected_stdout}' in stdout, but got: {exc_info.value.stdout}"

def test_mat_mult_base_quantize_and_relu_run():
    test_circuit = QuantizedMatrixMultiplicationReLU()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)


def test_quantized_mat_mult_relu_incorrect_output():
    test_circuit = QuantizedMatrixMultiplicationReLU()
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        with mock.patch.object(QuantizedMatrixMultiplicationReLU, 'get_inputs_for_circuit', side_effect=mat_mult_incorrect_output, autospec=True) as mock_get_output:
            test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    assert exc_info is not None, "Expected subprocess.CalledProcessError to be raised, but it was not."
    # Check stderr for the panic message or error message
    error_message = "assertion `left == right` failed"
    assert error_message in exc_info.value.stderr, f"Expected to find '{error_message}' in stderr, but got: {exc_info.value.stderr}"

    # Optionally check the stdout if necessary
    expected_stdout = "built layered circuit"
    assert expected_stdout in exc_info.value.stdout, f"Expected to find '{expected_stdout}' in stdout, but got: {exc_info.value.stdout}"

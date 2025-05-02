import os
import numpy as np
import pytest
import torch

from python_testing.circuit_models.demo_cnn import Demo
from python_testing.circuit_models.doom_model import Doom
from python_testing.circuit_models.simple_circuit import SimpleCircuit
from python_testing.circuit_models.doom_slices import DoomConv1, DoomConv2, DoomConv3, DoomFC1, DoomFC2
from python_testing.circuit_models.eth_fraud import Eth
from python_testing.circuit_components.convolution import Convolution, QuantizedConv, QuantizedConvRelu
from python_testing.circuit_components.relu import ReLU
from python_testing.circuit_components.matrix_multiplication import MatrixMultiplication, QuantizedMatrixMultiplication, QuantizedMatrixMultiplicationReLU
from python_testing.circuit_components.matrix_addition import MatrixAddition
from python_testing.circuit_components.scaled_matrix_product import ScaledMatrixProduct
from python_testing.circuit_components.scaled_matrix_product_sum import ScaledMatrixProductSum


GOOD_OUTPUT = ["Witness Generated"]
BAD_OUTPUT = ["assertion `left == right` failed", "Witness generation failed"]

MODELS_TO_TEST = [
    ("doom", Doom),
    ("simple_circuit", SimpleCircuit),
    ("cnn_demo", Demo),
    # ("doom_conv1", DoomConv1),
    # ("doom_conv2", DoomConv2),
    # ("doom_conv3", DoomConv3),
    # ("doom_fc1", DoomFC1),
    # ("doom_fc2", DoomFC2),
    # ("eth_fraud", Eth),
    # ("quantized_conv", QuantizedConv),
    # ("quantized_conv_relu", QuantizedConvRelu),
    # ("quantized_matrix_multiplication", QuantizedMatrixMultiplication),
    # ("quantized_matrix_multiplication_relu", QuantizedMatrixMultiplicationReLU),
    # ("matrix_addition", MatrixAddition),
    # ("scaled_matrix_product", ScaledMatrixProduct),
    # ("scaled_matrix_product_sum", ScaledMatrixProductSum),
    # # ("other_model", OtherModel),
]

@pytest.fixture
def temp_witness_file(tmp_path):
    witness_path = tmp_path / "temp_witness.txt"
    # Give it to the test
    yield witness_path

    # After the test is done, remove it
    if os.path.exists(witness_path):
        witness_path.unlink()

@pytest.fixture
def temp_input_file(tmp_path):
    input_path = tmp_path / "temp_input.txt"
    # Give it to the test
    yield input_path

    # After the test is done, remove it
    if os.path.exists(input_path):
        input_path.unlink()

@pytest.fixture
def temp_output_file(tmp_path):
    output_path = tmp_path / "temp_output.txt"
    # Give it to the test
    yield output_path

    # After the test is done, remove it
    if os.path.exists(output_path):
        output_path.unlink()

def add_1_to_first_element(x):
    """Safely adds 1 to the first element of any scalar/list/tensor."""
    if isinstance(x, (int, float)):
        return x + 1
    elif isinstance(x, torch.Tensor):
        x = x.clone()  # avoid in-place modification
        x.view(-1)[0] += 1
        return x
    elif isinstance(x, (list, tuple, np.ndarray)):
        x = list(x)
        x[0] = add_1_to_first_element(x[0])
        return x
    else:
        raise TypeError(f"Unsupported type for get_outputs patch: {type(x)}")

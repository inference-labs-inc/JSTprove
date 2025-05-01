import os
import numpy as np
import pytest
import torch

from python_testing.circuit_models.demo_cnn import Demo
from python_testing.circuit_models.doom_model import Doom
from python_testing.circuit_models.simple_circuit import SimpleCircuit

GOOD_OUTPUT = ["Witness Generated"]
BAD_OUTPUT = ["assertion `left == right` failed", "Warning: Witness generation failed"]

MODELS_TO_TEST = [
    # ("doom", Doom),
    ("simple_circuit", SimpleCircuit),
    # ("cnn_demo", Demo)

    # ("other_model", OtherModel),
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

import os
import numpy as np
import pytest
import torch

# from core.circuit_models.eth_fraud import Eth
from python.core.utils.helper_functions import RunType


GOOD_OUTPUT = ["Witness Generated"]
BAD_OUTPUT = ["assertion `left == right` failed", "Witness generation failed"]

@pytest.fixture(scope="module")
def model_fixture(request, tmp_path_factory):
    param = request.param
    name = f"{param.name}"
    model_class = param.loader
    args, kwargs = (), {}

    if len(param) == 3:
        if isinstance(param[2], dict):
            kwargs = param[2]
        else:
            args = param[2]
    elif len(param) == 4:
        args, kwargs = param[2], param[3]

    temp_dir = tmp_path_factory.mktemp(name)
    circuit_path = temp_dir / f"{name}_circuit.txt"
    quantized_path = temp_dir / f"{name}_quantized.pt"

    model = model_class(*args, **kwargs)

    model.base_testing(
        run_type=RunType.COMPILE_CIRCUIT,
        dev_mode=True,
        circuit_path=str(circuit_path),
        quantized_path = quantized_path
    )

    return {
        "name": name,
        "model_class": model_class,
        "circuit_path": circuit_path,
        "temp_dir": temp_dir,
        "model": model,
        "quantized_model": quantized_path, 
    }

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

@pytest.fixture
def temp_proof_file(tmp_path):
    output_path = tmp_path / "temp_proof.txt"
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
    

# Define models to be tested
circuit_compile_results = {}
witness_generated_results = {}

@pytest.fixture(scope="module")
def check_model_compiles(model_fixture):
    # Default to True; will be set to False if first test fails
    result = circuit_compile_results.get(model_fixture['model'])
    if result is False:
        pytest.skip(f"Skipping because the first test failed for param: {model_fixture['model']}")

@pytest.fixture(scope="module")
def check_witness_generated(model_fixture):
    # Default to True; will be set to False if first test fails
    result = witness_generated_results.get(model_fixture['model'])
    if result is False:
        pytest.skip(f"Skipping because the first test failed for param: {model_fixture['model']}")

def assert_very_close(inputs_1, inputs_2, model):
    for i in inputs_1.keys():
        x = torch.div(torch.as_tensor(inputs_1[i]), model.scale_base**model.scale_exponent)
        y = torch.div(torch.as_tensor(inputs_2[i]), model.scale_base**model.scale_exponent)

        assert torch.isclose(x, y, rtol = 1e-8).all()

import os
import pytest

# Assume these are your models
from python_testing.circuit_models.doom_model import Doom
from python_testing.circuit_models.simple_circuit import SimpleCircuit



# Enums, utils
from python_testing.utils.helper_functions import RunType

# Define models to be tested
MODELS_TO_TEST = [
    # ("doom", Doom),
    ("simple_circuit", SimpleCircuit),

    # ("other_model", OtherModel),
]

GOOD_OUTPUT = ["Witness Generated"]
BAD_OUTPUT = ["assertion `left == right` failed", "Warning: Witness generation failed"]

@pytest.fixture(scope="module", params=MODELS_TO_TEST)
def model_fixture(request, tmp_path_factory):
    """Compile circuit once per model and provide model instance and paths."""
    name, model_class = request.param
    temp_dir = tmp_path_factory.mktemp(name)
    circuit_path = temp_dir / f"{name}_circuit.txt"
    witness_path = temp_dir / f"{name}_witness.txt"
    
    # Compile once
    model = model_class()
    model.base_testing(
        run_type=RunType.COMPILE_CIRCUIT, 
        dev_mode=True,
        circuit_path=str(circuit_path)
    )
    
    return {
        "name": name,
        "model_class": model_class,
        "circuit_path": circuit_path,
        "witness_path": witness_path,
        "temp_dir": temp_dir
    }

@pytest.fixture
def temp_witness_file(tmp_path):
    witness_path = tmp_path / "temp_witness.txt"
    # print("TESTETS", os.path.exists(witness_path))

    # Create an empty file to simulate usage
    # witness_path.touch()
    
    # Give it to the test
    yield witness_path

    # After the test is done, remove it
    if os.path.exists(witness_path):
        witness_path.unlink()


def test_circuit_compiles(model_fixture):
    # Here you could just check that circuit file exists
    assert os.path.exists(model_fixture["circuit_path"])

def test_witness_dev(model_fixture, capsys, temp_witness_file):
    model = model_fixture["model_class"]()
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file= temp_witness_file,
        # circuit_path=str("eth_circuit.txt"),
        circuit_path=str(model_fixture["circuit_path"]),
        write_json=False
    )

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert os.path.exists(temp_witness_file)
    assert "Running cargo command:" in stdout
    for output in GOOD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."




def test_witness_wrong_circuit_dev(model_fixture, capsys, temp_witness_file):
    model = model_fixture["model_class"]()
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str("eth_circuit.txt"),
        # circuit_path=str(model_fixture["circuit_path"]),
        write_json=False
    )

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)

    assert not os.path.exists(temp_witness_file)
    assert "Running cargo command:" in stdout
    for output in GOOD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."
    for output in BAD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."


# def test_witness_prove_verify_true_inputs_dev(model_fixture):
#     model = model_fixture["model_class"]()
#     model.base_testing(
#         run_type=RunType.PROVE_WITNESS,
#         dev_mode=False,
#         witness_file=str(model_fixture["witness_path"]),
#         circuit_path=str(model_fixture["circuit_path"]),
#     )
#     model.base_testing(
#         run_type=RunType.GEN_VERIFY,
#         dev_mode=False,
#         witness_file=str(model_fixture["witness_path"]),
#         circuit_path=str(model_fixture["circuit_path"]),
#     )

# def test_circuit_compiles():
#     pass

# def test_witness_dev():
#     pass

# def test_witness_read_inputs_dev():
#     pass

# def test_witness_false_inputs_dev():
#     pass

# def test_witness_prove_verify_true_inputs_dev():
#     pass

# def test_witness_prove_verify_false_inputs_dev():
#     pass

# def test_witness_fresh_compile_dev():
#     pass

# def test_compile_witness_unique_weights():
#     pass
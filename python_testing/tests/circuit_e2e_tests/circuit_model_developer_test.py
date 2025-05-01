import json
import os
import pytest

# Assume these are your models


from python_testing.tests.circuit_e2e_tests.helper_fns_for_tests import *



# Enums, utils
from python_testing.utils.helper_functions import RunType

# Define models to be tested



@pytest.fixture(scope="module", params=MODELS_TO_TEST)
def model_fixture(request, tmp_path_factory):
    """Compile circuit once per model and provide model instance and paths."""
    name, model_class = request.param
    temp_dir = tmp_path_factory.mktemp(name)
    circuit_path = temp_dir / f"{name}_circuit.txt"
    
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
        "temp_dir": temp_dir
    }

def test_circuit_compiles(model_fixture):
    # Here you could just check that circuit file exists
    assert os.path.exists(model_fixture["circuit_path"])

def test_witness_dev(model_fixture, capsys, temp_witness_file, temp_input_file, temp_output_file):
    model = model_fixture["model_class"]()
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file= temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        write_json=True
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




def test_witness_wrong_outputs_dev(model_fixture, capsys, temp_witness_file, temp_input_file, temp_output_file, monkeypatch):
    model = model_fixture["model_class"]()
    original_get_outputs = model.get_outputs

    def patched_get_outputs(*args, **kwargs):
        result = original_get_outputs(*args, **kwargs)
        return add_1_to_first_element(result)

    monkeypatch.setattr(model, "get_outputs", patched_get_outputs)

    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file= temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        write_json=True
    )

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)
    # assert False

    assert not os.path.exists(temp_witness_file)
    assert "Running cargo command:" in stdout
    for output in GOOD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."
    for output in BAD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."

def test_witness_prove_verify_true_inputs_dev(model_fixture, temp_witness_file, temp_input_file, temp_output_file):
    model = model_fixture["model_class"]()
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file= temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        write_json=True
    )
    model.base_testing(
        run_type=RunType.PROVE_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
    )
    model.base_testing(
        run_type=RunType.GEN_VERIFY,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
    )


def test_witness_read_after_write_json(
    model_fixture,
    capsys,
    temp_witness_file,
    temp_input_file,
    temp_output_file
):
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model_class"]()
    model_write.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=True
    )

    if os.path.exists(temp_witness_file):
        os.remove(temp_witness_file)
    assert not os.path.exists(temp_witness_file)

    # Optional: Load the written input for inspection
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model_class"]()
    model_read.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=False
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert os.path.exists(temp_witness_file), "Witness file not generated"
    assert "Running cargo command:" in stdout

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."

    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

    # Optional: verify that input file content was actually read
    with open(temp_input_file, "r") as f:
        read_input_data = f.read()
    assert read_input_data == written_input_data, "Input JSON read is not identical to what was written"

def test_witness_fresh_compile_dev():
    pass


# Use once fixed input shape read in rust
# def test_witness_incorrect_input_shape(
#     model_fixture,
#     capsys,
#     temp_witness_file,
#     temp_input_file,
#     temp_output_file
# ):
#     # Step 1: Write the input file via write_json=True
#     model_write = model_fixture["model_class"]()
#     model_write.base_testing(
#         run_type=RunType.GEN_WITNESS,
#         dev_mode=False,
#         witness_file=temp_witness_file,
#         circuit_path=str(model_fixture["circuit_path"]),
#         input_file=temp_input_file,
#         output_file=temp_output_file,
#         write_json=True
#     )
#     if os.path.exists(temp_witness_file):
#         os.remove(temp_witness_file)
#     assert not os.path.exists(temp_witness_file)

    # # Optional: Load the written input for inspection
    # with open(temp_input_file, "r") as f:
    #     written_input_data = f.read()
    # input_data = json.loads(written_input_data)
    # for key in input_data:
    #     if isinstance(input_data[key], list):
    #         input_data[key] = torch.as_tensor(input_data[key]).flatten().tolist()
    #     assert torch.as_tensor(input_data[key]).dim() <= 1, f"Input data for {key} is not 1D tensor. This is a testing error, not a model error. Please fix this test to properly flatten."
    # with open(temp_input_file, "w") as f:
    #     json.dump(input_data, f)

#     # Step 2: Read from that same input file (write_json=False)
#     model_read = model_fixture["model_class"]()
#     model_read.base_testing(
#         run_type=RunType.GEN_WITNESS,
#         dev_mode=False,
#         witness_file=temp_witness_file,
#         circuit_path=str(model_fixture["circuit_path"]),
#         input_file=temp_input_file,
#         output_file=temp_output_file,
#         write_json=False
#     )

#     # Step 3: Validate expected outputs and no errors
#     captured = capsys.readouterr()
#     stdout = captured.out
#     stderr = captured.err

#     print(stdout)

#     assert os.path.exists(temp_witness_file), "Witness file not generated"
#     assert stdout.count("Running cargo command:") == 2, f"Expected 'Running cargo command: ' in stdout twice, but it was not found."

#     # Check good output appeared
#     for output in GOOD_OUTPUT:
#         assert stdout.count(output) == 2, f"Expected '{output}' in stdout, but it was not found."


#     # Ensure no unexpected errors
#     for output in BAD_OUTPUT:
#         assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

#     # Optional: verify that input file content was actually read
#     with open(temp_input_file, "r") as f:
#         read_input_data = f.read()
#     assert read_input_data == written_input_data, "Input JSON read is not identical to what was written"


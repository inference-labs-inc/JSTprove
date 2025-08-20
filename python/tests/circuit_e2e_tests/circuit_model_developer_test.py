import json
import os
import pytest

# Assume these are your models
# Enums, utils
from python.core.utils.helper_functions import RunType
from python.tests.circuit_e2e_tests.helper_fns_for_tests import *

@pytest.mark.e2e
def test_circuit_compiles(model_fixture):
    # Here you could just check that circuit file exists
    circuit_compile_results[model_fixture['model']] = False
    assert os.path.exists(model_fixture["circuit_path"])
    circuit_compile_results[model_fixture['model']] = True

@pytest.mark.e2e
def test_witness_dev(model_fixture, capsys, temp_witness_file, temp_input_file, temp_output_file, check_model_compiles):
    model = model_fixture["model"]
    witness_generated_results[model_fixture['model']] = False
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file= temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),
    )

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)
    # assert False

    assert os.path.exists(temp_witness_file)
    assert "Running cargo command:" in stdout
    for output in GOOD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

    witness_generated_results[model_fixture['model']] = True




@pytest.mark.e2e
def test_witness_wrong_outputs_dev(model_fixture, capsys, temp_witness_file, temp_input_file, temp_output_file, monkeypatch, check_model_compiles, check_witness_generated):
    model = model_fixture["model"]
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
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

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

@pytest.mark.e2e
def test_witness_prove_verify_true_inputs_dev(model_fixture, temp_witness_file, temp_input_file, temp_output_file, temp_proof_file, capsys, check_model_compiles, check_witness_generated):
    model = model_fixture["model"]
    print(model)
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file= temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

    )
    model.base_testing(
        run_type=RunType.PROVE_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        proof_file = temp_proof_file,
        quantized_path = str(model_fixture["quantized_model"]),

    )
    model.base_testing(
        run_type=RunType.GEN_VERIFY,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        proof_file = temp_proof_file,
        quantized_path = str(model_fixture["quantized_model"]),
    )
    # ASSERTIONS TODO

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)
    assert os.path.exists(temp_witness_file), "Witness file not generated"

    # Unexpected output
    assert stdout.count("poly.num_vars() == *params") == 0, f"'poly.num_vars() == *params' thrown. May need a dummy variable(s) to get rid of error. Dummy variables should be private variables. Can set = 1 in read_inputs and assert == 1 at end of circuit"
    assert stdout.count("Proof generation failed") == 0, f"Proof generation failed"
    assert os.path.exists(temp_proof_file), "Proof file not generated"

    assert stdout.count("Verification generation failed") == 0, f"Verification failed"
    # Expected output
    assert stdout.count("Running cargo command:") == 3, f"Expected 'Running cargo command: ' in stdout three times, but it was not found."
    assert stdout.count("Witness Generated") == 1, f"Expected 'Witness Generated' in stdout three times, but it was not found."

    assert stdout.count("proving") == 1, f"Expected 'proving' in stdout three times, but it was not found."
    assert stdout.count("Proved") == 1, f"Expected 'Proved' in stdout three times, but it was not found."

    assert stdout.count("Verified") == 1, f"Expected 'Verified' in stdout three times, but it was not found."

@pytest.mark.e2e
def test_witness_prove_verify_true_inputs_dev_expander_call(model_fixture, temp_witness_file, temp_input_file, temp_output_file, temp_proof_file, capsys,check_model_compiles, check_witness_generated):
    model = model_fixture["model"]
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file= temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),
    )
    model.base_testing(
        run_type=RunType.PROVE_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        proof_file = temp_proof_file,
        ecc = False,
        quantized_path = str(model_fixture["quantized_model"]),
    )
    model.base_testing(
        run_type=RunType.GEN_VERIFY,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        proof_file = temp_proof_file,
        ecc = False,
        quantized_path = str(model_fixture["quantized_model"]),
    )
    # ASSERTIONS TODO

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)
    print(stderr)
    # assert False
    assert os.path.exists(temp_witness_file), "Witness file not generated"


    # Unexpected output
    assert stdout.count("poly.num_vars() == *params") == 0, f"'poly.num_vars() == *params' thrown. May need a dummy variable(s) to get rid of error. Dummy variables should be private variables. Can set = 1 in read_inputs and assert == 1 at end of circuit"
    assert stdout.count("Proof generation failed") == 0, f"Proof generation failed"
    assert os.path.exists(temp_proof_file), "Proof file not generated"

    assert stdout.count("Verification generation failed") == 0, f"Verification failed"
    # Expected output
    assert stdout.count("Running cargo command:") == 1, f"Expected 'Running cargo command: ' in stdout once, but it was not found."
    assert stdout.count("Witness Generated") == 1, f"Expected 'Witness Generated' in stdout three times, but it was not found."

    assert stdout.count("proving") == 1, f"Expected 'proving' but it was not found."
    # assert stdout.count("Proved") == 1, f"Expected 'Proved' in stdout three times, but it was not found."

    assert stdout.count("verifying proof") == 1, f"Expected 'verifying proof' but it was not found."
    assert stdout.count("success") == 1, f"Expected 'success'  but it was not found."

    assert stdout.count("expander-exec verify succeeded") == 1, f"Expected 'expander-exec verify succeeded'  but it was not found."
    assert stdout.count("expander-exec prove succeeded") == 1, f"Expected 'expander-exec prove succeeded'  but it was not found."









@pytest.mark.e2e
def test_witness_read_after_write_json(
    model_fixture,
    capsys,
    temp_witness_file,
    temp_input_file,
    temp_output_file,
    check_model_compiles,
    check_witness_generated
):
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),
    )

    if os.path.exists(temp_witness_file):
        os.remove(temp_witness_file)
    assert not os.path.exists(temp_witness_file)

    # Optional: Load the written input for inspection
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=False,
        quantized_path = str(model_fixture["quantized_model"]),

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

@pytest.mark.e2e
def test_witness_fresh_compile_dev(capsys, model_fixture, temp_witness_file, temp_input_file, temp_output_file, check_model_compiles, check_witness_generated):
    model = model_fixture["model"]
    model.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=True,
        witness_file= temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file = temp_input_file,
        output_file = temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

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





# Use once fixed input shape read in rust
@pytest.mark.e2e
def test_witness_incorrect_input_shape(
    model_fixture,
    capsys,
    temp_witness_file,
    temp_input_file,
    temp_output_file,
    check_model_compiles,
    check_witness_generated
):
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

    )
    assert os.path.exists(temp_witness_file)
    os.remove(temp_witness_file)
    assert not os.path.exists(temp_witness_file)

    # Optional: Load the written input for inspection
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    for key in input_data:
        if isinstance(input_data[key], list):
            input_data[key] = torch.as_tensor(input_data[key]).flatten().tolist()
        assert torch.as_tensor(input_data[key]).dim() <= 1, f"Input data for {key} is not 1D tensor. This is a testing error, not a model error. Please fix this test to properly flatten."
    with open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=False,
        quantized_path = str(model_fixture["quantized_model"]),

    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert os.path.exists(temp_witness_file), "Witness file not generated"
    assert stdout.count("Running cargo command:") == 2, f"Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert stdout.count(output) == 2, f"Expected '{output}' in stdout, but it was not found."


    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

@pytest.mark.e2e
def test_witness_unscaled(
    model_fixture,
    capsys,
    temp_witness_file,
    temp_input_file,
    temp_output_file,
    check_model_compiles,
    check_witness_generated
):
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

    )
    if os.path.exists(temp_witness_file):
        os.remove(temp_witness_file)
    assert not os.path.exists(temp_witness_file)
    def contains_float(obj):
            if isinstance(obj, float):
                return True
            elif isinstance(obj, dict):
                return any(contains_float(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(contains_float(i) for i in obj)
            return False
    # Rescale
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    if hasattr(model_write, "scale_base") and hasattr(model_write, "scale_exponent"):
        for key in input_data:
            print(input_data[key])
            input_data[key] = torch.div(torch.as_tensor(input_data[key]), model_write.scale_base**model_write.scale_exponent).tolist()
            print(input_data[key])
    else:
        raise NotImplementedError("Model does not have scale_base attribute")
    assert contains_float(input_data), f"This is a testing error, not a model error. Please fix this test to properly turn data to float."
   
    with open(temp_output_file, "r") as f:
        written_output_data = f.read()
    os.remove(temp_output_file)
    
    with open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=False,
        quantized_path = str(model_fixture["quantized_model"]),

    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert os.path.exists(temp_witness_file), "Witness file not generated"
    assert stdout.count("Running cargo command:") == 2, f"Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert stdout.count(output) == 2, f"Expected '{output}' in stdout, but it was not found."


    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

    assert os.path.exists(temp_output_file), "Output file not generated"
    with open(temp_output_file, "r") as f:
        new_output_file = f.read()
    assert_very_close(json.loads(new_output_file), json.loads(written_output_data), model_write)
    # assert new_output_file == written_output_data, "Output file content does not match the expected output"


@pytest.mark.e2e
def test_witness_unscaled_and_incorrect_shape_input(
    model_fixture,
    capsys,
    temp_witness_file,
    temp_input_file,
    temp_output_file,
    check_model_compiles,
    check_witness_generated
):
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

    )
    if os.path.exists(temp_witness_file):
        os.remove(temp_witness_file)
    assert not os.path.exists(temp_witness_file)
    def contains_float(obj):
            if isinstance(obj, float):
                return True
            elif isinstance(obj, dict):
                return any(contains_float(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(contains_float(i) for i in obj)
            return False
    
    

    # flatten shape
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    for key in input_data:
        if isinstance(input_data[key], list):
            input_data[key] = torch.as_tensor(input_data[key]).flatten().tolist()
        assert torch.as_tensor(input_data[key]).dim() <= 1, f"Input data for {key} is not 1D tensor. This is a testing error, not a model error. Please fix this test to properly flatten."
    with open(temp_input_file, "w") as f:
        json.dump(input_data, f)
    # Rescale
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    if hasattr(model_write, "scale_base") and hasattr(model_write, "scale_exponent"):
        for key in input_data:
            input_data[key] = torch.div(torch.as_tensor(input_data[key]), model_write.scale_base**model_write.scale_exponent).tolist()
    else:
        raise NotImplementedError("Model does not have scale_base attribute")
    assert contains_float(input_data), f"This is a testing error, not a model error. Please fix this test to properly turn data to float."
   
    with open(temp_output_file, "r") as f:
        written_output_data = f.read()
    os.remove(temp_output_file)
    
    with open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Step 2: Read from that same input file (write_json=False) that has been rescaled and flattened
    model_read = model_fixture["model"]
    model_read.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=False,
        quantized_path = str(model_fixture["quantized_model"]),

    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert os.path.exists(temp_witness_file), "Witness file not generated"
    assert stdout.count("Running cargo command:") == 2, f"Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert stdout.count(output) == 2, f"Expected '{output}' in stdout, but it was not found."


    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

    assert os.path.exists(temp_output_file), "Output file not generated"
    with open(temp_output_file, "r") as f:
        new_output_file = f.read()
    assert_very_close(json.loads(new_output_file), json.loads(written_output_data), model_write)

    # assert new_output_file == written_output_data, "Output file content does not match the expected output"

@pytest.mark.e2e
def test_witness_unscaled_and_incorrect_and_bad_named_input(
    model_fixture,
    capsys,
    temp_witness_file,
    temp_input_file,
    temp_output_file,
    check_model_compiles,
    check_witness_generated
):
    
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

    )
    if os.path.exists(temp_witness_file):
        os.remove(temp_witness_file)
    assert not os.path.exists(temp_witness_file)
    def contains_float(obj):
            if isinstance(obj, float):
                return True
            elif isinstance(obj, dict):
                return any(contains_float(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(contains_float(i) for i in obj)
            return False
    
    

    # flatten shape
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    for key in input_data:
        if isinstance(input_data[key], list):
            input_data[key] = torch.as_tensor(input_data[key]).flatten().tolist()
        assert torch.as_tensor(input_data[key]).dim() <= 1, f"Input data for {key} is not 1D tensor. This is a testing error, not a model error. Please fix this test to properly flatten."
    with open(temp_input_file, "w") as f:
        json.dump(input_data, f)


    # Rescale
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    if hasattr(model_write, "scale_base") and hasattr(model_write, "scale_exponent"):
        for key in input_data:
            input_data[key] = torch.div(torch.as_tensor(input_data[key]), model_write.scale_base**model_write.scale_exponent).tolist()
    else:
        raise NotImplementedError("Model does not have scale_base attribute")
    assert contains_float(input_data), f"This is a testing error, not a model error. Please fix this test to properly turn data to float."
    
    with open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Rename

    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    new_input_data = {}
    count = 0
    for key in input_data.keys():
        if key == "input":
            new_input_data[f"input_TESTESTTEST_{count}"] = input_data[key]
            count +=1
        else:
            new_input_data[key] = input_data[key]
    assert "input" not in new_input_data.keys(), f"This is a testing error, not a model error. Please fix this test to not incldue 'input' as a key in the input data."

    with open(temp_input_file, "w") as f:
        json.dump(new_input_data, f)
   
    # Read outputs
    with open(temp_output_file, "r") as f:
        written_output_data = f.read()
    os.remove(temp_output_file)
    


    # Step 2: Read from that same input file (write_json=False) that has been rescaled and flattened
    model_read = model_fixture["model"]
    model_read.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=False,
        quantized_path = str(model_fixture["quantized_model"]),

    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert os.path.exists(temp_witness_file), "Witness file not generated"
    assert stdout.count("Running cargo command:") == 2, f"Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert stdout.count(output) == 2, f"Expected '{output}' in stdout, but it was not found."


    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

    assert os.path.exists(temp_output_file), "Output file not generated"
    with open(temp_output_file, "r") as f:
        new_output_file = f.read()

    # print(new_output_file, "TEST", written_output_data)
    
    assert_very_close(json.loads(new_output_file), json.loads(written_output_data), model_write)

    # assert new_output_file == written_output_data, "Output file content does not match the expected output"

@pytest.mark.e2e
def test_witness_wrong_name(
    model_fixture,
    capsys,
    temp_witness_file,
    temp_input_file,
    temp_output_file,
    check_model_compiles,
    check_witness_generated
):
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=True,
        quantized_path = str(model_fixture["quantized_model"]),

    )
    if os.path.exists(temp_witness_file):
        os.remove(temp_witness_file)
    assert not os.path.exists(temp_witness_file)
    def contains_float(obj):
            if isinstance(obj, float):
                return True
            elif isinstance(obj, dict):
                return any(contains_float(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(contains_float(i) for i in obj)
            return False
    # Rescale
    with open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    count = 0
    new_input_data = {}
    for key in input_data.keys():
        if key == "input":
            new_input_data[f"output"] = input_data[key]
            count +=1
        else:
            new_input_data[key] = input_data[key]
    assert "input" not in new_input_data.keys(), f"This is a testing error, not a model error. Please fix this test to not incldue 'input' as a key in the input data."
    assert ("output" in new_input_data.keys() or count ==0), f"This is a testing error, not a model error. Please fix this test to incldue 'output' as a key in the input data."

   
    with open(temp_output_file, "r") as f:
        written_output_data = f.read()
    os.remove(temp_output_file)
    
    with open(temp_input_file, "w") as f:
        json.dump(new_input_data, f)

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=False,
        witness_file=temp_witness_file,
        circuit_path=str(model_fixture["circuit_path"]),
        input_file=temp_input_file,
        output_file=temp_output_file,
        write_json=False,
        quantized_path = str(model_fixture["quantized_model"]),
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert os.path.exists(temp_witness_file), "Witness file not generated"
    assert stdout.count("Running cargo command:") == 2, f"Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert stdout.count(output) == 2, f"Expected '{output}' in stdout, but it was not found."


    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert output not in stdout, f"Did not expect '{output}' in stdout, but it was found."

    assert os.path.exists(temp_output_file), "Output file not generated"
    with open(temp_output_file, "r") as f:
        new_output_file = f.read()
    assert_very_close(json.loads(new_output_file), json.loads(written_output_data), model_write)

    assert new_output_file == written_output_data, "Output file content does not match the expected output"

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


import pytest
import torch

import python.tests.circuit_e2e_tests.helper_fns_for_tests  # noqa: F401
from python.core.circuits.errors import CircuitRunError

# Assume these are your models
# Enums, utils
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.tests.circuit_e2e_tests.helper_fns_for_tests import (
    BAD_OUTPUT,
    GOOD_OUTPUT,
    NestedArray,
    add_1_to_first_element,
    assert_very_close,
    check_model_compiles,  # noqa: F401
    check_witness_generated,  # noqa: F401
    circuit_compile_results,
    contains_float,
    model_fixture,  # noqa: F401
    temp_input_file,  # noqa: F401
    temp_output_file,  # noqa: F401
    temp_proof_file,  # noqa: F401
    temp_witness_file,  # noqa: F401
    witness_generated_results,
)

OUTPUTTWICE = 2
OUTPUTTHREETIMES = 3


@pytest.mark.e2e
def test_circuit_compiles(model_fixture: dict[str, Any]) -> None:
    # Here you could just check that circuit file exists
    circuit_compile_results[model_fixture["model"]] = False
    assert Path.exists(model_fixture["circuit_path"])
    circuit_compile_results[model_fixture["model"]] = True


@pytest.mark.e2e
def test_witness_dev(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
) -> None:
    _ = check_model_compiles

    model = model_fixture["model"]
    witness_generated_results[model_fixture["model"]] = False
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file)
    assert "Running cargo command:" in stdout
    for output in GOOD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."

    witness_generated_results[model_fixture["model"]] = True


@pytest.mark.e2e
def test_witness_wrong_outputs_dev(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    monkeypatch: Generator[pytest.MonkeyPatch, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
    caplog: Generator[pytest.LogCaptureFixture, None, None],
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    model = model_fixture["model"]
    original_get_outputs = model.get_outputs

    def patched_get_outputs(*args: tuple, **kwargs: dict[str, Any]) -> NestedArray:
        result = original_get_outputs(*args, **kwargs)
        return add_1_to_first_element(result)

    monkeypatch.setattr(model, "get_outputs", patched_get_outputs)
    with pytest.raises(CircuitRunError):
        model.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.GEN_WITNESS,
                dev_mode=False,
                witness_file=temp_witness_file,
                circuit_path=str(model_fixture["circuit_path"]),
                input_file=temp_input_file,
                output_file=temp_output_file,
                write_json=True,
                quantized_path=str(model_fixture["quantized_model"]),
            ),
        )
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)
    # assert False

    assert stderr == ""

    assert not Path.exists(temp_witness_file)
    assert "Running cargo command:" in stdout
    for output in GOOD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."
    for output in BAD_OUTPUT:
        assert (
            output in caplog.text
        ), f"Expected '{output}' in stdout, but it was not found."


@pytest.mark.e2e
def test_witness_prove_verify_true_inputs_dev(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    temp_proof_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    model = model_fixture["model"]
    print(model)
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.PROVE_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            proof_file=temp_proof_file,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_VERIFY,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            proof_file=temp_proof_file,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    # ASSERTIONS TODO

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file), "Witness file not generated"

    # Unexpected output
    assert stdout.count("poly.num_vars() == *params") == 0, (
        "'poly.num_vars() == *params' thrown. May need a dummy variable(s) "
        "to get rid of error. Dummy variables should be private variables. "
        "Can set = 1 in read_inputs and assert == 1 at end of circuit"
    )
    assert stdout.count("Proof generation failed") == 0, "Proof generation failed"
    assert Path.exists(temp_proof_file), "Proof file not generated"

    assert stdout.count("Verification generation failed") == 0, "Verification failed"
    # Expected output
    assert stdout.count("Running cargo command:") == OUTPUTTHREETIMES, (
        "Expected 'Running cargo command: ' in stdout three times, "
        "but it was not found."
    )
    assert (
        stdout.count("Witness Generated") == 1
    ), "Expected 'Witness Generated' in stdout three times, but it was not found."

    assert (
        stdout.count("proving") == 1
    ), "Expected 'proving' in stdout three times, but it was not found."
    assert (
        stdout.count("Proved") == 1
    ), "Expected 'Proved' in stdout three times, but it was not found."

    assert (
        stdout.count("Verified") == 1
    ), "Expected 'Verified' in stdout three times, but it was not found."


@pytest.mark.e2e
def test_witness_prove_verify_true_inputs_dev_expander_call(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    temp_proof_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    model = model_fixture["model"]
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.PROVE_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            proof_file=temp_proof_file,
            ecc=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_VERIFY,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            proof_file=temp_proof_file,
            ecc=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    # ASSERTIONS TODO

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)
    print(stderr)

    assert stderr == ""
    # assert False
    assert Path.exists(temp_witness_file), "Witness file not generated"

    # Unexpected output
    assert stdout.count("poly.num_vars() == *params") == 0, (
        "'poly.num_vars() == *params' thrown. May need a dummy variable(s) "
        "to get rid of error. Dummy variables should be private variables. "
        "Can set = 1 in read_inputs and assert == 1 at end of circuit"
    )
    assert stdout.count("Proof generation failed") == 0, "Proof generation failed"
    assert Path.exists(temp_proof_file), "Proof file not generated"

    assert stdout.count("Verification generation failed") == 0, "Verification failed"
    # Expected output
    assert (
        stdout.count("Running cargo command:") == 1
    ), "Expected 'Running cargo command: ' in stdout once, but it was not found."
    assert (
        stdout.count("Witness Generated") == 1
    ), "Expected 'Witness Generated' in stdout three times, but it was not found."

    assert stdout.count("proving") == 1, "Expected 'proving' but it was not found."

    assert (
        stdout.count("verifying proof") == 1
    ), "Expected 'verifying proof' but it was not found."
    assert stdout.count("success") == 1, "Expected 'success'  but it was not found."

    assert (
        stdout.count("expander-exec verify succeeded") == 1
    ), "Expected 'expander-exec verify succeeded' but it was not found."
    assert (
        stdout.count("expander-exec prove succeeded") == 1
    ), "Expected 'expander-exec prove succeeded' but it was not found."

    assert stdout.count("proving") == 1, "Expected 'proving' but it was not found."


@pytest.mark.e2e
def test_witness_read_after_write_json(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    if Path.exists(temp_witness_file):
        Path.unlink(temp_witness_file)
    assert not Path.exists(temp_witness_file)

    # Optional: Load the written input for inspection
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file), "Witness file not generated"
    assert "Running cargo command:" in stdout

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."

    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."

    # Optional: verify that input file content was actually read
    with Path.open(temp_input_file, "r") as f:
        read_input_data = f.read()

    assert (
        read_input_data == written_input_data
    ), "Input JSON read is not identical to what was written"


@pytest.mark.e2e
def test_witness_fresh_compile_dev(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    model = model_fixture["model"]
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=True,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file)
    assert "Running cargo command:" in stdout
    for output in GOOD_OUTPUT:
        assert output in stdout, f"Expected '{output}' in stdout, but it was not found."
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."


# Use once fixed input shape read in rust
@pytest.mark.e2e
def test_witness_incorrect_input_shape(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    assert Path.exists(temp_witness_file)
    Path.unlink(temp_witness_file)
    assert not Path.exists(temp_witness_file)

    # Optional: Load the written input for inspection
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    for key in input_data:
        if isinstance(input_data[key], list):
            input_data[key] = torch.as_tensor(input_data[key]).flatten().tolist()
        assert torch.as_tensor(input_data[key]).dim() <= 1, (
            f"Input data for {key} is not 1D tensor. "
            "This is a testing error, not a model error."
            "Please fix this test to properly flatten."
        )
    with Path.open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file), "Witness file not generated"
    assert (
        stdout.count("Running cargo command:") == OUTPUTTWICE
    ), "Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert (
            stdout.count(output) == OUTPUTTWICE
        ), f"Expected '{output}' in stdout, but it was not found."

    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."


@pytest.mark.e2e
def test_witness_unscaled(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    if Path.exists(temp_witness_file):
        Path.unlink(temp_witness_file)
    assert not Path.exists(temp_witness_file)

    # Rescale
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    if hasattr(model_write, "scale_base") and hasattr(model_write, "scale_exponent"):
        for key in input_data:
            print(input_data[key])
            input_data[key] = torch.div(
                torch.as_tensor(input_data[key]),
                model_write.scale_base**model_write.scale_exponent,
            ).tolist()
            print(input_data[key])
    else:
        msg = "Model does not have scale_base attribute"
        raise NotImplementedError(msg)
    assert contains_float(
        input_data,
    ), (
        "This is a testing error, not a model error. "
        "Please fix this test to properly turn data to float."
    )

    with Path.open(temp_output_file, "r") as f:
        written_output_data = f.read()
    Path.unlink(temp_output_file)

    with Path.open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file), "Witness file not generated"
    assert (
        stdout.count("Running cargo command:") == OUTPUTTWICE
    ), "Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert (
            stdout.count(output) == OUTPUTTWICE
        ), f"Expected '{output}' in stdout, but it was not found."

    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."

    assert Path.exists(temp_output_file), "Output file not generated"
    with Path.open(temp_output_file, "r") as f:
        new_output_file = f.read()
    assert_very_close(
        json.loads(new_output_file),
        json.loads(written_output_data),
        model_write,
    )


@pytest.mark.e2e
def test_witness_unscaled_and_incorrect_shape_input(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    if Path.exists(temp_witness_file):
        Path.unlink(temp_witness_file)
    assert not Path.exists(temp_witness_file)

    # flatten shape
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    for key in input_data:
        if isinstance(input_data[key], list):
            input_data[key] = torch.as_tensor(input_data[key]).flatten().tolist()
        assert torch.as_tensor(input_data[key]).dim() <= 1, (
            f"Input data for {key} is not 1D tensor. "
            "This is a testing error, not a model error."
            "Please fix this test to properly flatten."
        )
    with Path.open(temp_input_file, "w") as f:
        json.dump(input_data, f)
    # Rescale
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    if hasattr(model_write, "scale_base") and hasattr(model_write, "scale_exponent"):
        for key in input_data:
            input_data[key] = torch.div(
                torch.as_tensor(input_data[key]),
                model_write.scale_base**model_write.scale_exponent,
            ).tolist()
    else:
        msg = "Model does not have scale_base attribute"
        raise NotImplementedError(msg)
    assert contains_float(
        input_data,
    ), (
        "This is a testing error, not a model error. "
        "Please fix this test to properly turn data to float."
    )

    with Path.open(temp_output_file, "r") as f:
        written_output_data = f.read()
    Path.unlink(temp_output_file)

    with Path.open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Step 2: Read from that same input file (write_json=False)
    # that has been rescaled and flattened
    model_read = model_fixture["model"]
    model_read.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file), "Witness file not generated"
    assert (
        stdout.count("Running cargo command:") == OUTPUTTWICE
    ), "Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert (
            stdout.count(output) == OUTPUTTWICE
        ), f"Expected '{output}' in stdout, but it was not found."

    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."

    assert Path.exists(temp_output_file), "Output file not generated"
    with Path.open(temp_output_file, "r") as f:
        new_output_file = f.read()
    assert_very_close(
        json.loads(new_output_file),
        json.loads(written_output_data),
        model_write,
    )


@pytest.mark.e2e
def test_witness_unscaled_and_incorrect_and_bad_named_input(  # noqa: PLR0915
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles

    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    if Path.exists(temp_witness_file):
        Path.unlink(temp_witness_file)
    assert not Path.exists(temp_witness_file)

    # flatten shape
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    for key in input_data:
        if isinstance(input_data[key], list):
            input_data[key] = torch.as_tensor(input_data[key]).flatten().tolist()
        assert torch.as_tensor(input_data[key]).dim() <= 1, (
            f"Input data for {key} is not 1D tensor. "
            "This is a testing error, not a model error. "
            "Please fix this test to properly flatten."
        )
    with Path.open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Rescale
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    if hasattr(model_write, "scale_base") and hasattr(model_write, "scale_exponent"):
        for key in input_data:
            input_data[key] = torch.div(
                torch.as_tensor(input_data[key]),
                model_write.scale_base**model_write.scale_exponent,
            ).tolist()
    else:
        msg = "Model does not have scale_base attribute"
        raise NotImplementedError(msg)
    assert contains_float(
        input_data,
    ), (
        "This is a testing error, not a model error. "
        "Please fix this test to properly turn data to float."
    )

    with Path.open(temp_input_file, "w") as f:
        json.dump(input_data, f)

    # Rename

    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    new_input_data = {}
    count = 0
    for key in input_data:
        if key == "input":
            new_input_data[f"input_TESTESTTEST_{count}"] = input_data[key]
            count += 1
        else:
            new_input_data[key] = input_data[key]
    assert "input" not in new_input_data, (
        "This is a testing error, not a model error. "
        "Please fix this test to not include 'input' as a key in the input data."
    )

    with Path.open(temp_input_file, "w") as f:
        json.dump(new_input_data, f)

    # Read outputs
    with Path.open(temp_output_file, "r") as f:
        written_output_data = f.read()
    Path.unlink(temp_output_file)

    # Step 2: Read from that same input file (write_json=False)
    # that has been rescaled and flattened
    model_read = model_fixture["model"]
    model_read.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file), "Witness file not generated"
    assert (
        stdout.count("Running cargo command:") == OUTPUTTWICE
    ), "Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert (
            stdout.count(output) == OUTPUTTWICE
        ), f"Expected '{output}' in stdout, but it was not found."

    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."

    assert Path.exists(temp_output_file), "Output file not generated"
    with Path.open(temp_output_file, "r") as f:
        new_output_file = f.read()

    assert_very_close(
        json.loads(new_output_file),
        json.loads(written_output_data),
        model_write,
    )


@pytest.mark.e2e
def test_witness_wrong_name(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    _ = check_witness_generated
    _ = check_model_compiles
    # Step 1: Write the input file via write_json=True
    model_write = model_fixture["model"]
    model_write.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )
    if Path.exists(temp_witness_file):
        Path.unlink(temp_witness_file)
    assert not Path.exists(temp_witness_file)

    # Rescale
    with Path.open(temp_input_file, "r") as f:
        written_input_data = f.read()
    input_data = json.loads(written_input_data)
    count = 0
    new_input_data = {}
    for key in input_data:
        if key == "input":
            new_input_data["output"] = input_data[key]
            count += 1
        else:
            new_input_data[key] = input_data[key]
    assert "input" not in new_input_data, (
        "This is a testing error, not a model error. "
        "Please fix this test to not include 'input' as a key in the input data."
    )
    assert "output" in new_input_data or count == 0, (
        "This is a testing error, not a model error. "
        "Please fix this test to include 'output' as a key in the input data."
    )

    with Path.open(temp_output_file, "r") as f:
        written_output_data = f.read()
    Path.unlink(temp_output_file)

    with Path.open(temp_input_file, "w") as f:
        json.dump(new_input_data, f)

    # Step 2: Read from that same input file (write_json=False)
    model_read = model_fixture["model"]
    model_read.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=False,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    # Step 3: Validate expected outputs and no errors
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    print(stdout)

    assert stderr == ""

    assert Path.exists(temp_witness_file), "Witness file not generated"
    assert (
        stdout.count("Running cargo command:") == OUTPUTTWICE
    ), "Expected 'Running cargo command: ' in stdout twice, but it was not found."

    # Check good output appeared
    for output in GOOD_OUTPUT:
        assert (
            stdout.count(output) == OUTPUTTWICE
        ), f"Expected '{output}' in stdout, but it was not found."

    # Ensure no unexpected errors
    for output in BAD_OUTPUT:
        assert (
            output not in stdout
        ), f"Did not expect '{output}' in stdout, but it was found."

    assert Path.exists(temp_output_file), "Output file not generated"
    with Path.open(temp_output_file, "r") as f:
        new_output_file = f.read()
    assert_very_close(
        json.loads(new_output_file),
        json.loads(written_output_data),
        model_write,
    )

    assert (
        new_output_file == written_output_data
    ), "Output file content does not match the expected output"


def add_to_first_scalar(data: list, delta: float = 0.1) -> bool:
    """
    Traverse nested lists until the first scalar (non-list) element is found,
    then add `delta` to it. Returns True if modified, False otherwise.
    """
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            return add_to_first_scalar(data[0], delta)
        data[0] = data[0] + delta
        return True
    return False


@pytest.mark.e2e
def test_witness_prove_verify_false_inputs_dev(
    model_fixture: dict[str, Any],
    capsys: Generator[pytest.CaptureFixture[str], None, None],
    temp_witness_file: Generator[Path, None, None],
    temp_input_file: Generator[Path, None, None],
    temp_output_file: Generator[Path, None, None],
    temp_proof_file: Generator[Path, None, None],
    check_model_compiles: None,
    check_witness_generated: None,
) -> None:
    """
    Same as test_witness_prove_verify_true_inputs_dev, but deliberately
    corrupts the witness outputs to trigger verification failure.
    """
    _ = check_witness_generated
    _ = check_model_compiles

    model = model_fixture["model"]

    # Step 1: Generate witness
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            write_json=True,
            quantized_path=str(model_fixture["quantized_model"]),
        ),
    )

    # Step 2: Corrupt the witness file by flipping some bytes
    with Path(temp_input_file).open(encoding="utf-8") as f:
        input_data = json.load(f)

    first_key = next(iter(input_data))  # get the first key
    modified = add_to_first_scalar(input_data[first_key], 0.1)

    if not modified:
        pytest.skip("Input file format not suitable for tampering test.")

    tampered_input_file = temp_input_file.parent / "tampered_input.json"
    with Path(tampered_input_file).open("w", encoding="utf-8") as f:
        json.dump(input_data, f)

    # Step 3: Try to prove with corrupted witness
    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.PROVE_WITNESS,
            dev_mode=False,
            witness_file=temp_witness_file,
            circuit_path=str(model_fixture["circuit_path"]),
            input_file=temp_input_file,
            output_file=temp_output_file,
            proof_file=temp_proof_file,
            quantized_path=str(model_fixture["quantized_model"]),
            ecc=False,
        ),
    )

    # Step 4: Attempt verification
    with pytest.raises(CircuitRunError) as excinfo:
        model.base_testing(
            CircuitExecutionConfig(
                run_type=RunType.GEN_VERIFY,
                dev_mode=False,
                witness_file=temp_witness_file,
                circuit_path=str(model_fixture["circuit_path"]),
                input_file=tampered_input_file,
                output_file=temp_output_file,
                proof_file=temp_proof_file,
                quantized_path=str(model_fixture["quantized_model"]),
                ecc=False,
            ),
        )

    # ---- ASSERTIONS ----
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    print(stdout)
    print(stderr)

    print(excinfo.value)
    assert "Witness does not match provided inputs and outputs" in str(
        excinfo.value,
    )
    assert "'Verify' failed" in str(
        excinfo.value,
    )

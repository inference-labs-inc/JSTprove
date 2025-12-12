from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from python.core.circuits.zk_model_base import ZKModelBase
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType

GOOD_OUTPUT = ["Witness Generated"]
BAD_OUTPUT = [
    "Witness generation failed",
    "Outputs generated do not match outputs supplied",
]

NUMPARAMS3 = 3
NUMPARAMS4 = 4


@pytest.fixture(scope="module")
def model_fixture(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    param = request.param
    name = f"{param.name}"
    model_class = param.loader
    args, kwargs = (), {}

    if len(param) == NUMPARAMS3:
        if isinstance(param[2], dict):
            kwargs = param[2]
        else:
            args = param[2]
    elif len(param) == NUMPARAMS4:
        args, kwargs = param[2], param[3]

    temp_dir = tmp_path_factory.mktemp(name)
    circuit_path = temp_dir / f"{name}_circuit.txt"
    quantized_path = temp_dir / f"{name}_quantized.pt"

    model = model_class(*args, **kwargs)

    model.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.COMPILE_CIRCUIT,
            dev_mode=True,
            circuit_path=str(circuit_path),
            quantized_path=quantized_path,
        ),
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
def temp_witness_file(tmp_path: str) -> Generator[Path, None, None]:
    witness_path = tmp_path / "temp_witness.txt"
    # Give it to the test
    yield witness_path

    # After the test is done, remove it
    if Path.exists(witness_path):
        witness_path.unlink()


@pytest.fixture
def temp_input_file(tmp_path: str) -> Generator[Path, None, None]:
    input_path = tmp_path / "temp_input.txt"
    # Give it to the test
    yield input_path

    # After the test is done, remove it
    if Path.exists(input_path):
        input_path.unlink()


@pytest.fixture
def temp_output_file(tmp_path: str) -> Generator[Path, None, None]:
    output_path = tmp_path / "temp_output.txt"
    # Give it to the test
    yield output_path

    # After the test is done, remove it
    if Path.exists(output_path):
        output_path.unlink()


@pytest.fixture
def temp_proof_file(tmp_path: str) -> Generator[Path, None, None]:
    output_path = tmp_path / "temp_proof.txt"
    # Give it to the test
    yield output_path

    # After the test is done, remove it
    if Path.exists(output_path):
        output_path.unlink()


ScalarOrTensor: TypeAlias = int | float | torch.Tensor
NestedArray: TypeAlias = (
    ScalarOrTensor | list["NestedArray"] | tuple["NestedArray"] | np.ndarray
)


def add_1_to_first_element(x: NestedArray) -> NestedArray:
    """Safely adds 1 to the first element of any scalar/list/tensor."""
    if isinstance(x, (int, float)):
        return x + 1
    if isinstance(x, torch.Tensor):
        x = x.clone()  # avoid in-place modification
        x.view(-1)[0] += 1
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        x = list(x)
        x[0] = add_1_to_first_element(x[0])
        return x
    msg = f"Unsupported type for get_outputs patch: {type(x)}"
    raise TypeError(msg)


# Define models to be tested
circuit_compile_results = {}
witness_generated_results = {}

Nested: TypeAlias = float | Mapping[str, "Nested"] | Sequence["Nested"]


def contains_float(obj: Nested) -> bool:
    if isinstance(obj, float):
        return True
    if isinstance(obj, dict):
        return any(contains_float(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_float(i) for i in obj)
    return False


@pytest.fixture(scope="module")
def check_model_compiles(model_fixture: dict[str, Any]) -> None:
    # Default to True; will be set to False if first test fails
    result = circuit_compile_results.get(model_fixture["model"])
    if result is False:
        pytest.skip(
            f"Skipping because the first test failed for: {model_fixture['model']}",
        )
    return result


@pytest.fixture(scope="module")
def check_witness_generated(model_fixture: dict[str, Any]) -> None:
    # Default to True; will be set to False if first test fails
    result = witness_generated_results.get(model_fixture["model"])
    if result is False:
        pytest.skip(
            f"Skipping because the first test failed for: {model_fixture['model']}",
        )
    return result


def assert_very_close(
    inputs_1: np.array,
    inputs_2: np.array,
    model: ZKModelBase,
) -> None:
    for i in inputs_1:
        x = torch.div(
            torch.as_tensor(inputs_1[i]),
            model.scale_base**model.scale_exponent,
        )
        y = torch.div(
            torch.as_tensor(inputs_2[i]),
            model.scale_base**model.scale_exponent,
        )

        assert torch.isclose(x, y, rtol=1e-8).all()

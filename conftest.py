from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.config import Config, Parser
    from _pytest.nodes import Item
    from _pytest.python import Metafunc
import pytest

from python.core.utils.model_registry import get_models_to_test, list_available_models


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--model",
        action="append",
        default=None,
        help="Model(s) to test. Use multiple times to test more than one.",
    )
    parser.addoption(
        "--list-models",
        action="store_true",
        default=False,
        help="List all available circuit models.",
    )
    parser.addoption(
        "--source",
        action="store",
        choices=["class", "pytorch", "onnx"],
        default=None,
        help="Restrict models to a specific source: class, pytorch, or onnx.",
    )
    parser.addoption(
        "--unit",
        action="store_true",
        default=False,
        help="Run only unit tests.",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run only integration tests.",
    )
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run only end-to-end tests.",
    )
    parser.addoption(
        "--onnx-opset-versions",
        action="store",
        default=None,
        help=(
            "Comma-separated ONNX opset versions to test. "
            "When specified, valid and e2e tests will be duplicated and run "
            "for each version (e.g., --onnx-opset-versions 14,15,16,19)"
        ),
    )
    parser.addoption(
        "--all-opset-versions",
        action="store_true",
        default=False,
        help=("All opset versions 7-23 will be tested when this is specified"),
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    run_unit = config.getoption("--unit")
    run_integration = config.getoption("--integration")
    run_e2e = config.getoption("--e2e")

    # If no filters set, run all
    if not any([run_unit, run_integration, run_e2e]):
        return

    selected = []
    deselected = []

    for item in items:
        has_unit = "unit" in item.keywords
        has_integration = "integration" in item.keywords and "e2e" not in item.keywords
        has_e2e = "e2e" in item.keywords

        if (
            (run_unit and has_unit)
            or (run_integration and has_integration)
            or (run_e2e and has_e2e)
        ):
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_generate_tests(metafunc: Metafunc) -> None:
    if "model_fixture" in metafunc.fixturenames:
        selected_models = metafunc.config.getoption("model")
        selected_source = metafunc.config.getoption("source")

        models = get_models_to_test(selected_models, selected_source)
        ids = [
            f"{model.name}:{model.source}" for model in models
        ]  # Extract readable names

        metafunc.parametrize(
            "model_fixture",
            models,
            indirect=True,
            scope="module",
            ids=ids,
        )


def pytest_configure(config: Config) -> None:
    # If the --list-models option is used, list models and exit
    if config.getoption("list_models"):
        available_models = list_available_models()
        print("\nAvailable Circuit Models:")  # noqa: T201
        for model in available_models:
            print(f"- {model}")  # noqa: T201
        pytest.exit(
            "Exiting after listing available models.",
        )  # This prevents tests from running

    from python.tests.onnx_quantizer_tests.layers.factory import (  # noqa: PLC0415
        set_onnx_opset_versions,
    )

    onnx_versions_str = config.getoption("onnx_opset_versions")
    if onnx_versions_str:
        try:
            onnx_versions = [int(v.strip()) for v in onnx_versions_str.split(",")]
            set_onnx_opset_versions(onnx_versions)
            config.onnx_opset_versions = onnx_versions
            print(f"\nONNX opset versions for testing: {onnx_versions}")  # noqa: T201
        except ValueError:
            pytest.exit(
                f"Invalid ONNX opset versions: {onnx_versions_str}. "
                "Expected comma-separated integers (e.g., 14,15,16,19)",
            )
    else:
        set_onnx_opset_versions(None)
        config.onnx_opset_versions = None

    all_onnx_versions = config.getoption("all_opset_versions")
    if all_onnx_versions:
        set_onnx_opset_versions(list(range(7, 23)))


@pytest.fixture(scope="session", autouse=True)
def ensure_dev_mode_compile_for_e2e(
    request: pytest.FixtureRequest,
) -> None:
    """
    Ensure that rust code is recompiled before e2e tests are performed.
    """
    # Only run this for e2e tests
    if not request.config.getoption("--e2e"):
        return

    # Skip if there are no e2e tests being run
    if not any("e2e" in item.keywords for item in request.session.items):
        return

    import subprocess  # noqa: PLC0415

    result = subprocess.run(
        ["cargo", "build", "--release"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )

    print("stdout:", result.stdout)  # noqa: T201
    print("stderr:", result.stderr)  # noqa: T201

    # On initial tests this approach works. If this breaks, we can run
    # compilation of a basic circuit with dev_mode = True

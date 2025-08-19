import pytest
<<<<<<< HEAD
from python.core.utils.model_registry import get_models_to_test, list_available_models
=======
from python.testing.core.utils.model_registry import (
    get_models_to_test,
    list_available_models,
)

>>>>>>> main

def pytest_addoption(parser):
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
        "--unit", action="store_true", default=False, help="Run only unit tests."
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run only integration tests.",
    )
    parser.addoption(
        "--e2e", action="store_true", default=False, help="Run only end-to-end tests."
    )


def pytest_collection_modifyitems(config, items):
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

        if run_unit and has_unit:
            selected.append(item)
        elif run_integration and has_integration:
            selected.append(item)
        elif run_e2e and has_e2e:
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_generate_tests(metafunc):
    if "model_fixture" in metafunc.fixturenames:
        selected_models = metafunc.config.getoption("model")
        selected_source = metafunc.config.getoption("source")

        models = get_models_to_test(selected_models, selected_source)
        ids = [
            f"{model.name}:{model.source}" for model in models
        ]  # Extract readable names

        metafunc.parametrize(
            "model_fixture", models, indirect=True, scope="module", ids=ids
        )


def pytest_configure(config):
    # If the --list-models option is used, list models and exit
    if config.getoption("list_models"):
        available_models = list_available_models()
        print("\nAvailable Circuit Models:")
        for model in available_models:
            print(f"- {model}")
        pytest.exit(
            "Exiting after listing available models."
        )  # This prevents tests from running

import os
import subprocess
import time
import pytest 
from python_testing.utils.model_registry import get_models_to_test, list_available_models

def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="append",
        default=None,
        help="Model(s) to test. Use multiple times to test more than one."
    )
    parser.addoption(
        "--list-models",
        action="store_true",
        default=False,
        help="List all available circuit models."
    )
    parser.addoption(
        "--source",
        action="store",
        choices=["class", "pytorch", "onnx"],
        default=None,
        help="Restrict models to a specific source: class, pytorch, or onnx."
    )


def pytest_generate_tests(metafunc):
    if "model_fixture" in metafunc.fixturenames:
        selected_models = metafunc.config.getoption("model")
        selected_source = metafunc.config.getoption("source")

        models = get_models_to_test(selected_models, selected_source)
        ids = [f"{model.name}:{model.source}" for model in models]  # Extract readable names

        metafunc.parametrize("model_fixture", models, indirect=True, scope="module", ids=ids)

def pytest_configure(config):
    # If the --list-models option is used, list models and exit
    if config.getoption("list_models"):
        available_models = list_available_models()
        print("\nAvailable Circuit Models:")
        for model in available_models:
            print(f"- {model}")
        pytest.exit("Exiting after listing available models.")  # This prevents tests from running
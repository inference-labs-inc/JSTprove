from collections import namedtuple
import importlib
import os
import pkgutil
import types
from typing import Callable, List

from python.core import circuit_models
from python.core import circuits

from python.core.circuits.base import Circuit
from python.core.circuit_models.generic_onnx import GenericModelONNX


ModelEntry = namedtuple("ModelEntry", ["name", "source", "loader", "args", "kwargs"])


def scan_model_files(directory: str, extension: str, loader_fn: Callable, prefix: str) -> List[ModelEntry]:
    """Scan a directory for model files and return each discovered model in a callable.

    Args:
        directory (str): Path to the directory to scan.
        extension (str): File extension to filter by (e.g., ".onnx").
        loader_fn (Callable): Loader function that can instantiate a model from a file path.
        prefix (str): Source prefix used to categorize models (e.g., "onnx").

    Returns:
        List[ModelEntry]: A list of ModelEntry objects representing discovered models.
    """
    entries = []
    for file_or_foldername in os.listdir(directory):
        if prefix == "onnx":
            if os.path.isfile(os.path.join(directory, file_or_foldername)) and file_or_foldername[-5:] == ".onnx":
                name = file_or_foldername[0:len(file_or_foldername) - 5]
                path = os.path.join(directory, file_or_foldername)
                entries.append(
                    ModelEntry(name=f"{name}", source=prefix, loader=lambda p=path: loader_fn(p), args=(), kwargs={})
                )
    return entries

def import_all_submodules(package: types.ModuleType):
    """Import all submodules of a given package.

    Args:
        package (types.ModuleType): The Python package to import submodules from.
    """
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(name)

# Import all submodules so their classes are registered
import_all_submodules(circuit_models)
import_all_submodules(circuits)

def all_subclasses(cls):
    """Recursively find all subclasses of a given class.

    Args:
        cls (type): The base class to search subclasses for.

    Returns:
        _type_: A set of all subclasses of the given class.
    """
    subclasses = set(cls.__subclasses__())
    return subclasses.union(
        s for c in subclasses for s in all_subclasses(c)
    )


def build_models_to_test() -> List[ModelEntry]:
    """Build a list of model entries to be tested.

    - Collects all subclasses of the Circuit base class.
    - Filters out unwanted base or placeholder models.
    - Adds discovered ONNX models from the models directory.

    Returns:
        List[ModelEntry]: A list of model entries to be used in tests.
    """
    models = []
    for cls in all_subclasses(Circuit):
        name = cls.__name__.lower()
        models.append(
            ModelEntry(name=name, source="class", loader=cls, args=(), kwargs={})
        )
    # Filter unwanted class models
    models = [
        m for m in models
        if m.name not in {"zkmodel", "genericmodelonnx", "zktorchmodel", "zkmodelbase"}
    ]
    # Add ONNX models
    models += scan_model_files("python/models/models_onnx", ".onnx", GenericModelONNX, "onnx")
    return models


MODELS_TO_TEST = build_models_to_test()



def list_available_models() -> List[str]:
    """List all available models in a human-readable format.

    Returns:
        List[str]: A sorted list of strings in the form "source: model_name".
    """
    return sorted(f"{model.source}: {model.name}" for model in MODELS_TO_TEST)


def get_models_to_test(selected_models: List[str] = None, source_filter: str =None) -> List[ModelEntry]:
    """Retrieve models to be tested with optional filtering.

    Args:
        selected_models (List[str], optional): A list of model names to include. Defaults to None.
        source_filter (str, optional): Restrict models to a specific source (e.g., "onnx", "class"). Defaults to None.

    Returns:
        List[ModelEntry]: A filtered list of model entries.
    """
    models = MODELS_TO_TEST

    if selected_models is not None:
        models = [m for m in models if m.name in selected_models]

    if source_filter is not None:
        models = [m for m in models if m.source == source_filter]

    return models

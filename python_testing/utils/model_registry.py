import importlib
import pkgutil
import pytest 
import python_testing.circuit_models
import python_testing.circuit_components

from python_testing.circuit_models.demo_cnn import Demo
from python_testing.circuit_components.relu import ReLU, ConversionType
from python_testing.utils.helper_functions import RunType
from python_testing.circuit_components.circuit_helpers import Circuit


def import_all_submodules(package):
    """Recursively import all submodules of a given package."""
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(name)

# Import all submodules so their classes are registered
import_all_submodules(python_testing.circuit_models)
import_all_submodules(python_testing.circuit_components)

def all_subclasses(cls):
    """Recursively find all subclasses of a given class."""
    subclasses = set(cls.__subclasses__())
    return subclasses.union(
        s for c in subclasses for s in all_subclasses(c)
    )

def build_models_to_test():
    models = []
    for cls in all_subclasses(Circuit):
        name = cls.__name__.lower()
        models.append((name, cls))
    return models

MODELS_TO_TEST = build_models_to_test()
MODELS_TO_TEST = [t for t in MODELS_TO_TEST if (t[0] != "zkmodel" and t[0] != "doomslice" and t[0] != "slice")]
MODELS_TO_TEST = [t for t in MODELS_TO_TEST if (t[0] != "genericmodelforcircuit")]

MODELS_TO_TEST.append(("relu_dual", ReLU, {"conversion_type":ConversionType.TWOS_COMP}))

def modify_models_based_on_class(models_to_test):
    """Loop through the models and modify arguments/kwargs for specific classes."""
    updated_models = []
    
    for model_name, model_class in models_to_test:
        args, kwargs = (), {}

        # Check specific model class and add parameters if needed
        if model_class == Demo:
            args = (3, 4)  # Example args for SimpleCircuit
            kwargs = {"layers":["conv1", "relu", "reshape", "fc1", "relu", "fc2"]} # Example kwargs for SimpleCircuit
        
        updated_models.append((model_name, model_class, args, kwargs))
    
    return updated_models


def list_available_models():
    """List all available circuit model names."""
    # models = all_subclasses(Circuit)
    models = MODELS_TO_TEST

    available_models = [cls[0] for cls in models]
    return sorted(available_models)


def get_models_to_test(selected_models = None):
    if selected_models is None:
        return MODELS_TO_TEST
    return [m for m in MODELS_TO_TEST if m[0] in selected_models]
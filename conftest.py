import pytest 
from python_testing.circuit_components.extrema import Extrema
from python_testing.circuit_components.matmul import MatMul
from python_testing.circuit_components.matmul_bias import MatMulBias
from python_testing.circuit_components.maxpooling import MaxPooling2D
from python_testing.circuit_models.demo_cnn import Demo
from python_testing.circuit_models.doom_model import Doom
from python_testing.circuit_models.net_model import NetModel, NetConv1Model, NetConv2Model, NetFC1Model, NetFC2Model, NetFC3Model
from python_testing.circuit_models.simple_circuit import SimpleCircuit
from python_testing.circuit_models.doom_slices import DoomConv1, DoomConv2, DoomConv3, DoomFC1, DoomFC2
from python_testing.circuit_models.eth_fraud import Eth
from python_testing.circuit_components.convolution import Convolution, QuantizedConv, QuantizedConvRelu
from python_testing.circuit_components.relu import ReLU, ConversionType
from python_testing.circuit_components.matrix_multiplication import MatrixMultiplication, QuantizedMatrixMultiplication, QuantizedMatrixMultiplicationReLU
from python_testing.circuit_components.matrix_addition import MatrixAddition
from python_testing.circuit_components.scaled_matrix_product import ScaledMatrixProduct
from python_testing.circuit_components.scaled_matrix_product_sum import ScaledMatrixProductSum
from python_testing.utils.helper_functions import RunType
from python_testing.circuit_components.circuit_helpers import Circuit


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
# MODELS_TO_TEST = [
#     ("doom", Doom),
#     ("simple_circuit", SimpleCircuit),
#     ("doom_conv1", DoomConv1),
#     ("doom_conv2", DoomConv2),
#     ("doom_conv3", DoomConv3),
#     ("doom_fc1", DoomFC1),
#     ("doom_fc2", DoomFC2),
#     ("eth_fraud", Eth),
#     ("net", NetModel),
#     ("net_conv1", NetConv1Model),
#     ("net_conv2", NetConv2Model),
#     ("net_fc1", NetFC1Model),
#     ("net_fc2", NetFC2Model),
#     ("net_fc3", NetFC3Model),

#     ("conv", Convolution),
#     ("quantized_conv", QuantizedConv),
#     ("quantized_conv_relu", QuantizedConvRelu),
#     ("relu", ReLU, {"conversion_type":ConversionType.DUAL_MATRIX}),
#     ("relu", ReLU, {"conversion_type":ConversionType.TWOS_COMP}),


#     ("matrix_multiplication", MatrixMultiplication),
#     ("quantized_matrix_multiplication", QuantizedMatrixMultiplication),
#     ("quantized_matrix_multiplication_relu", QuantizedMatrixMultiplicationReLU),
#     ("matrix_addition", MatrixAddition),
#     ("scaled_matrix_product", ScaledMatrixProduct),
#     ("scaled_matrix_product_sum", ScaledMatrixProductSum),

#     ("cnn_demo", Demo, {"layers":["conv1", "relu", "reshape", "fc1", "relu", "fc2"]}),

#     ("extrema", Extrema),
#     ("matmul", MatMul),
#     ("matmul_bias", MatMulBias),
#     ("maxpooling", MaxPooling2D),
# ]
MODELS_TO_TEST = build_models_to_test()
MODELS_TO_TEST = [t for t in MODELS_TO_TEST if (t[0] != "zkmodel" and t[0] != "doomslice" and t[0] != "slice")]
print(MODELS_TO_TEST)
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

def list_available_models():
    """List all available circuit model names."""
    # models = all_subclasses(Circuit)
    models = MODELS_TO_TEST

    available_models = [cls[0] for cls in models]
    return sorted(available_models)


def get_models_to_test(config):
    # print(MODELS_TO_TEST)
    # raise
    selected_models = config.getoption("model")
    if selected_models is None:
        return MODELS_TO_TEST
    return [m for m in MODELS_TO_TEST if m[0] in selected_models]

def pytest_generate_tests(metafunc):
    if "model_fixture" in metafunc.fixturenames:
        models = get_models_to_test(metafunc.config)
        metafunc.parametrize("model_fixture", models, indirect=True, scope="module")

def pytest_configure(config):
    # If the --list-models option is used, list models and exit
    if config.getoption("list_models"):
        available_models = list_available_models()
        print("\nAvailable Circuit Models:")
        for model in available_models:
            print(f"- {model}")
        pytest.exit("Exiting after listing available models.")  # This prevents tests from running

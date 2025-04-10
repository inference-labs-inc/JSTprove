# from circom.reward_fn import generate_sample_inputs
import json
import os
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import RunType, get_files, to_json, prove_and_verify
from python_testing.utils.pytorch_helpers import PytorchConverter, QuantizedLinear, ZKModel
import torch.nn as nn
import torch.nn.functional as F


from enum import Enum
import sys
from python_testing.circuit_components.circuit_helpers import Circuit



class MatMultType(Enum):
    Traditional = "traditional"
    Naive = "naive"
    Naive_array = "naive_array"
    Naive1 = "naive1"
    Naive2 = "naive2"
    Naive3 = "naive3"

class MatrixMultiplicationModel(nn.Module):
    def __init__(self, in_channels, out_channels, bias = False):
        super(MatrixMultiplicationModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.fc1(x)
    
class MatrixMultiplicationReLUModel(nn.Module):
    def __init__(self, in_channels, out_channels, bias = False):
        super(MatrixMultiplicationReLUModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels, bias = False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return F.relu(self.fc1(x))

class MatrixMultiplication(ZKModel):

    def __init__(self, circuit_type: MatMultType = MatMultType.Naive2, file_name="model/matrix_multiplication.pth", rescale = False):
        self.required_keys = ["input"]
        self.name = "matrix_multiplication"
        self.input_data_file = "doom_data/doom_input.json"


        self.scaling = 21

        self.N_ROWS_A: int = 1; # m
        self.N_COLS_A: int = 1568; # n
        self.N_ROWS_B: int = self.N_COLS_A; # n
        self.N_COLS_B: int = 256; # k
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MatrixMultiplicationModel(self.N_COLS_A, self.N_COLS_B, bias = False).to(device)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        rescale_config = {"fc1": rescale}

        self.quantized_model = self.quantize_model(model, 2**self.scaling, rescale_config=rescale_config)
        self.quantized_model.eval()
        

        # self.input_shape = [self.N_ROWS_A, self.N_COLS_A]
        if not rescale:
            self.quantized = False
        else:
            self.quantized = True
        self.is_relu = False
        self.circuit_type = circuit_type

    @property
    def input_shape(self):
        return [self.N_ROWS_A, self.N_COLS_A]
    
    def format_inputs(self, inputs):
        return {"matrix_a": inputs.tolist()}
    
    def format_outputs(self, outputs):
        return {"matrix_product_ab": outputs.tolist()}
    
    def read_input(self, file_name = "doom_data/doom_input.json"):
        """Reads the inputs to each layer of the model from text files."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data["matrix_a"]


    def get_weights(self):
        weights =  super().get_weights()
        weights["quantized"] = self.quantized
        # weights["scaling"] = self.scaling
        weights["is_relu"] = self.is_relu
        weights['circuit_type'] =  self.circuit_type.value

        weights["matrix_b"] = weights.pop("fc_weights")[0]
        return weights
        

class QuantizedMatrixMultiplication(MatrixMultiplication):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, circuit_type: MatMultType = MatMultType.Naive2, file_name="model/matrix_multiplication.pth"):
        super().__init__(circuit_type, file_name, rescale = True)


class QuantizedMatrixMultiplicationReLU(MatrixMultiplication):
    def __init__(self, circuit_type: MatMultType = MatMultType.Naive2, file_name="model/matrix_multiplication_relu.pth", rescale = True):
        self.required_keys = ["input"]
        self.name = "matrix_multiplication"
        self.input_data_file = "doom_data/doom_input.json"


        self.scaling = 21

        self.N_ROWS_A: int = 1; # m
        self.N_COLS_A: int = 1568; # n
        self.N_ROWS_B: int = self.N_COLS_A; # n
        self.N_COLS_B: int = 256; # k
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MatrixMultiplicationReLUModel(self.N_COLS_A, self.N_COLS_B, bias = False).to(device)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        rescale_config = {"fc1": rescale}

        self.quantized_model = self.quantize_model(model, 2**self.scaling, rescale_config=rescale_config)
        self.quantized_model.eval()
        

        # self.input_shape = [self.N_ROWS_A, self.N_COLS_A]
        if not rescale:
            self.quantized = False
        else:
            self.quantized = True
        self.is_relu = True
        self.circuit_type = circuit_type
    
    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    # #Rework inputs to function

    d = MatrixMultiplication()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = MatrixMultiplication()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = MatrixMultiplication()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

    d = QuantizedMatrixMultiplication()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = QuantizedMatrixMultiplication()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = QuantizedMatrixMultiplication()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

    d = QuantizedMatrixMultiplicationReLU()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = QuantizedMatrixMultiplicationReLU()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = QuantizedMatrixMultiplicationReLU()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

    


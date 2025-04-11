import json
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import RunType, get_files, to_json, prove_and_verify
import os
from python_testing.circuit_components.relu import ReLU, ConversionType

from python_testing.circuit_components.convolution import Convolution, QuantizedConv
# from python_testing.matrix_multiplication import QuantizedMatrixMultiplication
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.utils.pytorch_helpers import ZKModel



class DoomAgent(nn.Module):
    def __init__(self, n_actions=7):
        super(DoomAgent, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.fc_input_dim = 32 * 7 * 7

        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(7)

        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            )
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()
        

class Doom(ZKModel):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom"
        self.input_data_file = "doom_data/doom_input.json"
        self.model_file_name = file_name


        self.scaling = 21
        self.input_shape = [1, 4, 28, 28]
        self.rescale_config = {"fc2": False}
        self.model_type = DoomAgent

    

if __name__ == "__main__":
    name = "doom"
    d = Doom()
    # d.base_testing()
    # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = Doom()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)
    # d.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
    # d.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")

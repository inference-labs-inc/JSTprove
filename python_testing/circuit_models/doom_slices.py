import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.circuit_components.convolution import Convolution, QuantizedConv
from python_testing.circuit_components.gemm import QuantizedGemm

from python_testing.utils.helper_functions import RunType, read_from_json, to_json
from python_testing.utils.pytorch_helpers import ZKModel
from python_testing.utils.pytorch_partial_models import ZKModel


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
        

class DoomConv1(ZKModel):
    def __init__(self, file_name="model/doom_checkpoint.pth", layer = "conv1"):
        self.required_keys = ["input"]
        self.name = "convolution"
        self.input_data_file = "doom_data/doom_input.json"


        self.scaling = 21
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DoomAgent().to(device)
        # checkpoint = torch.load(file_name, map_location=device)
        # model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        rescale_config = {"fc2": False}

        self.quantized_model = self.quantize_model(model, 2**self.scaling, rescale_config=rescale_config)
        self.quantized_model.eval()

        self.input_shape = [1, 4, 28, 28]
    


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

if __name__ == "__main__":
    # Doom().base_testing()
    names = [
        "conv1", 
        "conv2", 
        "conv3",
        "fc1",
        "fc2"
        ]
    weights = read_from_json("weights/doom_weights.json")
    weights2 = read_from_json("weights/doom_weights2.json")
    for name in names:
        print(name)
        inputs = read_from_json(f"doom_data/doom_{name}.json")
        if type(inputs) == list:
            inputs = inputs[0]
        if "conv" in name:
            d = QuantizedConv(relu = True)
            d.input_arr = torch.tensor(inputs["input"])
            d.weights = torch.tensor(weights[f"{name}_weights"])
            d.bias = torch.tensor(weights[f"{name}_bias"])
            d.strides = weights[f"{name}_strides"]
            d.kernel_shape = torch.tensor(weights[f"{name}_kernel_shape"])
            d.group = torch.tensor(weights[f"{name}_group"])
            d.dilation = weights[f"{name}_dilation"]
            d.pads = weights[f"{name}_pads"]
        elif "fc" in name:
            if name == "fc1":
                relu = True
                reshape = (1,32,7,7)
                d = QuantizedGemm(relu = relu, reshape=reshape)

                d.matrix_b = torch.tensor(weights[f"{name}_weights"])
                d.matrix_c = torch.tensor(weights[f"{name}_bias"])
                d.matrix_a = torch.tensor(inputs["input"]).reshape((-1,1568))

            else:
                relu = False
                reshape = []
                d = QuantizedGemm(relu = relu, reshape=reshape)
                d.matrix_b = torch.tensor(weights2[f"{name}_weights"])
                d.matrix_c = torch.tensor(weights2[f"{name}_bias"])
                d.matrix_a = torch.tensor(inputs["input"])

            d.alpha = torch.tensor(1)
            d.beta = torch.tensor(1)
            print(d.matrix_a.shape)
            print(d.matrix_b.shape)
        d.name = f"doom_{name}"
        d.scaling = weights[f"scaling"]
        to_json(d.get_model_params(d.get_outputs()),f"doom_data/doom_{name}.json")
        
            
            

        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, witness_file=f"doom_{name}_witness.txt", circuit_path=f"doom_{name}_circuit.txt")
        d.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"doom_{name}_witness.txt", circuit_path=f"doom_{name}_circuit.txt", write_json = True)

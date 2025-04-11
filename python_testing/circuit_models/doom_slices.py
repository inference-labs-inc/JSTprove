import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.circuit_components.convolution import Conv2DModel, Conv2DModelReLU
from python_testing.circuit_components.matrix_multiplication import MatrixMultiplicationModel, MatrixMultiplicationReLUModel


from python_testing.utils.helper_functions import RunType, read_from_json, to_json
from python_testing.utils.pytorch_helpers import ZKModel
from python_testing.circuit_models.doom_model import Doom, DoomAgent


# class DoomAgent(nn.Module):
#     def __init__(self, n_actions=7):
#         super(DoomAgent, self).__init__()

#         self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

#         self.fc_input_dim = 32 * 7 * 7

#         self.fc1 = nn.Linear(self.fc_input_dim, 256)
#         self.fc2 = nn.Linear(256, n_actions)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))

#         x = x.reshape(-1, self.fc_input_dim)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

#     def act(self, state, epsilon=0.0):
#         if np.random.random() < epsilon:
#             return np.random.randint(7)

#         with torch.no_grad():
#             state_tensor = (
#                 torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
#             )
#             q_values = self.forward(state_tensor)
#             return q_values.argmax().item()

class DoomSlice(ZKModel):
    def get_model_and_quantize(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.large_model_file_name, map_location=device)
        large_model = DoomAgent()
        large_model.load_state_dict(checkpoint["model_state_dict"])


        try:
            model = self.model_type(**getattr(self, 'model_params', {})).to(device)
        except AttributeError:
            raise NotImplementedError(f"Must specify the model type as a pytorch model (as variable self.model_type) in object {self.__class__.__name__}")
        except TypeError as e: 
            raise NotImplementedError(f"{e}. \n Must specify the model parameters of the pytorch model (as dictionary in self.model_params) in object {self.__class__.__name__}.")
        
        setattr(model,self.slice_name_in_model, getattr(large_model,self.large_model_slice_name))
        model.eval()
        self.model = model
        self.quantized_model = self.quantize_model(model, 2**self.scaling, rescale_config=getattr(self,"rescale_config",{}))
        self.quantized_model.eval()  

    def read_input(self, file_name = "doom_data/doom_input.json"):
        """Reads the inputs to each layer of the model from text files."""
        print(file_name)
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data["output"]
class DoomConv1(DoomSlice):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_conv1"
        self.input_data_file = "doom_data/doom_input.json"
        self.large_model_file_name = file_name
        # self.model_file_name = "model/doom_conv1_checkpoint.pth"


        self.scaling = 21
        self.input_shape = [1, 4, 28, 28]
        self.model_type = Conv2DModelReLU
        self.model_params = {"in_channels": 4, "out_channels": 16, "kernel_size": 3, "stride": 1, 'padding': 1}
        self.slice_name_in_model = "conv"
        self.large_model_slice_name = "conv1"

    def read_input(self, file_name = "doom_data/doom_input.json"):
        """Reads the inputs to each layer of the model from text files."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data["input"]
class DoomConv2(DoomSlice):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_conv2"
        self.input_data_file = "output/doom_conv1_output.json"
        self.large_model_file_name = file_name
        # self.model_file_name = "model/doom_conv1_checkpoint.pth"


        self.scaling = 21
        self.input_shape = [1, 16, 28, 28]
        self.model_type = Conv2DModelReLU
        self.model_params = {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, 'padding': 1}
        self.slice_name_in_model = "conv"
        self.large_model_slice_name = "conv2"

class DoomConv3(DoomSlice):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_conv3"
        self.input_data_file = "output/doom_conv2_output.json"
        self.large_model_file_name = file_name
        # self.model_file_name = "model/doom_conv1_checkpoint.pth"


        self.scaling = 21
        self.input_shape = [1, 32, 14, 14]
        self.model_type = Conv2DModelReLU
        self.model_params = {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 2, 'padding': 1}
        self.slice_name_in_model = "conv"
        self.large_model_slice_name = "conv3"

class DoomFC1(DoomSlice):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_fc1"
        self.input_data_file = "output/doom_conv3_output.json"
        self.large_model_file_name = file_name
        # self.model_file_name = "model/doom_conv1_checkpoint.pth"


        self.scaling = 21
        self.input_shape = [1, 1568]
        self.model_type = MatrixMultiplicationReLUModel
        self.model_params = {"in_channels": 1568, "out_channels":256, "bias" : True}
        self.slice_name_in_model = "fc1"
        self.large_model_slice_name = "fc1"

    def read_input(self, file_name = "doom_data/doom_input.json"):
        """Reads the inputs to each layer of the model from text files."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            x = torch.tensor(data["output"])
            return x.reshape([-1,1568])#.view(x.size(0), -1)
        
class DoomFC2(DoomSlice):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_fc2"
        self.input_data_file = "output/doom_fc1_output.json"
        self.large_model_file_name = file_name
        # self.model_file_name = "model/doom_conv1_checkpoint.pth"


        self.scaling = 21
        self.input_shape = [1, 256]
        self.model_type = MatrixMultiplicationModel
        self.rescale_config = {"fc1": False}
        self.model_params = {"in_channels": 256, "out_channels":7, "bias":False}
        self.slice_name_in_model = "fc1"
        self.large_model_slice_name = "fc2"

if __name__ == "__main__":
    # Doom().base_testing()
    names = [
        "conv1", 
        "conv2", 
        "conv3",
        "fc1",
        "fc2"
        ]
    name = "doom"
    d = Doom()
    # d.base_testing()
    # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = Doom()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

    d = DoomConv1()
    name = "doom_conv1"
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = DoomConv1()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

    d = DoomConv2()
    name = "doom_conv2"
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = DoomConv2()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_2.input_data_file, write_json = False)

    d = DoomConv3()
    name = "doom_conv3"
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = DoomConv3()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_2.input_data_file, write_json = False)

    d = DoomFC1()
    name = "doom_fc1"
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = DoomFC1()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_2.input_data_file, write_json = False)

    d = DoomFC2()
    name = "doom_fc2"
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = DoomFC2()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_2.input_data_file, write_json = False)
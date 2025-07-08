import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.circuit_components.convolution import Conv2DModel, Conv2DModelReLU
from python_testing.circuit_components.matrix_multiplication import MatrixMultiplicationModel, MatrixMultiplicationReLUModel


from python_testing.utils.helper_functions import RunType, read_from_json, to_json
from python_testing.utils.pytorch_helpers import ZKTorchModel
from python_testing.circuit_models.doom_model import Doom, DoomAgent



class Conv1Segment(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x


class Conv2Segment(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv2(x))
        return x


class Conv3Segment(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Flatten before the FC layer
        x = F.relu(self.conv3(x))
        return x
    

class FC1Segment(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_input_dim = 32 * 7 * 7
        self.fc1 = nn.Linear(self.fc_input_dim, 256, bias=True)

    def forward(self, x):
        x = x.reshape(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        return x
    
class FC2Segment(nn.Module):
    def __init__(self, n_actions = 7):
        super().__init__()
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.fc2(x)
        return x




class DoomSlice(ZKTorchModel):
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
        self.quantized_model = self.quantize_model(model, self.scale_base**self.scaling, rescale_config=getattr(self,"rescale_config",{}))
        self.quantized_model.eval()  
        

class DoomConv1(ZKTorchModel):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_conv1"
        self.input_data_file = "doom_data/doom_input.json"
        self.model_file_name = "model/doom/conv_0.pt"

        
        self.scaling = Doom().scaling
        self.scale_base = Doom().scale_base
        self.input_shape = [1, 4, 28, 28]
        self.model_type = Conv1Segment
        # self.model_params = {"in_channels": 4, "out_channels": 16, "kernel_size": 3, "stride": 1, 'padding': 1}
          
class DoomConv2(ZKTorchModel):

    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_conv2"
        self.input_data_file = "output/doom_conv1_output.json"
        self.large_model_file_name = file_name
        self.model_file_name = "model/doom/conv_1.pt"


        self.scale_base = Doom().scale_base
        self.scaling = Doom().scaling
        self.input_shape = [1, 16, 28, 28]
        self.model_type = Conv2Segment
        # self.model_params = {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, 'padding': 1}

class DoomConv3(ZKTorchModel):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_conv3"
        self.input_data_file = "output/doom_conv2_output.json"
        self.model_file_name = "model/doom/conv_2.pt"


        self.scaling = Doom().scaling
        self.scale_base = Doom().scale_base
        self.input_shape = [1, 32, 14, 14]
        self.model_type = Conv3Segment
        # self.model_params = {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 2, 'padding': 1}
        # self.slice_name_in_model = "conv"

class DoomFC1(ZKTorchModel):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_fc1"
        self.input_data_file = "output/doom_conv3_output.json"
        self.model_file_name = "model/doom/fc_3.pt"


        self.scaling = Doom().scaling
        self.scale_base = Doom().scale_base
        self.input_shape = [1, 32, 7, 7]
        self.model_type = FC1Segment
        # self.model_params = {"in_channels": 1568, "out_channels":256, "bias" : True}

    # def get_outputs(self, inputs):
    #     return super().get_outputs(inputs.flatten().unsqueeze(0))
        
class DoomFC2(ZKTorchModel):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.required_keys = ["input"]
        self.name = "doom_fc2"
        self.input_data_file = "output/doom_fc1_output.json"
        # self.large_model_file_name = file_name
        self.model_file_name = "model/doom/fc_4.pt"


        self.scale_base = Doom().scale_base
        self.scaling = Doom().scaling
        self.input_shape = [1, 256]
        self.model_type = FC2Segment
        self.rescale_config = {"fc2": False}
        # self.model_params = {"in_channels": 256, "out_channels":7, "bias":False}
        # self.slice_name_in_model = "fc1"
        # self.large_model_slice_name = "fc2"

if __name__ == "__main__":
    # Doom().base_testing()
    names = [
        "conv1", 
        "conv2", 
        "conv3",
        "fc1",
        "fc2"
        ]
    # name = "doom"
    # d = Doom()
    # # d.base_testing()
    # # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    # d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # # d.save_quantized_model("quantized_model.pth")
    # d_2 = Doom()
    # # d_2.load_quantized_model("quantized_model.pth")
    # d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

    d = DoomConv1()
    name = "doom_conv1"
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d.save_quantized_model("quantized_model.pth")
    d_2 = DoomConv1()
    # d_2.load_quantized_model("quantized_model.pth")
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt",input_file="inputs/doom_input.json", circuit_path=f"{name}_circuit.txt", write_json = False)
    # d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

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
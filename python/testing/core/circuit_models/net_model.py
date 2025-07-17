from python.testing.core.utils.helper_functions import RunType
import torch
import torch.nn as nn
import torch.nn.functional as F
from python.testing.core.utils.pytorch_helpers import ZKTorchModel
# from python_testing.circuit_models.doom_slices import Slice

class Net(nn.Module):
    def __init__(self, target_params=None):
        super().__init__()

        # Default values
        c1, c2 = 6, 16
        fc1, fc2, fc3 = 120, 84, 10

        # if target_params:
        #     # Adjust parameters iteratively to fit within target
        #     scale = (target_params / estimate_params(c1, c2, fc1, fc2, fc3)) ** 0.5
        #     c1, c2 = int(c1 * scale), int(c2 * scale)
        #     fc1, fc2 = int(fc1 * scale), int(fc2 * scale)

        #     # Recalculate to get final parameter count
        #     final_params = estimate_params(c1, c2, fc1, fc2, fc3)
        #     assert abs(final_params - target_params) / target_params <= 0.05, "Final params exceed 5% tolerance"

        self.conv1 = nn.Conv2d(3, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.fc1 = nn.Linear(c2 * 5 * 5, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv1Segment(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x


class Conv2Segment(nn.Module):
    def __init__(self, in_channels=6, out_channels=16):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        return x


class FC1Segment(nn.Module):
    def __init__(self, in_features=16 * 5 * 5, out_features=120):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Flatten before the FC layer
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x
    


class FC2Segment(nn.Module):
    def __init__(self, in_features=120, out_features=84):
        super().__init__()
        self.fc2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        return x


class FC3Segment(nn.Module):
    def __init__(self, in_features=84, out_features=10):
        super().__init__()
        self.fc3 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc3(x)
    
# class Slice(ZKModel):
#     def get_model_and_quantize(self):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         checkpoint = torch.load(self.large_model_file_name, map_location=device)
        
#         large_model = Net()
#         if "model_state_dict" in checkpoint.keys():
#             large_model.load_state_dict(checkpoint["model_state_dict"])
#         else:
#             large_model.load_state_dict(checkpoint)

        
#         try:
#             model = self.model_type(**getattr(self, 'model_params', {})).to(device)
#         except AttributeError:
#             raise NotImplementedError(f"Must specify the model type as a pytorch model (as variable self.model_type) in object {self.__class__.__name__}")
#         except TypeError as e: 
#             raise NotImplementedError(f"{e}. \n Must specify the model parameters of the pytorch model (as dictionary in self.model_params) in object {self.__class__.__name__}.")
        
#         for name in self.slice_name_in_model:
#             setattr(model,name, getattr(large_model,name))
#         model.eval()
#         self.model = model
#         self.quantized_model = self.quantize_model(model, 2**self.scaling, rescale_config=getattr(self,"rescale_config",{}))
#         self.quantized_model.eval()  

#     def read_input(self, file_name = "doom_data/doom_input.json"):
#         """Reads the inputs to each layer of the model from text files."""
#         print(file_name)
#         with open(file_name, 'r') as file:
#             data = json.load(file)
#             return data["input"]


# class NetConv1Model(ZKModel):
#     def __init__(self):
#         self.required_keys = ["input"]
#         self.name = "net_conv1"
#         self.input_data_file = "input/net_input.json"
#         # self.model_file_name = "net_model.pth"


#         self.scaling = 21
#         self.input_shape = [1,3,32,32]
#         self.rescale_config = {}
#         self.model_type = Conv1Segment
class NetModel(ZKTorchModel):
    def __init__(self):
        self.required_keys = ["input"]
        self.name = "net"
        self.input_data_file = "input/net_input.json"
        self.model_file_name = "model/net/model.pth"


        self.scale_base = 2
        self.scaling = 21
        self.input_shape = [1,3,32,32]
        self.rescale_config = {"fc3": False}
        self.model_type = Net

class NetConv1Model(ZKTorchModel):
    def __init__(self):
        self.required_keys = ["input"]
        self.name = "net_conv1"
        self.input_data_file = "input/net_input.json"
        self.model_file_name = "model/net/conv_0.pt"

        self.scale_base = 2
        self.scaling = 21
        self.input_shape = [1,3,32,32]
        self.rescale_config = {}
        self.model_type = Conv1Segment
        
class NetConv2Model(ZKTorchModel):
    def __init__(self):
        self.required_keys = ["input"]
        self.name = "net_conv2"
        self.input_data_file = "output/net_conv1_output.json"
        self.model_file_name = "model/net/conv_1.pt"

        self.scale_base = 2
        self.scaling = 21
        self.input_shape = [1,6,14,14]
        self.rescale_config = {}
        self.model_type = Conv2Segment

class NetFC1Model(ZKTorchModel):
    def __init__(self):
        self.required_keys = ["input"]
        self.name = "net_fc1"
        self.input_data_file = "output/net_conv2_output.json"
        self.model_file_name = "model/net/fc_2.pt"

        self.scale_base = 2
        self.scaling = 21
        self.input_shape = [1,16,5,5]
        self.rescale_config = {}
        self.model_type = FC1Segment


class NetFC2Model(ZKTorchModel):
    def __init__(self):
        self.required_keys = ["input"]
        self.name = "net_fc2"
        self.input_data_file = "output/net_fc1_output.json"
        self.model_file_name = "model/net/fc_3.pt"

        self.scale_base = 2
        self.scaling = 21
        self.input_shape = [1,120]
        self.rescale_config = {}
        self.model_type = FC2Segment


class NetFC3Model(ZKTorchModel):
    def __init__(self):
        self.required_keys = ["input"]
        self.name = "net_fc3"
        self.input_data_file = "output/net_fc2_output.json"
        self.model_file_name = "model/net/fc_4.pt"

        self.scale_base = 2
        self.scaling = 21
        self.input_shape = [1,84]
        self.rescale_config = {"fc3": False}
        self.model_type = FC3Segment

    

if __name__ == "__main__":
    name = "net"
    d = NetModel()
    # # d.save_model("net_model.pth")
    # # sys.exit()

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d_2 = NetModel()
    # d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)

    d_3 = NetModel()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file="inputs/random_input_1.json", write_json = False)

    # d_3.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file="inputs/random_input.json", write_json = False)
    # d_3.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file="inputs/random_input.json", write_json = False)


    name = "net_conv1"
    d = NetConv1Model()
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    # d_2 = NetConv1Model()
    # d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)

    d_3 = NetConv1Model()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file="inputs/random_input_1.json",  write_json = False)

    name = "net_conv2"
    d = NetConv2Model()
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_3 = NetConv2Model()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_3.input_data_file, write_json = False)

    name = "net_fc1"
    d = NetFC1Model()
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_3 = NetFC1Model()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_3.input_data_file, write_json = False)

    name = "net_fc2"
    d = NetFC2Model()
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_3 = NetFC2Model()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_3.input_data_file, write_json = False)

    name = "net_fc3"
    d = NetFC3Model()
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_3 = NetFC3Model()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", input_file=d_3.input_data_file, write_json = False)
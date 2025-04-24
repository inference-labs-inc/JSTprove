import torch
from python_testing.utils.helper_functions import RunType
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.utils.pytorch_helpers import ZKModel

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
    

class NetModel(ZKModel):
    def __init__(self):
        self.required_keys = ["input"]
        self.name = "net"
        self.input_data_file = "input/net_input.json"
        # self.model_file_name = file_name


        self.scaling = 21
        self.input_shape = [1,3,32,32]
        self.rescale_config = {"fc3": False}
        self.model_type = Net

    

if __name__ == "__main__":
    name = "net"
    d = NetModel()
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = NetModel()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    # d.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
    # d.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")

    d_3 = NetModel()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)
    # d_3.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
    # d_3.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")

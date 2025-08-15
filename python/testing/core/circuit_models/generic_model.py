import torch.nn as nn
import torch.nn.functional as F
from python.testing.core.utils.pytorch_helpers import ZKTorchModel
from python.testing.core.utils.helper_functions import RunType



class CNNDemo(nn.Module):
    def __init__(self, n_actions=10, layers=["conv1", "relu", "conv2", "relu", "conv3", "relu", "reshape", "fc1", "relu", "fc2"]):
        super(CNNDemo, self).__init__()
        self.layers = layers 

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        for i in range(2,40):
            self.__setattr__(f"conv{i}", nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))

        # Default to the shape after conv layers (depends on whether each conv layer is used)
        self.fc_input_dim = 16 * 28 * 28

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        for i in range(2,1000):
            self.__setattr__(f"fc{i}", nn.Linear(256, 256))

        self.final = nn.Linear(256, n_actions)

        # If the last layer in the list is one of the fully connected layers, set them to `final`
        if layers[-1] == "fc1":
            self.fc1 = nn.Linear(self.fc_input_dim, n_actions)
        else: 
            self.__setattr__(layers[-1], self.final)


    def forward(self, x):
        print(self.layers)
        for l in self.layers:
            if "fc1" == l:
                x = self.fc1(x)
                continue
            if "fc" in l:
                layer_fn = self.__getattr__(l)  # Get the function
                if callable(layer_fn):  # Ensure it's callable
                    x = layer_fn(x)  # Call it with parameter x
            if "reshape" in l:
                x = x.reshape(-1, self.fc_input_dim)  # Flatten before fully connected layers
            if "conv" in l:
                layer_fn = self.__getattr__(l)  # Get the function
                if callable(layer_fn):  # Ensure it's callable
                    x = layer_fn(x)  # Call it with parameter x
            if "relu" in l:
                x = F.relu(x)
        return x
    
class GenericDemo(ZKTorchModel):
    def __init__(self, model_file_path: str = None, quantized_model_file_path: str = None, layers = None):
        if layers == None:
            self.layers = []
            # Add conv layers
            for i in range(1,2):
                self.layers.append(f"conv{i}")
                self.layers.append("relu")
            self.layers.append("reshape")
            self.layers.append("fc1")

            # # Add FC layers
            for i in range(2,3):
                self.layers.append("relu")
                self.layers.append(f"fc{i}")
        else:
            self.layers = layers

        self.name = "generic_demo"
        self.required_keys = ["input"]
        self.scale_base = 2
        self.scaling = 21


        self.input_shape = [1, 4, 28, 28]
        self.rescale_config = {self.layers[-1]: False}
        self.model_type = CNNDemo
        self.model_params = {"layers": self.layers, "n_actions": 10}

    

if __name__ == "__main__":
    # names = ["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]
    names = ["demo"]
    for n in names:
        # name = f"{n}_conv1"
        name = n
        d = GenericDemo()
        # d.base_testing()
        # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
        # d.save_quantized_model("quantized_model.pth")
        d_2 = GenericDemo()
        # # # d_2.load_quantized_model("quantized_model.pth")
        d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        # # d.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        # # d.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")

import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from python_testing.relu import ReLU, ConversionType



class LayerInfo():
    def __init__(self, name, input_shape, output_shape, weight_shape = None):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight_shape = weight_shape
        self.inputs = None
        self.outputs = None
        self.weights = None

    def update_inputs(self, inputs):
        self.inputs = inputs.reshape(self.input_shape)

    def update_outputs(self, outputs):
        self.outputs = outputs.reshape(self.output_shape)

    def update_weights(self, weights):
        if self.weight_shape:
            self.weights = weights.reshape(self.weight_shape)
        else:
            self.weights = None


class Doom():
    def __init__(self):
        self.layers = {}
        
        self.layers["input"] = LayerInfo("input", [4, 28, 28], [4, 28, 28])
        
        self.layers["conv1"] = LayerInfo("conv1", [4, 28, 28], [16,28,28], [16, 4, 3, 3])
        self.layers["conv1_relu"] = LayerInfo("conv1_relu", [16,28,28], [16,28,28])

        self.layers["conv2"] = LayerInfo("conv2", [16,28,28], [32,14,14], [32, 16, 3, 3])
        self.layers["conv2_relu"] = LayerInfo("conv2_relu", [32,14,14], [32,14,14])

        self.layers["conv3"] = LayerInfo("conv3", [32,14,14], [32,7,7], [32, 32, 3, 3])
        self.layers["conv3_relu"] = LayerInfo("conv3_relu", [32,7,7], [32,7,7])

        self.layers["reshape"] = LayerInfo("reshape", [32,7,7], [1568])

        self.layers["fc1"] = LayerInfo("fc1", [1568], [256], [256, 1568])
        self.layers["fc1_relu"] = LayerInfo("fc1_relu", [256], [256])

        self.layers["fc2"] = LayerInfo("fc2", [256], [7], [7, 256])
        self.layers["output"] = LayerInfo("output", [7], [7])

        
    def read_tensor_from_file(self, file_name):
        """Reads a tensor from a file and returns it as a PyTorch tensor."""
        with open(file_name, 'r') as f:
            data = f.read().split()
            # Convert data to a float and then to a PyTorch tensor
            tensor_data = torch.tensor([float(d) for d in data])
        return tensor_data
    
    def read_weights(self, layer_name, base_dir="doom_data"):
        """Reads the weights for the layers of the model from files."""
        # layer_files = [f for f in os.listdir(base_dir) if "weights" in f]  # Assuming weights are in files named with 'weights'

        prefix = f"{base_dir}/weight"

        file_name = f"{prefix}_{layer_name}.txt"

        if ("input" in layer_name) or ("output" in layer_name) or ("relu" in layer_name) or ("reshape" in layer_name):
            print("This layer should not have weights")
            return
        
        if not os.path.exists(file_name):
            print("Weights do not exist for this layer")
            return

        weight_tensor = self.read_tensor_from_file(file_name)
        self.layers[layer_name].update_weights(weight_tensor)
        print(f"Read weights for layer {layer_name}: {weight_tensor.shape}")

    def read_input(self, layer_name, sample_idx = 0, seed = 0, point = 0, base_dir="doom_data"):
        """Reads the inputs to each layer of the model from text files."""

        doom_layers = [
            "conv1", "conv1_relu", "conv2", "conv2_relu","conv3", "conv3_relu",
            "fc1", "fc1_relu", "fc2", 
            "input", "output", "reshape"]
        if "input" in layer_name:
            input_layer = layer_name
        else:
            input_layer = doom_layers[doom_layers.index(layer_name) - 1]

        prefix = f"{base_dir}/sample_{sample_idx}_seed{seed}_point_{point}"

        file_name = f"{prefix}_{input_layer}.txt"
        
        input_tensor = self.read_tensor_from_file(file_name)
        # Initialize LayerInfo with inputs
        self.layers[layer_name].update_inputs(input_tensor)
        print(f"Read input for layer {input_layer}: {input_tensor.shape}")


    def read_output(self, layer_name, sample_idx = 0, seed = 0, point = 0, base_dir="doom_data"):
        """Reads the outputs for each layer of the model from text files."""
        

        prefix = f"{base_dir}/sample_{sample_idx}_seed{seed}_point_{point}"

        file_name = f"{prefix}_{layer_name}.txt"
        
        output_tensor = self.read_tensor_from_file(file_name)
        if layer_name in self.layers:
            self.layers[layer_name].update_outputs(output_tensor)
        else:
            print(f"Layer {layer_name} not found. Skipping output loading.")
        print(f"Read output for layer {layer_name}: {output_tensor.shape}")

    def run_circuit(self):
        """Simulates running the model by passing inputs through layers with weights."""
        print("Running circuit...")
        
        #Relu 1
        self.read_input("conv1_relu")
        self.read_output("conv1_relu")

        proof_system = ZKProofSystems.Expander
        proof_folder = "analysis"
        output_folder = "output"
        temp_folder = "temp"
        input_folder = "inputs"
        weights_folder = "weights"
        circuit_folder = ""
        #Rework inputs to function
        test_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
        test_circuit.inputs_1 = self.layers["conv1_relu"].inputs
        test_circuit.outputs = self.layers["conv1_relu"].outputs
        test_circuit.convert_to_relu_form()
        test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, weights_folder, proof_system, output_folder)

if __name__ == "__main__":
    Doom().run_circuit()


    
    

    
    # proof_system = ZKProofSystems.Expander
    # proof_folder = "analysis"
    # output_folder = "output"
    # temp_folder = "temp"
    # input_folder = "inputs"
    # weights_folder = "weights"
    # circuit_folder = ""
    # #Rework inputs to function
    # test_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
    # test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, weights, proof_system, output_folder)


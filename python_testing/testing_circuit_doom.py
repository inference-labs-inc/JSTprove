import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from python_testing.relu import ReLU, ConversionType

from python_testing.convolution import Convolution, QuantizedConv



class LayerInfo():
    def __init__(self, name, input_shape, output_shape, weight_shape = None):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight_shape = weight_shape
        self.inputs = None
        self.outputs = None
        self.weights = None
        self.bias = None

    def update_inputs(self, inputs):
        self.inputs = inputs.reshape(self.input_shape)

    def update_outputs(self, outputs):
        self.outputs = outputs.reshape(self.output_shape)

    def update_weights(self, weights):
        if self.weight_shape:
            self.weights = weights.reshape(self.weight_shape)
        else:
            self.weights = None

    # def update_weights(self, bias):
    #     if self.weight_shape:
    #         self.bias = bias.reshape(self.weight_shape)
    #     else:
    #         self.weights = None


class Doom():
    def __init__(self):
        self.layers = {}

        self.scaling = 21
        
        self.layers["input"] = LayerInfo("input", [1, 4, 28, 28], [1, 4, 28, 28])
        
        self.layers["conv1"] = LayerInfo("conv1", [1, 4, 28, 28], [1, 16,28,28], [16, 4, 3, 3])
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
    
    def read_weights(self, layer_name, base_dir="doom_weights", is_weights = True):
        """Reads the weights for the layers of the model from files."""
        # layer_files = [f for f in os.listdir(base_dir) if "weights" in f]  # Assuming weights are in files named with 'weights'
        if is_weights:
            prefix = f"{base_dir}/weight"
        else:
            prefix = f"{base_dir}/bias"

        file_name = f"{prefix}_{layer_name}.txt"

        if ("input" in layer_name) or ("output" in layer_name) or ("relu" in layer_name) or ("reshape" in layer_name):
            print("This layer should not have weights")
            return
        
        if not os.path.exists(file_name):
            print("Weights do not exist for this layer")
            return

        weight_tensor = self.read_tensor_from_file(file_name)
        if is_weights:
            self.layers[layer_name].update_weights(weight_tensor)
        else:
            self.layers[layer_name].bias = weight_tensor
        print(f"Read weights for layer {layer_name}: {weight_tensor.shape}")

    def read_layer_shape(self, layer_name, base_dir="doom_weights"):

        prefix = f"{base_dir}/layer"

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
            "input", "conv1", "conv1_relu", "conv2", "conv2_relu","conv3", "conv3_relu",
            "fc1", "fc1_relu", "fc2", 
            "output"]
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
        proof_system = ZKProofSystems.Expander
        proof_folder = "analysis"
        output_folder = "output"
        temp_folder = "temp"
        input_folder = "inputs"
        weights_folder = "weights"
        circuit_folder = ""
        name = "doom"

        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_file, output_file = get_files(
                input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, name, output_folder, proof_system)

        exclude_keys = ['quantized', 'scaling']
        
        #Rework inputs to function
        (conv_1_inputs, conv_1_weights, conv_1_outputs) = self.get_circuit_conv_1()
        conv_1_output_tensor = torch.IntTensor(conv_1_outputs["conv_out"])
        weights = {"conv_1_" + key if key not in exclude_keys else key: value for key, value in conv_1_weights.items()}


        (relu_inputs, relu_outputs) = self.get_relu_1(conv_1_output_tensor)
        relu_1_output_tensor = torch.IntTensor(relu_outputs["outputs"])
        relu_1_input_tensor = torch.IntTensor(relu_inputs["inputs_1"])

        # Check outputs of conv is same as inputs of relu
        self.check_4d_eq(conv_1_output_tensor,relu_1_input_tensor)

        (conv_2_inputs, conv_2_weights, conv_2_outputs) = self.get_circuit_conv_2(relu_1_output_tensor)
        conv_2_input_tensor = torch.IntTensor(conv_2_inputs["input_arr"])
        conv_2_output_tensor = torch.IntTensor(conv_2_outputs["conv_out"])
        # Check outputs of relu is same as inputs of conv
        self.check_4d_eq(relu_1_output_tensor,conv_2_input_tensor)

        conv_2_weights = {"conv_2_" + key if key not in exclude_keys else key: value for key, value in conv_2_weights.items()}

        weights.update(conv_2_weights)

        (relu_2_inputs, relu_2_outputs) = self.get_relu_1(conv_2_output_tensor)
        relu_2_output_tensor = torch.IntTensor(relu_2_outputs["outputs"])
        relu_2_input_tensor = torch.IntTensor(relu_2_inputs["inputs_1"])
        # Check inputs of relu is same as outputs of conv
        self.check_4d_eq(relu_2_input_tensor,conv_2_output_tensor)
        

        # NO NEED TO CHANGE anything below here!
        to_json(conv_1_inputs, input_file)

        # Write output to json
        outputs = {"outputs": value for key, value in relu_2_outputs.items()}
        to_json(outputs, output_file)

        to_json(weights, weights_file)

        ## Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)

    def check_4d_eq(self, input_tensor_1, input_tensor_2):
        for i in range(input_tensor_1.shape[0]):
            for j in range(input_tensor_1.shape[1]):
                for k in range(input_tensor_1.shape[2]):
                    for l in range(input_tensor_1.shape[3]):
                        assert(abs(input_tensor_1[i][j][k][l] -  input_tensor_2[i][j][k][l]) < 1)

    def get_circuit_conv_1(self):
        self.read_input("conv1")
        self.read_output("conv1")
        self.read_weights("conv1")
        self.read_weights("conv1", is_weights=False)
        conv1_circuit = QuantizedConv()
        conv1_circuit.input_arr = torch.mul(self.layers["conv1"].inputs,2**self.scaling).long()
        # conv1_circuit.out = torch.mul(self.layers["conv1"].outputs, 2**self.scaling).long()
        conv1_circuit.weights = torch.mul(self.layers["conv1"].weights, 2**self.scaling).long()
        conv1_circuit.bias = torch.mul(self.layers["conv1"].bias, 2**self.scaling).long()
        conv1_circuit.scaling = self.scaling
        # conv1_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)
        return conv1_circuit.get_model_params(conv1_circuit.get_output())
    

    def get_relu_1(self, inputs):
        test_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
        test_circuit.inputs_1 = inputs
        out = test_circuit.get_outputs()
        return test_circuit.get_twos_comp_model_data(out)
        # test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, weights_folder, proof_system, output_folder)


    def get_circuit_conv_2(self, inputs):
        self.read_input("conv2")
        self.read_output("conv2")
        self.read_weights("conv2")
        self.read_weights("conv2", is_weights=False)
        layers = self.layers["conv2"]
        conv2_circuit = QuantizedConv()
        # conv1_circuit.input_arr = torch.mul(self.layers["conv1"].inputs,2**self.scaling).long()
        conv2_circuit.input_arr = inputs
        # conv2_circuit.out = torch.mul(layers.outputs, 2**self.scaling).long()
        conv2_circuit.weights = torch.mul(layers.weights, 2**self.scaling).long()
        conv2_circuit.bias = torch.mul(layers.bias, 2**self.scaling).long()

        conv2_circuit.scaling = self.scaling
        conv2_circuit.strides = (2,2)
        # conv1_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)
        return conv2_circuit.get_model_params(conv2_circuit.get_output())

        

    

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


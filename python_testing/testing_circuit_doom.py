from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from enum import Enum


class ConversionType(Enum):
    DUAL_MATRIX = "dual_matrix"
    TWOS_COMP = "twos_comp"
    SIGNED_MAG = "signed_mag"

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
        circuit_folder = ""
        #Rework inputs to function
        test_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
        test_circuit.inputs_1 = self.layers["conv1_relu"].inputs
        test_circuit.outputs = self.layers["conv1_relu"].outputs
        test_circuit.convert_to_relu_form()
        test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, proof_system, output_folder)




class ReLU():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, conversion_type):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        self.conversion_type = conversion_type
        if conversion_type == ConversionType.DUAL_MATRIX:
            # Specify
            self.name = "relu"
            
            # Function input generation

            self.inputs_1 = torch.randint(low=0, high=100000000, size=(256,))
            self.inputs_2 = torch.randint(low=0, high=2, size=(256,))
            self.outputs = None
            self.scaling = 2 ** 21
        elif conversion_type == ConversionType.TWOS_COMP:
            # Specify
            self.name = "relu_twos_comp"
            
            # Function input generation

            # self.inputs_1 = torch.randint(low=-2**21, high=2**21, size=(1000,))
            self.inputs_1 = torch.randint(low=-2**31, high=2**31, size=(16,8,4))

            self.outputs = None
            self.scaling = 2 ** 21

        elif conversion_type == ConversionType.SIGNED_MAG:
            self.name = "relu_twos_comp"
            
            # Function input generation

            self.inputs_1 = torch.randint(low=0, high=100000000, size=(256,))
            self.outputs = None
            self.scaling = 2 ** 21

        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    def convert_to_relu_form(self, num_bits = 32):

        def twos_comp_integer(val, bits):
            """compute the 2's complement of int value val"""
            return int(f"{val & ((1 << bits) - 1):0{bits}b}", 2)
        
        def twos_comp(val, bits):
            """compute the 2's complement of int value val"""
            mask = 2**torch.arange(bits - 1, -1, -1).to(val.device, val.dtype)
            return val.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        
        def bin2dec(b, bits):
            mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
            return torch.sum(mask * b, -1)
        
        def test(inputs):
            def to_binary_2s(x, n_bits):
                res = []
                for i in range(n_bits):
                
                    y = x >> i
                    res.append(y & 1)
                return res
            

            def binary_check(bits):
                for i in range(len(bits)):
                    bin_check = 1 - bits[i]
                    x = bin_check * bits[i]
                    assert(x == 0)

            def from_binary_2s(bits, n_bits):
                res = 0
                length = n_bits - 1
                for i in range(length):
                    coef = 1 << i
                    cur = coef * bits[i]
                    res = res + cur
                binary_check(bits)
                cur = 1 << length * bits[length]
                temp = 0
                out = temp - cur
                return out + res
            
            for i in range(len(inputs)):
                for j in range(len(inputs[i])):
                    for k in range(len(inputs[i][j])):
                        bits = to_binary_2s(inputs[i][j][k], 32)
                        total = from_binary_2s(bits, 32)
                        assert(total, inputs[i][j][k])

                

        

        if self.conversion_type ==ConversionType.DUAL_MATRIX:
            self.inputs_2 = (self.inputs_1 < 0).int()
            self.inputs_1 = torch.mul(torch.abs(self.inputs_1), self.scaling)
        elif self.conversion_type:
            self.inputs_1 = torch.mul(self.inputs_1, self.scaling).int()
            # print(self.inputs_1[0][0][2])
            # self.inputs_1 = twos_comp(self.inputs_1, 32)
            # self.inputs_1 =  torch.tensor([twos_comp(val.item(), num_bits) for val in self.inputs_1.flatten()])
            # print(self.inputs_1[0][0][2])


        else:
            pass



    
    def base_testing(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str, proof_system: ZKProofSystems, output_folder: str = None):

        # NO NEED TO CHANGE!
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, self.name, output_folder, proof_system)
        

        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        ## Perform calculation here

        # outputs = torch.where(self.inputs_1 > self.inputs_2, torch.tensor(1), 
        #              torch.where(self.inputs_1 == self.inputs_2, torch.tensor(0), torch.tensor(-1)))
        if self.conversion_type == ConversionType.TWOS_COMP:
            if self.outputs == None:
                outputs = torch.relu(self.inputs_1)
            else:
                outputs = torch.mul(self.outputs, self.scaling)
        elif self.conversion_type == ConversionType.DUAL_MATRIX:
            if self.outputs == None:
                inputs_3 = torch.mul(torch.sub(1,torch.mul(self.inputs_2,2)),self.inputs_1)
                outputs = torch.relu(inputs_3)
            else:
                outputs = torch.mul(self.outputs, self.scaling)

        ## Define inputs and outputs
        if self.conversion_type == ConversionType.TWOS_COMP:
            inputs = {
                'inputs_1': self.inputs_1.int().tolist()
                }
            outputs = {
                'outputs': outputs.int().tolist(),
            }
        elif self.conversion_type == ConversionType.DUAL_MATRIX:
            try:
                inputs = {
                    'inputs_1': [int(i) for i in self.inputs_1.tolist()],
                    'inputs_2': [int(i) for i in self.inputs_2.tolist()]
                    }
                outputs = {
                    'outputs': [int(i) for i in outputs.tolist()],
                }
            except:
                inputs = {
                    'inputs_1': self.inputs_1.int().tolist(),
                    'inputs_2': self.inputs_2.int().tolist()
                    }
                outputs = {
                    'outputs': outputs.int().tolist(),
                }
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

        # When needed, can specify model parameters into json as well



        # NO NEED TO CHANGE anything below here!
        to_json(inputs, input_file)
        # Write output to json
        to_json(outputs, output_file)

        ## Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)

    
if __name__ == "__main__":
    Doom().run_circuit()


    
    

    
    # proof_system = ZKProofSystems.Expander
    # proof_folder = "analysis"
    # output_folder = "output"
    # temp_folder = "temp"
    # input_folder = "inputs"
    # circuit_folder = ""
    # #Rework inputs to function
    # test_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
    # test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, proof_system, output_folder)


import json
import torch
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort

from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
from python_testing.relu import ReLU, ConversionType
from python_testing.convolution import Convolution, QuantizedConv
from python_testing.gemm import QuantizedGemm, Gemm
from types import SimpleNamespace

class GeneralLayerFunctions():
    def check_4d_eq(self, input_tensor_1, input_tensor_2):
        for i in range(input_tensor_1.shape[0]):
            for j in range(input_tensor_1.shape[1]):
                for k in range(input_tensor_1.shape[2]):
                    for l in range(input_tensor_1.shape[3]):
                        assert(abs(input_tensor_1[i][j][k][l] - input_tensor_2[i][j][k][l]) < 1)

    def weights_onnx_to_torch_format(self, onnx_model):
        w_and_b = {}
        for i in onnx_model.graph.initializer:
            layer_name, param_type = i.name.split(".")  # Split into layer name and param type
            if layer_name not in w_and_b:
                w_and_b[layer_name] = {}

            w_and_b[layer_name][param_type] = torch.tensor(onnx.numpy_helper.to_array(i))

        for e in w_and_b.keys():
            w_and_b[e] = SimpleNamespace(**w_and_b[e])
        return w_and_b

    def check_2d_eq(self, input_tensor_1, input_tensor_2):
        for i in range(input_tensor_1.shape[0]):
            for j in range(input_tensor_1.shape[1]):
                assert(abs(input_tensor_1[i][j] -  input_tensor_2[i][j]) < 1)

    def read_tensor_from_file(self, file_name):
        """Reads a tensor from a file and returns it as a PyTorch tensor."""
        with open(file_name, 'r') as f:
            data = f.read().split()
            # Convert data to a float and then to a PyTorch tensor
            tensor_data = torch.tensor([float(d) for d in data])
        return tensor_data
    
    def read_weights(self, model, layer_name):
        """Reads the weights for the layers of the model from files."""
        pass


    def read_input(self, file_name = "doom_data/doom_input.json"):
        """Reads the inputs to each layer of the model from text files."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data["input_data"]



    def read_output(self, model, input_data, is_torch = True):
        """Reads the outputs for each layer of the model from text files."""
        if is_torch:
            with torch.no_grad():  # Disable gradient calculation during inference
                output = model(torch.tensor(input_data))
                return output
        else:
            session = ort.InferenceSession(model)

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data})
            return outputs


        


class Layers():
    def get_layer(self, inputs, layer_name, layer, **kwargs):
        if layer_name == "input":
            return self.get_inputs(kwargs.get("file_name", ".json"))
        elif "conv" in layer_name:
            return self.get_circuit_conv(inputs, layer, kwargs.get("strides", (1,1)),kwargs.get("kernel_shape",(3,3)), kwargs.get("group", [1]), kwargs.get("dilation", (1,1)), kwargs.get("pads", (1,1,1,1)))
        elif "relu" in layer_name:
            return self.get_relu(inputs)
        elif "fc" in  layer_name:
            return self.get_mat_mult(inputs, layer, kwargs.get("quant", True))
        else:
            raise(ValueError("Layer not found"))


    def get_inputs(self, file_name):
        inputs = self.read_input(file_name)
        return torch.mul(torch.tensor(inputs),2**self.scaling).long()

    def get_relu(self, inputs):
        relu_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
        relu_circuit.inputs_1 = inputs
        out = relu_circuit.get_outputs()
        input, output = relu_circuit.get_twos_comp_model_data(out)
        return (input, None, output)
    
    def get_circuit_conv(self, inputs, layer, strides = (1,1), kernel_shape = (3,3), group = [1], dilation = (1,1), pads = (1,1,1,1)):
        

        weights = layer.weight
        bias = layer.bias
        # layers = self.layers[layer_name]
        # conv_circuit = Convolution()
        conv_circuit = QuantizedConv()

        conv_circuit.input_arr = inputs
        conv_circuit.weights = torch.mul(weights, 2**self.scaling).long()
        conv_circuit.bias = torch.mul(bias, 2**(self.scaling*2)).long()
        

        conv_circuit.scaling = self.scaling
        conv_circuit.strides = strides
        conv_circuit.kernel_shape = torch.tensor(kernel_shape)
        conv_circuit.group = torch.tensor(group) 
        conv_circuit.dilation = dilation
        conv_circuit.pads = pads

        return conv_circuit.get_model_params(conv_circuit.get_output())
    
    def get_mat_mult(self, inputs, layer, quant = True):
        weights = layer.weight
        bias = layer.bias

        if quant:
            mat_mult_circuit = QuantizedGemm()
        else:
            mat_mult_circuit = Gemm()

        mat_mult_circuit.matrix_a = inputs.long()
        mat_mult_circuit.matrix_b = torch.transpose(torch.mul(weights, 2**self.scaling),0,1).long()

        # Scale up matrix c, twofold, to account for the multiplication that has just taken place
        mat_mult_circuit.matrix_c = torch.reshape(torch.mul(bias, 2**(self.scaling*2)), [mat_mult_circuit.matrix_a.shape[0],mat_mult_circuit.matrix_b.shape[1]]).long()
        
        mat_mult_circuit.scaling = self.scaling
        mat_mult_circuit.alpha = torch.tensor(1)
        mat_mult_circuit.beta = torch.tensor(1)

        gemm = mat_mult_circuit.get_outputs()
        return mat_mult_circuit.get_model_params(gemm)
    
# TODO CHANGE THIS NESTED STRUCTURE, DONE FOR EASE FOR NOW, BUT IT NEEDS IMPROVEMENT
class ZKModel(Layers, GeneralLayerFunctions):
    def __init__(self):
        raise(NotImplementedError, "Must implement")

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
        

        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_file, output_file = get_files(
                input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, self.name, output_folder, proof_system)

        inputs, weights, output = self.get_model_params()



        # NO NEED TO CHANGE anything below here!
        to_json(inputs, input_file)

        # Write output to json
        outputs = {"output": value for key, value in output.items()}
        # outputs = {"outputs": reshape_out.tolist()}
        to_json(outputs, output_file)
        for (i, w) in enumerate(weights):
            if i == 0:
                to_json(w, weights_file)
            else:
                val = i + 1
                to_json(w, weights_file[:-5] + f"{val}" + weights_file[-5:])


        # ## Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)
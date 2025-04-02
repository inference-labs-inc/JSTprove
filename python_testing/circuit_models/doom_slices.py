import torch
from python_testing.circuit_components.convolution import Convolution, QuantizedConv
from python_testing.circuit_components.gemm import QuantizedGemm

from python_testing.utils.helper_functions import RunType, read_from_json, to_json



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
        
            
            

        # d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, witness_file=f"doom_{name}_witness.txt", circuit_path=f"doom_{name}_circuit.txt")
        d.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"doom_{name}_witness.txt", circuit_path=f"doom_{name}_circuit.txt", write_json = True)

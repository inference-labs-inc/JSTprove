import time
import torch
from python_testing.utils.pytorch_helpers import ZKModel
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import RunType, get_files, to_json, prove_and_verify
import os
from enum import Enum
import sys
from python_testing.circuit_components.circuit_helpers import Circuit
import torch.nn as nn
import torch.nn.functional as F
from python_testing.utils.pytorch_partial_models import Conv2DModel, Conv2DModelReLU



import numpy as np




class Convolution(ZKModel):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, file_name = "quantized_conv.pth", rescale = False, quantize_model = False):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "convolution"

        self.strides = (1,1)
        self.kernel_shape = 3
        self.group = torch.tensor([1])
        self.dilation = (1,1)
        self.pads = (1,1)
        self.bias = True


        self.out = None

        self.required_keys = ["input"]
        self.input_data_file = "doom_data/doom_input.json"


        self.scale_base = 2
        self.scaling = 21

        dim_0 = 1
        dim_1 = 4
        dim_2 = 28
        dim_3 = 28
        out_channels = 16
        self.quantized = False

        self.model_type = Conv2DModel
        self.model_params = {"in_channels": dim_1, "out_channels": out_channels, "kernel_size": self.kernel_shape, "stride": self.strides, 'padding': self.pads, "bias":self.bias}
        self.rescale_config = {"conv": rescale}
        self.input_shape = [dim_0, dim_1, dim_2, dim_3]
        

        # self.input_shape = [self.N_ROWS_A, self.N_COLS_A]
        if not rescale:
            self.quantized = False
        else:
            self.quantized = True
        self.is_relu = False

        
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    def conv_run(self, X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides):
        if dilations is None:
            dilations = [1 for s in X.shape[2:]]
        if kernel_shape is None:
            kernel_shape = W.shape[2:]
        if pads is None:
            pads = [0 for s in X.shape[2:]] * 2
        if strides is None:
            strides = [1 for s in X.shape[2:]]

        if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
            raise ValueError(
                f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}, "
                f"W should be {(W.shape[0], X.shape[1] // group, np.prod(W.shape[1:]) // X.shape[1] * group)}."
        )
        if len(X.shape) == 4:
            sN, sC, sH, sW = X.shape
            # M, C_group, kH, kW = W.shape
            kh, kw = kernel_shape
            sth, stw = strides

            h_out = int(((sH - kh + pads[0] + pads[2]) / sth) + 1)
            w_out = int(((sW - kw + pads[1] + pads[3]) / stw) + 1)

            h0, w0 = pads[0], pads[1]
            oh, ow = -1 * (kh % 2), -1 * (kw % 2)
            bh, bw = -h0, -w0
            eh, ew = h_out * sth, w_out * stw
            res = np.zeros((X.shape[0], W.shape[0], h_out, w_out))  # type: ignore[assignment]
            if B is not None:
                res[:, :, :, :] = B.reshape((1, -1, 1, 1))  # type: ignore

            for n in range(sN):
                for nw in range(W.shape[0]):
                    for c in range(sC):
                        w = W[nw : nw + 1, c : c + 1]
                        for io in range(bh, eh, sth):
                            hr = (io - bh) // sth
                            if hr >= h_out:
                                continue
                            i = io + kh % 2
                            ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
                            for jo in range(bw, ew, stw):
                                wr = (jo - bw) // stw
                                if wr >= w_out:
                                    continue
                                j = jo + kw % 2
                                iw1, iw2 = max(0, j + ow), min(j + ow + kw, sW)
                                img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2]
                                if img.shape != w.shape:
                                    jh1, jh2 = (
                                        max(-oh - i, 0),
                                        min(kh, kh + sH - (i + oh + kh)),
                                    )
                                    jw1, jw2 = (
                                        max(-ow - j, 0),
                                        min(kw, kw + sW - (j + ow + kw)),
                                    )
                                    w_ = w[:1, :1, jh1:jh2, jw1:jw2]
                                    if img.shape != w_.shape:
                                        raise RuntimeError(
                                            f"Unexpected shape {img.shape} != {w_.shape}, oh={oh}, ow={ow}, "
                                            f"i={i}, j={j}, kh={kh}, kw={kw}, sH={sH}, sW={sW}, sth={sth}, stw={stw}."
                                        )
                                    s = np.dot(img.reshape((1, -1)), w_.reshape((-1, 1)))[
                                        0, 0
                                    ]  # (img * w_).sum()
                                else:
                                    s = np.dot(img.reshape((1, -1)), w.reshape((-1, 1)))[
                                        0, 0
                                    ]  # (img * w).sum()
                                res[n, nw, hr, wr] += s  # type: ignore

                                # TODO
                                # THE BIAS NEEDS TO BE SCALED UP AS WELL!!!!!!

            return res
        

    def get_weights(self):
        weights =  super().get_weights()
        weights["quantized"] = self.quantized
        weights["is_relu"] = self.is_relu
        return weights
    
    def get_outputs(self, inputs):
        out = super().get_outputs(inputs)
        # total_out = self.conv_run(self.input_arr, self.weights, self.bias, "NOTSET",self.dilation, self.group, self.kernel_shape,self.pads, self.strides)
        # for i in range(len(out)):  # Iterate over the first dimension
        #     for j in range(len(out[i])):  # Iterate over the second dimension
        #         for k in range(len(out[i][j])):  # Iterate over the third dimension
        #             for l in range(len(out[i][j][k])):  # Iterate over the fourth dimension
        #                 assert abs(total_out[i][j][k][l].long() - out[i][j][k][l]) < 1
        return out




class QuantizedConv(Convolution):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__(rescale=True)

        self.quantized = True

class QuantizedConvRelu(Convolution):
    def __init__(self, file_name = "quantized_conv.pth", rescale = True):
        self.name = "convolution"

        self.strides = (1,1)
        self.kernel_shape = 3
        self.group = torch.tensor([1])
        self.dilation = (1,1)
        self.pads = (1,1)
        self.bias = True


        self.out = None

        self.required_keys = ["input"]
        self.input_data_file = "doom_data/doom_input.json"


        self.scale_base = 2
        self.scaling = 21

        dim_0 = 1
        dim_1 = 4
        dim_2 = 28
        dim_3 = 28
        out_channels = 16
        self.quantized = True
        
        self.model_type = Conv2DModelReLU
        self.model_params = {"in_channels": dim_1, "out_channels": out_channels, "kernel_size": self.kernel_shape, "stride": self.strides, 'padding': self.pads, "bias":self.bias}
        self.rescale_config = {"conv": rescale}
        self.input_shape = [dim_0, dim_1, dim_2, dim_3]
        self.input_shape = [dim_0, dim_1, dim_2, dim_3]
        

        # self.input_shape = [self.N_ROWS_A, self.N_COLS_A]
        if not rescale:
            self.quantized = False
        else:
            self.quantized = True
        self.is_relu = True



if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "analysis"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    #Rework inputs to function
    # test_circuit = Convolution()
    # test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    # test_circuit = QuantizedConv()
    # test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    d = Convolution()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = Convolution()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = Convolution()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

    d = QuantizedConv()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = QuantizedConv()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = QuantizedConv()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)


    d = QuantizedConvRelu()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = QuantizedConvRelu()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = QuantizedConvRelu()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

import time
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from enum import Enum
import sys

import numpy as np

class Convolution():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "convolution"
        
        # Function input generation
        # 4, 28, 28
        # Initial array size
        dim_0 = 1
        dim_1 = 4
        dim_2 = 28
        dim_3 = 28

        self.test = "hi"

        self.input_arr = torch.randint(low=0, high=2**21, size=(dim_0, dim_1, dim_2, dim_3)) 
        self.bias = torch.randint(low=-2**21, high=2**21, size=(16,)) 
        self.weights = torch.randint(low=-2**21, high=2**21, size=(16,4,3,3)) 

        self.strides = (1,1)
        self.kernel_shape = torch.tensor([3,3])
        self.group = torch.tensor([1])
        self.dilation = (1,1)
        self.pads = (1,1,1,1)

        
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
                            # print(hr)
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

            return res
        

    def convolution(input_data, filters):
        in_channels, H, W = input_data.shape
        num_filters, filt_channels, kH, kW = filters.shape
        
        # Calculate padding size for "same" convolution (assuming stride=1)
        pad_h = kH // 2  # For kH = 3, pad_h = 1
        pad_w = kW // 2  # For kW = 3, pad_w = 1

        # Pad the input_data
        padded_input = np.pad(input_data,
                            pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                            mode='constant', constant_values=0)
        print("PADDED INPUT")
        print(padded_input)
        print("FILTERS")
        print(filters)
        
        output = np.zeros((num_filters, H, W))

        # Loop over each filter
        for f in range(num_filters):
            # Loop over each row and column of the "cross-sections of the input"
            for i in range(H):
                for j in range(W):
                    # Extract the window from the padded input
                    window = padded_input[:, i:i + kH, j:j + kW]
                    # Element-wise multiply the window and the filter, then sum the results
                    # output[f, i, j] = np.sum(window * filters[f]) # We'd use this if we didn't have to circuitize
                    # Initialize the output value
                    output_value = 0
                    # Loop over input channels (should match filter channels)
                    for c in range(in_channels):
                        # Loop over the filter height (3x3 kernel)
                        for kh in range(kH):
                            for kw in range(kW):
                                # Multiply corresponding elements and accumulate
                                output_value += window[c][kh][kw] * filters[f][c][kh][kw]

                    output[f][i][j] = output_value
        return output
    

    def base_testing(self, input_folder:str, proof_folder: str, temp_folder: str, weights_folder:str, circuit_folder:str, proof_system: ZKProofSystems, output_folder: str = None):

            # NO NEED TO CHANGE!
            witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_file, output_file = get_files(
                input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, self.name, output_folder, proof_system)
            
            '''
            #######################################################################################################
            #################################### This is the block for changes ####################################
            #######################################################################################################
            '''


            ## Perform calculation here
            pads = (1,1)
            #Ensure that onnx representation matches torch model
            output_onnx = _conv_implementation(self.input_arr, self.weights, self.bias, "NOTSET",self.dilation, self.group, self.kernel_shape, self.pads, self.strides)
            # raise
            total_out = torch.conv2d(self.input_arr, self.weights, self.bias, self.strides, pads, self.dilation, self.group)
            for i in range(len(output_onnx)):  # Iterate over the first dimension
                for j in range(len(output_onnx[i])):  # Iterate over the second dimension
                    for k in range(len(output_onnx[i][j])):  # Iterate over the third dimension
                        for l in range(len(output_onnx[i][j][k])):  # Iterate over the fourth dimension
                            assert abs(total_out[i][j][k][l] - output_onnx[i][j][k][l]) < 0.000000001

            output = self.conv_run(self.input_arr, self.weights, self.bias, "NOTSET",self.dilation, self.group, self.kernel_shape,self.pads, self.strides)
            # for i in range(len(output_onnx)):  # Iterate over the first dimension
            #     for j in range(len(output_onnx[i])):  # Iterate over the second dimension
            #         for k in range(len(output_onnx[i][j])):  # Iterate over the third dimension
            #             for l in range(len(output_onnx[i][j][k])):  # Iterate over the fourth dimension
            #                 assert abs(total_out[i][j][k][l] - output[i][j][k][l]) < 0.000000001

            # matrix_product_ab = torch.conv2d(self.matrix_a, self.matrix_b)

            ## Define inputs and outputs
            # time.sleep(10)

            inputs = {
                'input_arr': self.input_arr.tolist()
                }
            
            weights = {
                'weights': self.weights.tolist(),
                'bias': self.bias.tolist(),
                'strides': self.strides,
                'kernel_shape': self.kernel_shape.tolist(),
                'group': self.group.tolist(),
                'dilation': self.dilation,
                'pads': self.pads,
                'input_shape': self.input_arr.shape
            }
            outputs = {
                'conv_out': output.astype(np.int64).tolist(),
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

            to_json(weights, weights_file)

            ## Run the circuit
            prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)

def _conv_implementation( 
    X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides
):
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
    if group > 1:
        res = []
        td = 0
        mg = W.shape[0] // group
        dw = W.shape[1]

        for b in range(X.shape[0]):
            for g in range(group):
                gx = X[b : b + 1, g * dw : (g + 1) * dw]
                gw = W[g * mg : (g + 1) * mg]
                try:
                    cv = _conv_implementation(
                        gx,
                        gw,
                        None,
                        auto_pad,
                        dilations,
                        1,
                        kernel_shape,
                        pads,
                        strides,
                    )
                except (ValueError, RuntimeError) as e:
                    raise ValueError(
                        f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={g}/{group}, "
                        f"gx.shape={gx.shape}, gw.shape={gw.shape}, auto_pad={auto_pad}, "
                        f"dilations={dilations}, kernel_shape={kernel_shape}, pads={pads}, "
                        f"strides={strides}."
                    ) from e
                if b == 0:
                    td += cv.shape[1]
                res.append((b, cv))

        new_shape = [X.shape[0], *list(res[0][1].shape[1:])]
        new_shape[1] = td
        final = np.zeros(tuple(new_shape), dtype=res[0][1].dtype)
        p = 0
        for b, cv in res:
            final[b : b + 1, p : p + cv.shape[1]] = cv
            p += cv.shape[1]
            if p >= final.shape[1]:
                p = 0
        if B is not None:
            new_shape = [1 for s in final.shape]
            new_shape[1] = B.shape[0]
            b = B.reshape(tuple(new_shape))
            final += b
        return final

    if dilations[0] != 1 or min(dilations) != max(dilations):
        # Let's compute the dilated kernel.
        nd = len(dilations)
        new_kernel_shape = []
        new_shape = list(W.shape[:-nd])
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            new_shape.append(W.shape[di] + (W.shape[di] - 1) * (d - 1))
            new_kernel_shape.append(kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1))
        new_w = np.zeros(tuple(new_shape), dtype=W.dtype)
        indices = [slice(0, new_w.shape[0]), slice(0, new_w.shape[1])]
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            indices.append(slice(0, new_w.shape[di], d))
        new_w[tuple(indices)] = W
        W = new_w
        kernel_shape = new_kernel_shape

    if auto_pad in {"SAME_LOWER", "SAME_UPPER", "VALID"}:
        head = []
        tail = []
        for i in range(len(X.shape) - 2):
            d = X.shape[i]
            target_size = (d + strides[i] - 1) // strides[i]
            pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
            if auto_pad == "SAME_LOWER":
                pad_head = (pad_needed + 1) // 2
            else:
                pad_head = pad_needed // 2
            pad_tail = pad_needed - pad_head
            head.append(pad_head)
            tail.append(pad_tail)
        pads = head + tail

    if len(X.shape) == 3:
        sN, sC, sH = X.shape
        # M, C_group, kH, kW = W.shape
        (kh,) = kernel_shape
        (sth,) = strides

        h_out = int(((sH - kh + pads[0] + pads[1]) / sth) + 1)

        h0 = pads[0]
        oh = -1 * (kh % 2)
        bh = -h0
        eh = h_out * sth
        res = np.zeros((X.shape[0], W.shape[0], h_out))  # type: ignore[assignment]
        if B is not None:
            res[:, :, :] += B.reshape((1, -1, 1))  # type: ignore

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
                        img = X[n : n + 1, c : c + 1, ih1:ih2]
                        if img.shape != w.shape:
                            jh1, jh2 = max(-oh - i, 0), min(kh, kh + sH - (i + oh + kh))
                            w_ = w[:1, :1, jh1:jh2]
                            if img.shape != w_.shape:
                                raise RuntimeError(
                                    f"Unexpected shape {img.shape} != {w_.shape}, oh={oh}, "
                                    f"i={i}, kh={kh}, sH={sH}, sth={sth}."
                                )
                            s = np.dot(img.reshape((1, -1)), w_.reshape((-1, 1)))[
                                0, 0
                            ]  # (img * w_).sum()
                        else:
                            s = np.dot(img.reshape((1, -1)), w.reshape((-1, 1)))[
                                0, 0
                            ]  # (img * w).sum()
                        res[n, nw, hr] += s  # type: ignore

        return res

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

        return res

    if len(X.shape) == 5:
        sN, sC, sH, sW, sZ = X.shape
        kh, kw, kz = kernel_shape
        sth, stw, stz = strides

        h_out = int(((sH - kh + pads[0] + pads[3]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[4]) / stw) + 1)
        z_out = int(((sZ - kz + pads[2] + pads[5]) / stz) + 1)

        h0, w0, z0 = pads[0], pads[1], pads[2]
        oh, ow, oz = -1 * (kh % 2), -1 * (kw % 2), -1 * (kz % 2)
        bh, bw, bz = -h0, -w0, -z0
        eh, ew, ez = h_out * sth, w_out * stw, z_out * stz
        res = np.zeros((X.shape[0], W.shape[0], h_out, w_out, z_out))  # type: ignore[assignment]
        if B is not None:
            res[:, :, :, :, :] = B.reshape((1, -1, 1, 1, 1))  # type: ignore

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
                            for zo in range(bz, ez, stz):
                                zr = (zo - bz) // stz
                                if zr >= z_out:
                                    continue
                                z = zo + kz % 2
                                iz1, iz2 = max(0, z + oz), min(z + oz + kz, sZ)
                                img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2, iz1:iz2]
                                if img.shape != w.shape:
                                    jh1, jh2 = (
                                        max(-oh - i, 0),
                                        min(kh, kh + sH - (i + oh + kh)),
                                    )
                                    jw1, jw2 = (
                                        max(-ow - j, 0),
                                        min(kw, kw + sW - (j + ow + kw)),
                                    )
                                    jz1, jz2 = (
                                        max(-oz - z, 0),
                                        min(kz, kz + sZ - (z + oz + kz)),
                                    )
                                    w_ = w[:1, :1, jh1:jh2, jw1:jw2, jz1:jz2]
                                    if img.shape != w_.shape:
                                        raise RuntimeError(
                                            f"Unexpected shape {img.shape} != {w_.shape}, oh={oh}, ow={ow}, oz={oz}, "
                                            f"i={i}, j={j}, z={z}, kh={kh}, kw={kw}, kz={kz}, "
                                            f"sH={sH}, sW={sW}, sZ={sZ}, sth={sth}, stw={stw}, stz={stz}."
                                        )
                                    s = np.dot(
                                        img.reshape((1, -1)), w_.reshape((-1, 1))
                                    )[0, 0]  # (img * w_).sum()
                                else:
                                    s = np.dot(
                                        img.reshape((1, -1)), w.reshape((-1, 1))
                                    )[0, 0]  # (img * w).sum()
                                res[n, nw, hr, wr, zr] += s  # type: ignore

        return res

    raise RuntimeError(
        f"The convolution for X.shape={X.shape}, W.shape={W.shape}, "
        f"kernel_shape={kernel_shape} is not implemented yet."
    )
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "analysis"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    #Rework inputs to function
    test_circuit = Convolution()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

from python.testing.python_testing.utils.pytorch_helpers import ZKTorchModel
from python.testing.python_testing.utils.run_proofs import ZKProofSystems
from python.testing.python_testing.utils.helper_functions import RunType
from python.testing.python_testing.utils.pytorch_partial_models import MaxPooling2DModel


import numpy as np




class MaxPooling2D(ZKTorchModel):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, file_name = "quantized_maxpool.pth"):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "maxpooling"

        self.strides = 2
        self.kernel_shape = 2

        self.padding = 0 
        self.dilation = 1
        self.return_indeces = False
        self.ceil_mode = False


        self.required_keys = ["input"]
        # self.input_data_file = "doom_data/doom_input.json"

        self.scale_base = 2
        self.scaling = 21

        dim_0 = 1
        dim_1 = 4
        dim_2 = 28
        dim_3 = 28
        # self.quantized = False

        self.model_type = MaxPooling2DModel
        self.model_params = {"kernel_size": self.kernel_shape, "stride": self.strides, 'padding': self.padding, "return_indeces":self.return_indeces, "ceil_mode": self.ceil_mode}
        self.rescale_config = {}
        self.input_shape = [dim_0, dim_1, dim_2, dim_3]
        
        self.is_relu = False

        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    def maxpool2d(self, x, kernel_shape, strides, pads = 0 , dilations = 1, return_indeces = False, ceil_mode = False, auto_pad = "NOTSET", storage_order = 0):
        if pads is None:
            pads = [0 for i in range(len(kernel_shape) * 2)]
        if strides is None:
            strides = [1 for i in range(len(kernel_shape))]
        if dilations is None:
            dilations = [1 for i in range(len(kernel_shape))]

        n_dims = len(kernel_shape)
        new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])

        input_spatial_shape = x.shape[2:]
        output_spatial_shape = [0 for s in input_spatial_shape]
        if ceil_mode:
            for i in range(len(input_spatial_shape)):
                output_spatial_shape[i] = int(
                    np.ceil(
                        (
                            input_spatial_shape[i]
                            + new_pads[i].sum()
                            - ((kernel_shape[i] - 1) * dilations[i] + 1)
                        )
                        / strides[i]
                        + 1
                    )
                )
                need_to_reduce_out_size_in_ceil_mode = (
                    output_spatial_shape[i] - 1
                ) * strides[i] >= input_spatial_shape[i] + new_pads[i][0]
                if need_to_reduce_out_size_in_ceil_mode:
                    output_spatial_shape[i] -= 1
        else:
            for i in range(len(input_spatial_shape)):
                output_spatial_shape[i] = int(
                    np.floor(
                        (
                            input_spatial_shape[i]
                            + new_pads[i].sum()
                            - ((kernel_shape[i] - 1) * dilations[i] + 1)
                        )
                        / strides[i]
                        + 1
                    )
                )

        global_pooling = False

        y_dims = x.shape[:2] + tuple(output_spatial_shape)
        # print(y_dims, output_spatial_shape)
        x = np.array(x, dtype = np.int64)
        y = np.zeros(y_dims, dtype=x.dtype)
        indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
        x_dims = x.shape
        channels = x_dims[1]
        height = x_dims[2]
        width = x_dims[3] if len(kernel_shape) > 1 else 1
        pooled_height = y_dims[2]
        pooled_width = y_dims[3] if len(kernel_shape) > 1 else 1
        total_channels = x_dims[0] * channels
        stride_h = 1 if global_pooling else strides[0]
        stride_w = 1 if global_pooling else strides[1]

        x_step = height * width
        y_step = pooled_height * pooled_width
        dilation_h = dilations[0]
        dilation_w = dilations[1]

        X_data = x.ravel()
        Y_data = y.ravel()


        def iteration(c):  # type: ignore
            x_d = c * x_step  # X_data
            y_d = c * y_step  # Y_data
            for ph in range(pooled_height):
                hstart = ph * stride_h - new_pads[0, 0]
                hend = hstart + kernel_shape[0] * dilation_h
                for pw in range(pooled_width):
                    wstart = pw * stride_w - new_pads[1, 0]
                    wend = wstart + kernel_shape[1] * dilation_w
                    pool_index = ph * pooled_width + pw
                    max_elements = []
                    for h in range(hstart, hend, dilation_h):
                        if h < 0 or h >= height:
                            continue
                        for w in range(wstart, wend, dilation_w):
                            if w < 0 or w >= width:
                                continue
                            input_index = h * width + w
                            if input_index < 0 or input_index > X_data.shape[0]:
                                continue
                            max_elements.append(X_data[x_d + input_index])
                    if len(max_elements) == 0:
                        continue
                    Y_data[y_d + pool_index] = max(max_elements)
        
        for c in range(total_channels):
            iteration(c)
        return Y_data.reshape(y_dims)
        

    def get_weights(self):
        weights = super().get_weights()
        weights["maxpool_is_relu"] = self.is_relu
        return weights
    
    # def get_outputs(self, inputs):
    #     out = super().get_outputs(inputs)
    #     # out_2 = nn.MaxPool2d(self.kernel_shape, self.strides, self.padding, self.dilation, self.return_indeces, self.ceil_mode)(inputs)
    #     # self.check_4d_eq(out, out_2)
    #     # pads = [self.padding, self.padding, self.padding, self.padding]
    #     # kernel = (self.kernel_shape, self.kernel_shape)
    #     # dilation = (self.dilation, self.dilation)
    #     # strides = (self.strides, self.strides)
    #     # out_3 = torch.as_tensor(self.maxpool2d(inputs, kernel, strides, pads, dilation, self.return_indeces, self.ceil_mode))
    #     # self.check_4d_eq(out, out_3)
    #     return out


if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "analysis"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    d = MaxPooling2D()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = MaxPooling2D()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = MaxPooling2D()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

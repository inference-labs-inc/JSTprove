import json
import torch
class GeneralLayerFunctions():
    def read_input(self, file_name = "doom_data/doom_input.json"):
        """Reads the inputs to each layer of the model from text files."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data["input"]
        
    def get_inputs_from_file(self, file_name, is_scaled: bool = False):
        inputs = self.read_input(file_name)
        if is_scaled:
            out =  torch.as_tensor(inputs).long()
        else:
            out =  torch.mul(torch.as_tensor(inputs),self.scale_base**self.scaling).long()


        if hasattr(self, "input_shape"):
            shape = self.input_shape
            if hasattr(self, 'adjust_shape') and callable(getattr(self, 'adjust_shape')):
                shape = self.adjust_shape(shape)

            out = out.reshape(shape)
        return out
    
    def get_inputs(self, file_path:str = None, is_scaled = False):
        if file_path == None:
            return self.create_new_inputs()
        if hasattr(self, "input_shape"):
            return self.get_inputs_from_file(file_path, is_scaled=is_scaled).reshape(self.input_shape)
        else:
            raise NotImplementedError("Must define attribute input_shape")
    
    def create_new_inputs(self):
        # ONNX inputs will be in this form, and require inputs to not be scaled up
        if isinstance(self.input_shape, dict):
            keys = self.input_shape.keys()
            if len(keys) == 1:
                # If unknown dim in batch spot, assume batch size of 1
                input_shape = self.input_shape[list(keys)[0]]
                input_shape[0] = 1 if input_shape[0] < 1 else input_shape[0]
                return self.get_rand_inputs(input_shape)
            inputs = {}
            for key in keys:
                # If unknown dim in batch spot, assume batch size of 1
                input_shape = self.input_shape[keys[key]]
                input_shape[0] = 1 if input_shape[0] < 1 else input_shape[0]
                inputs[key] = self.get_rand_inputs(input_shape)
            return inputs
        
        return torch.mul(self.get_rand_inputs(self.input_shape), self.scale_base**self.scaling).long()
    
    def get_rand_inputs(self, input_shape):
        return torch.rand(input_shape)*2 - 1

    def format_inputs(self, inputs):
        return {"input": inputs.long().tolist()}
    
    def format_outputs(self, outputs):
        if hasattr(self, "scaling") and hasattr(self, "scale_base"):
            # Must change how rescaled_outputs is specified TODO
            return {"output": outputs.long().tolist(), "rescaled_output": torch.div(outputs, self.scale_base**(self.scaling)).tolist()}
        return {"output": outputs.long().tolist()}
    
    def format_inputs_outputs(self, inputs, outputs):
        return self.format_inputs(inputs), self.format_outputs(outputs)
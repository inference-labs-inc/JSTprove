from abc import ABC, abstractmethod

import onnx

class ModelConverter(ABC):
    """
    Abstract base class for AI model conversion, quantization, and I/O operations.

    This class defines the required interface for implementing a model converter
    that can handle:
    - Saving/loading models in various formats
    - Quantizing models
    - Extracting model weights
    - Generating model outputs

    Concrete subclasses must implement all abstract methods to provide
    model-specific conversion logic.
    """

    @abstractmethod
    def save_model(self, file_path: str):
        """ Save the current model to the specified file path.

        Args:
            file_path (str): Path to save the model file.
        """      
        pass
    
    @abstractmethod
    def load_model(self, file_path: str, model_type = None):
        """
        Load a model from a file.

        Args:
            file_path (str): Path to the model file.
            model_type (optional): Optional identifier for the model format/type.
                                   Useful if multiple formats are supported.
        """
        pass

    @abstractmethod
    def save_quantized_model(self, file_path: str):
        """Save the quantized version of the model to the specified file path.

        Args:
            file_path (str): Path to save the quantized model file.
        """        
        pass

    @abstractmethod
    def load_quantized_model(self, file_path: str):
        """Load a quantized model from a file.

        Args:
            file_path (str): Path to the quantized model file.
        """        
        pass
    
    @abstractmethod
    def quantize_model(self, model: onnx.ModelProto, scale_exponent: int, rescale_config: dict = None):
        """Quantize a model with a given scale and optional rescaling configuration.

        Args:
            model (onnx.ModelProto): The model instance to quantize.
            scale_exponent (int): Quantization scale factor.
            rescale_config (dict, optional): Configuration for rescaling layers or weights during quantization.. Defaults to None.
        """
        pass

    # TODO JG suggestion - can maybe make the layers into a factory here, similar to how its done in Rust? Can refactor to this later imo?
    @abstractmethod
    def get_weights(self, flatten: bool = False):
        """Retrieve the model's weights.

        Args:
            flatten (bool, optional): To flatten or not. Defaults to False.
        
        Returns:
            The model's weights in the specified format which can be read by rust backend.
        """    
        pass

    @abstractmethod
    def get_model_and_quantize(self):
        """Retrieve the model and quantize it in a single operation.
        """        
        pass

    @abstractmethod
    def get_outputs(self, inputs):
        """
        Run inference on the given inputs and return model outputs.

        Args:
            inputs: Input data in the format expected by the model.

        Returns:
            Model outputs after processing the inputs.
        """
        pass

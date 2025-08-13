from abc import ABC, abstractmethod

class ModelConverter(ABC):

    @abstractmethod
    def save_model(self, file_path: str):
        pass
    
    @abstractmethod
    def load_model(self, file_path: str, model_type = None):
        pass

    @abstractmethod
    def save_quantized_model(self, file_path: str):
        pass

    @abstractmethod
    def load_quantized_model(self, file_path: str):
        pass
    
    @abstractmethod
    def quantize_model(self, model, scale: int, rescale_config: dict = None):
        pass

    # TODO JG suggestion - can maybe make the layers into a factory here, similar to how its done in Rust? Can refactor to this later imo?
    @abstractmethod
    def get_weights(self, flatten = False):
        pass

    @abstractmethod
    def get_model_and_quantize(self):
        pass

    @abstractmethod
    def get_outputs(self, inputs):
        pass

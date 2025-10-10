import numpy as np

from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class GemmConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Gemm layers"""

    @property
    def layer_name(self) -> str:
        return "Gemm"

    def get_config(self) -> LayerTestConfig:
        rng = np.random.default_rng()
        return LayerTestConfig(
            op_type="Gemm",
            valid_inputs=["input", "gemm_weight", "gemm_bias"],
            valid_attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
            required_initializers={
                "gemm_weight": rng.random((128, 256)),
                "gemm_bias": rng.random(128),
            },
        )

import onnx
from onnx import version_converter

model = onnx.load("network.onnx")
converted_model = version_converter.convert_version(model, 11)
onnx.save(converted_model, "network_op11.onnx")
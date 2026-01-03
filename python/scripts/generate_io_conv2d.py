print("Generating input and output for Conv2D ONNX model...")

import onnxruntime as ort
import torch
import json
import numpy as np

try:
    input_tensor = torch.randn(1, 1, 28, 28)
    input_numpy = input_tensor.numpy()

    session = ort.InferenceSession("/Users/elenapashkova/GravyTesting-Internal/python/models/models_onnx/model_conv2d.onnx")
    print("ONNX model loaded")

    outputs = session.run(None, {"input": input_numpy})
    print("Inference complete")

    with open("input.json", "w") as f:
        json.dump(input_numpy.tolist(), f)
    print("Input saved")

    with open("expected_output.json", "w") as f:
        json.dump(outputs[0].tolist(), f)
    print("Output saved")
except Exception as e:
    print("Error:", e)
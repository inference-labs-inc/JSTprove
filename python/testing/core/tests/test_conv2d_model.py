import onnxruntime as ort
import numpy as np
import json
import os

def test_conv2d_model_output():
    # Paths - using relative paths since files are in the same directory as the test
    onnx_model_path = "python/models/models_onnx/model_conv2d.onnx"
    input_json_path = "input.json"  # File is in the same directory as this test
    expected_output_json_path = "expected_output.json"  # File is in the same directory as this test

    # Get the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build full paths to the JSON files
    input_json_path = os.path.join(test_dir, "input.json")
    expected_output_json_path = os.path.join(test_dir, "expected_output.json")

    # Load input and expected output
    with open(input_json_path, "r") as f:
        input_data = np.array(json.load(f)).astype(np.float32)

    with open(expected_output_json_path, "r") as f:
        expected_output = np.array(json.load(f)).astype(np.float32)

    # Run ONNX inference
    session = ort.InferenceSession(onnx_model_path)
    outputs = session.run(None, {"input": input_data})
    output = outputs[0]

    # Check that output is close enough
    np.testing.assert_allclose(output, expected_output, rtol=1e-3, atol=1e-5)
    print("Test passed!")

if __name__ == "__main__":
    test_conv2d_model_output()

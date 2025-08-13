print("Testing Redundant Layers model...")

import torch
import json
import os
import sys

# Add the models directory to Python path
sys.path.append('/Users/elenapashkova/GravyTesting-Internal/python/models/models_onnx')

try:
    print("Importing RedundantModel...")
    from redundant_layers_model import RedundantModel

    print("Creating model instance...")
    model = RedundantModel()
    model.eval()

    print("Generating dummy input...")
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print("Running model inference...")
    output = model(dummy_input)

    print("Converting tensors to lists for JSON...")
    # Convert tensors to lists for JSON
    input_data = dummy_input.numpy().tolist()
    output_data = output.detach().numpy().tolist()

    print("Creating output directory...")
    # Save input/output
    io_dir = "python/testing/core/input_output_data/model_redundant/"
    os.makedirs(io_dir, exist_ok=True)

    print("Saving input JSON...")
    with open(io_dir + "input.json", "w") as f:
        json.dump({"input": input_data}, f)

    print("Saving output JSON...")
    with open(io_dir + "output.json", "w") as f:
        json.dump({"output": output_data}, f)

    print("Saved input/output JSONs")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Files saved to:", io_dir)

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure redundant_layers_model.py exists and is accessible")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
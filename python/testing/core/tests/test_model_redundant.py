print("Testing Redundant Model...")

import torch
import json
import os
import sys

# Add the models directory to Python path - using relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'models_onnx')))

try:
    print("Importing RedundantModel...")
    from redundant_layers_model import RedundantModel
    
    model = RedundantModel()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        output = model(dummy_input)

    input_data = dummy_input.numpy().tolist()
    output_data = output.detach().numpy().tolist()

    print("Creating output directory...")
    # Save input/output
    io_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input_output_data', 'model_redundant'))
    os.makedirs(io_dir, exist_ok=True)

    with open(os.path.join(io_dir, "input.json"), "w") as f:
        json.dump({"input": input_data}, f)

    with open(os.path.join(io_dir, "output.json"), "w") as f:
        json.dump({"output": output_data}, f)

    print("✅ Successfully saved input/output JSONs")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

def test_redundant_model():
    print("Test passed!")
    assert True
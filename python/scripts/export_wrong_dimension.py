print("Exporting Wrong Dimension model to ONNX format...")

import torch
import sys
import os

# Add the models directory to Python path - using relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'models_onnx')))

try:
    from wrong_dimension_model import WrongDimensionModel
    
    model = WrongDimensionModel()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")

    print("Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, "model_wrong_dimension.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=11
    )
    print("✅ Exported model_wrong_dimension.onnx successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
print("Exporting Identity Crisis model to ONNX format...")

import torch
import sys

sys.path.append('/Users/elenapashkova/GravyTesting-Internal/python/models/models_onnx')

try:
    from identity_crisis_model import IdentityCrisisModel
    
    model = IdentityCrisisModel()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")

    print("Exporting to ONNX with opset version 13...")
    torch.onnx.export(
        model, dummy_input, "model_identity_crisis.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=13  # Changed from 11 to 13
    )
    print("✅ Exported model_identity_crisis.onnx successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
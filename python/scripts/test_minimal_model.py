print("Testing Minimal Model that should work with JSTProve...")

import torch
import torch.nn as nn
import os

class MinimalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        return self.flatten(x)

try:
    model = MinimalModel()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

    print("Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, "model_minimal.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=11
    )
    
    # Move to models directory
    os.rename("model_minimal.onnx", "python/models/models_onnx/model_minimal.onnx")
    print("✅ Exported model_minimal.onnx successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
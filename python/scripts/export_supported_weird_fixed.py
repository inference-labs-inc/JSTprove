print("Creating fixed Supported Weird model with Conv2D instead of Conv1D...")

import torch
import torch.nn as nn
import os

class SupportedWeirdModelFixed(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv layer (keep as is)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        
        # Fix: Change Conv1d to Conv2d
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # Was Conv1d, now Conv2d
        
        # Rest of the architecture (basic operations)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3136, 1)  # 16*14*14 = 3136
        
    def forward(self, x):
        # Input: [1, 1, 28, 28]
        x = torch.relu(self.conv1(x))  # [1, 8, 28, 28]
        x = torch.relu(self.conv2(x))  # [1, 16, 28, 28] - Now using Conv2d
        x = self.pool(x)               # [1, 16, 14, 14]
        x = self.flatten(x)            # [1, 3136]
        x = self.linear(x)             # [1, 1]
        return x

try:
    model = SupportedWeirdModelFixed()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

    print("Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, "model_supported_weird_fixed.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=11
    )
    
    # Move to models directory
    os.rename("model_supported_weird_fixed.onnx", "python/models/models_onnx/model_supported_weird_fixed.onnx")
    print("✅ Exported model_supported_weird_fixed.onnx successfully!")
    
    # Also create the original with Conv1d for comparison
    print("\nCreating original version with Conv1d for comparison...")
    
    class SupportedWeirdModelOriginal(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv1d = nn.Conv1d(8, 16, 3, padding=1)  # The problematic Conv1d
            self.pool = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(3136, 1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))    # [1, 8, 28, 28]
            # Reshape for Conv1d: [1, 8, 784] (flatten spatial dims)
            x = x.view(x.size(0), x.size(1), -1)  # [1, 8, 784]
            x = torch.relu(self.conv1d(x))   # [1, 16, 782] - Conv1d output
            # Reshape back for pooling
            x = x.view(x.size(0), x.size(1), 28, 28)  # Back to [1, 16, 28, 28]
            x = self.pool(x)                 # [1, 16, 14, 14]
            x = self.flatten(x)              # [1, 3136]
            x = self.linear(x)               # [1, 1]
            return x
    
    original_model = SupportedWeirdModelOriginal()
    original_model.eval()
    
    torch.onnx.export(
        original_model, dummy_input, "model_supported_weird_original.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=11
    )
    
    os.rename("model_supported_weird_original.onnx", "python/models/models_onnx/model_supported_weird_original.onnx")
    print("✅ Also exported original version for comparison!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
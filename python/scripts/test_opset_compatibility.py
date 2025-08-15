print("Testing ONNX opset compatibility for Identity Crisis model...")

import torch
import torch.nn as nn
import sys
import os

# Add the models directory to Python path - using relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'models_onnx')))

class MinimalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        return self.flatten(x)

try:
    from identity_crisis_model import IdentityCrisisModel
    
    model = IdentityCrisisModel()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")

    # Test different opset versions
    opset_versions = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    successful_exports = []
    failed_exports = []
    
    for opset in opset_versions:
        try:
            print(f"\n Testing ONNX opset version {opset}...")
            filename = f"model_identity_crisis_opset{opset}.onnx"
            
            torch.onnx.export(
                model, dummy_input, filename,
                input_names=["input"], output_names=["output"],
                opset_version=opset
            )
            print(f"✅ Opset {opset}: SUCCESS")
            successful_exports.append(opset)
            
            # Move to models directory
            os.rename(filename, f"python/models/models_onnx/{filename}")
            
        except Exception as e:
            print(f"❌ Opset {opset}: FAILED - {str(e)[:100]}...")
            failed_exports.append((opset, str(e)))
    
    print(f"\nOPSET COMPATIBILITY RESULTS:")
    print(f"✅ Successful opsets: {successful_exports}")
    print(f"❌ Failed opsets: {[opset for opset, _ in failed_exports]}")
    
    print(f"\nDETAILED FAILURE REASONS:")
    for opset, error in failed_exports:
        print(f"Opset {opset}: {error[:150]}...")
        
    print(f"\nMINIMUM WORKING OPSET: {min(successful_exports) if successful_exports else 'None'}")
    print(f"MAXIMUM TESTED OPSET: {max(successful_exports) if successful_exports else 'None'}")
    
except Exception as e:
    print(f"❌ Model creation/testing error: {e}")
    import traceback
    traceback.print_exc()
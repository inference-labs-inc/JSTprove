print("Creating Simple Identity Crisis Model (without unflatten)...")

import torch
import torch.nn as nn

try:
    class SimpleIdentityCrisisModel(nn.Module):
        def __init__(self):
            super(SimpleIdentityCrisisModel, self).__init__()
            
            # Conflicting normalization layers (but skip InstanceNorm for compatibility)
            self.batch_norm = nn.BatchNorm2d(8)
            self.group_norm = nn.GroupNorm(2, 8)
            
            # Only ReLU (skip LeakyReLU for compatibility)
            self.relu = nn.ReLU()
            
            # Light dropout only
            self.dropout = nn.Dropout(0.5)
            
            # Only max pooling (skip averaging operations)
            self.pool = nn.MaxPool2d(2)
            
            # Core layers
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            
            # Simple flatten
            self.flatten = nn.Flatten()
            
            # Final layers
            self.linear1 = nn.Linear(3136, 64)  # 16*14*14 = 3136
            self.linear2 = nn.Linear(64, 1)
            
        def forward(self, x):
            # Initial convolution
            x = self.conv1(x)  # [1, 8, 28, 28]
            
            # Apply normalizations (last one wins)
            x = self.batch_norm(x)
            x = self.group_norm(x)
            
            # Activation
            x = self.relu(x)
            
            # Dropout
            x = self.dropout(x)
            
            # Second convolution
            x = self.conv2(x)  # [1, 16, 28, 28]
            
            # Simple pooling
            x = self.pool(x)  # [1, 16, 14, 14]
            
            # Flatten and linear layers
            x = self.flatten(x)  # [1, 3136]
            x = self.linear1(x)  # [1, 64]
            x = self.relu(x)
            x = self.linear2(x)  # [1, 1]
            
            return x

    print("SimpleIdentityCrisisModel class created successfully!")
    
except Exception as e:
    print(f"Error creating model: {e}")
    import traceback
    traceback.print_exc()
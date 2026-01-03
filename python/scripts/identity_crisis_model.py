print("Creating Identity Crisis Model...")

import torch
import torch.nn as nn

try:
    class IdentityCrisisModel(nn.Module):
        def __init__(self):
            super(IdentityCrisisModel, self).__init__()
            
            # Conflicting normalization layers
            self.batch_norm = nn.BatchNorm2d(8)
            self.group_norm = nn.GroupNorm(2, 8)
            self.instance_norm = nn.InstanceNorm2d(8)
            
            # Conflicting activation functions
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
            
            # Extreme dropout patterns
            self.light_dropout = nn.Dropout(0.1)
            self.heavy_dropout = nn.Dropout(0.95)
            
            # Conflicting pooling operations
            self.max_pool = nn.MaxPool2d(2)
            self.avg_pool = nn.AvgPool2d(2)
            
            # Core layers
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            
            # Flatten and unflatten (identity crisis about shape)
            self.flatten = nn.Flatten()
            self.unflatten = nn.Unflatten(1, (16, 7, 7))
            
            # Final layers
            self.linear1 = nn.Linear(784, 64)
            self.linear2 = nn.Linear(64, 1)
            
        def forward(self, x):
            # Initial convolution
            x = self.conv1(x)  # [1, 8, 28, 28]
            
            # Apply conflicting normalizations (last one wins)
            x = self.batch_norm(x)
            x = self.group_norm(x)
            x = self.instance_norm(x)
            
            # Apply conflicting activations
            x = self.relu(x)
            x = self.leaky_relu(x)  # ReLU already made it positive
            
            # Apply conflicting dropout patterns
            x = self.light_dropout(x)
            x = self.heavy_dropout(x)  # Heavy dropout dominates
            
            # Second convolution
            x = self.conv2(x)  # [1, 16, 28, 28]
            
            # Apply conflicting pooling and average results
            x1 = self.max_pool(x)  # [1, 16, 14, 14]
            x2 = self.avg_pool(x)  # [1, 16, 14, 14]
            x = (x1 + x2) / 2
            
            # Pool to 7x7
            x = nn.functional.adaptive_avg_pool2d(x, (7, 7))  # [1, 16, 7, 7]
            
            # Flatten then unflatten (the true identity crisis!)
            x = self.flatten(x)  # [1, 784]
            x = self.unflatten(x)  # [1, 16, 7, 7] - back to original shape
            
            # Final flatten for linear layers
            x = self.flatten(x)  # [1, 784]
            
            # Linear layers
            x = self.linear1(x)  # [1, 64]
            x = self.relu(x)
            x = self.linear2(x)  # [1, 1]
            
            return x

    print("IdentityCrisisModel class created successfully!")
    
except Exception as e:
    print(f"Error creating model: {e}")
    import traceback
    traceback.print_exc()
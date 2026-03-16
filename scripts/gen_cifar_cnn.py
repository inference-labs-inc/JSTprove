import os

import torch
import torch.nn as nn


class CifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 60, bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(60, 10, bias=True)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    torch.manual_seed(0)

    model = CifarCNN()
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total}")

    dummy = torch.randn(1, 3, 32, 32)

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "rust",
        "jstprove_remainder",
        "models",
        "cifar_cnn.onnx",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    print(f"Exported to {out_path}")


if __name__ == "__main__":
    main()

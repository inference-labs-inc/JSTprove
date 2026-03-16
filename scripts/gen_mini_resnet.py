import os

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.block1 = ResBlock(8)
        self.block2 = ResBlock(8)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 7 * 7, 10, bias=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def main():
    torch.manual_seed(0)

    model = MiniResNet()
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "rust",
        "jstprove_remainder",
        "models",
        "mini_resnet.onnx",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    print(f"Exported to {out_path}")


if __name__ == "__main__":
    main()

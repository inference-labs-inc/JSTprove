import os

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper


class ClipCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 10, bias=True)

    def forward(self, x):
        x = torch.clamp(self.conv1(x), 0.0, 6.0)
        x = self.pool1(x)
        x = torch.clamp(self.conv2(x), 0.0, 6.0)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def inline_constants(model_path):
    model = onnx.load(model_path)
    const_map = {}
    keep_nodes = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            name = node.output[0]
            for attr in node.attribute:
                if attr.name == "value":
                    const_map[name] = numpy_helper.to_array(attr.t)
        else:
            keep_nodes.append(node)

    for name, arr in const_map.items():
        tensor = numpy_helper.from_array(arr, name=name)
        model.graph.initializer.append(tensor)

    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def main():
    torch.manual_seed(0)

    model = ClipCNN()
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    out_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "rust",
        "jstprove_remainder",
        "models",
        "clip_cnn.onnx",
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

    inline_constants(out_path)
    print(f"Exported to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import onnx

from python.core.model_processing.onnx_quantizer.exceptions import (
    InvalidParamError,
    UnsupportedOpError,
)
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)
from python.frontend.commands.base import BaseCommand


class ModelCheckCommand(BaseCommand):
    """Check if a model is supported for quantization."""

    name: ClassVar[str] = "model_check"
    aliases: ClassVar[list[str]] = ["check"]
    help: ClassVar[str] = "Check if the model is supported for quantization."

    @classmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "pos_model_path",
            nargs="?",
            metavar="model_path",
            help="Path to the ONNX model.",
        )
        parser.add_argument(
            "-m",
            "--model-path",
            help="Path to the ONNX model.",
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        args.model_path = args.model_path or args.pos_model_path

        if not args.model_path:
            raise ValueError("model_check requires model_path")

        cls._ensure_file_exists(args.model_path)

        model = onnx.load(args.model_path)
        quantizer = ONNXOpQuantizer()
        try:
            quantizer.check_model(model)
            print(f"Model {args.model_path} is supported.")
        except UnsupportedOpError as e:
            raise RuntimeError(f"Model {args.model_path} is NOT supported: Unsupported operations {e.unsupported_ops}") from e
        except InvalidParamError as e:
            raise RuntimeError(f"Model {args.model_path} is NOT supported: {e.message}") from e

    @staticmethod
    def _ensure_file_exists(path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
        if not p.exists() or not p.stat().st_mode & 0o444:
            raise PermissionError(f"Cannot read file: {path}")

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.frontend.commands.base import BaseCommand


class ModelCheckCommand(BaseCommand):
    """Check if a model is supported for quantization."""

    name: ClassVar[str] = "model_check"
    aliases: ClassVar[list[str]] = ["check"]
    help: ClassVar[str] = "Check if the model is supported for quantization."

    @classmethod
    def configure_parser(
        cls: type[ModelCheckCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
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
    @BaseCommand.validate_required("model_path")
    @BaseCommand.validate_paths("model_path")
    def run(cls: type[ModelCheckCommand], args: argparse.Namespace) -> None:
        import onnx  # noqa: PLC0415

        from python.core.model_processing.onnx_quantizer.exceptions import (  # noqa: PLC0415
            InvalidParamError,
            UnsupportedOpError,
        )
        from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (  # noqa: PLC0415
            ONNXOpQuantizer,
        )

        model = onnx.load(args.model_path)
        quantizer = ONNXOpQuantizer()
        try:
            quantizer.check_model(model)
            print(f"Model {args.model_path} is supported.")  # noqa: T201
        except UnsupportedOpError as e:
            msg = (
                f"Model {args.model_path} is NOT supported: "
                f"Unsupported operations {e.unsupported_ops}"
            )
            raise RuntimeError(msg) from e
        except InvalidParamError as e:
            msg = f"Model {args.model_path} is NOT supported: {e.message}"
            raise RuntimeError(msg) from e

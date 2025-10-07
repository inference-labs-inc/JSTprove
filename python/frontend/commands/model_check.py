from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.frontend.commands.args import MODEL_PATH
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
        MODEL_PATH.add_to_parser(parser)

    @classmethod
    @BaseCommand.validate_required(MODEL_PATH)
    @BaseCommand.validate_paths(MODEL_PATH)
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

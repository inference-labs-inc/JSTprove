from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.base import BaseCommand


class CompileCommand(BaseCommand):
    """Compile an ONNX model to a circuit."""

    name: ClassVar[str] = "compile"
    aliases: ClassVar[list[str]] = ["comp"]
    help: ClassVar[str] = (
        "Compile a circuit (writes circuit + quantized model + weights)."
    )

    @classmethod
    def configure_parser(
        cls: type[CompileCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "pos_model_path",
            nargs="?",
            metavar="model_path",
            help="Path to the original ONNX model.",
        )
        parser.add_argument(
            "pos_circuit_path",
            nargs="?",
            metavar="circuit_path",
            help="Output path for the compiled circuit (e.g., circuit.txt).",
        )
        parser.add_argument(
            "-m",
            "--model-path",
            help="Path to the original ONNX model.",
        )
        parser.add_argument(
            "-c",
            "--circuit-path",
            help="Output path for the compiled circuit (e.g., circuit.txt).",
        )

    @classmethod
    @BaseCommand.validate_required("model_path", "circuit_path")
    @BaseCommand.validate_paths("model_path")
    @BaseCommand.validate_parent_paths("circuit_path")
    def run(cls: type[CompileCommand], args: argparse.Namespace) -> None:
        model_name_hint = Path(args.model_path).stem
        circuit = cls._build_circuit(model_name_hint)

        circuit.model_file_name = args.model_path
        circuit.onnx_path = args.model_path
        circuit.model_path = args.model_path

        try:
            circuit.base_testing(
                CircuitExecutionConfig(
                    run_type=RunType.COMPILE_CIRCUIT,
                    circuit_path=args.circuit_path,
                    dev_mode=True,
                ),
            )
        except CircuitRunError as e:
            raise RuntimeError(e) from e

        print(f"[compile] done → circuit={args.circuit_path}")  # noqa: T201

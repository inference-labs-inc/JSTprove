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
    def run(cls: type[CompileCommand], args: argparse.Namespace) -> None:
        args.model_path = args.model_path or args.pos_model_path
        args.circuit_path = args.circuit_path or args.pos_circuit_path

        if not args.model_path or not args.circuit_path:
            msg = "compile requires model_path and circuit_path"
            raise ValueError(msg)

        cls._ensure_file_exists(args.model_path)
        cls._ensure_parent_dir(args.circuit_path)

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

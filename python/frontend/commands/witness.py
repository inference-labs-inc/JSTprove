from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.base import BaseCommand


class WitnessCommand(BaseCommand):
    """Generate witness from circuit and inputs."""

    name: ClassVar[str] = "witness"
    aliases: ClassVar[list[str]] = ["wit"]
    help: ClassVar[str] = "Generate witness using a compiled circuit."

    @classmethod
    def configure_parser(
        cls: type[WitnessCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "pos_circuit_path",
            nargs="?",
            metavar="circuit_path",
            help="Path to the compiled circuit.",
        )
        parser.add_argument(
            "pos_input_path",
            nargs="?",
            metavar="input_path",
            help="Path to input JSON.",
        )
        parser.add_argument(
            "pos_output_path",
            nargs="?",
            metavar="output_path",
            help="Path to write model outputs JSON.",
        )
        parser.add_argument(
            "pos_witness_path",
            nargs="?",
            metavar="witness_path",
            help="Path to write witness.",
        )
        parser.add_argument(
            "-c",
            "--circuit-path",
            help="Path to the compiled circuit.",
        )
        parser.add_argument("-i", "--input-path", help="Path to input JSON.")
        parser.add_argument(
            "-o",
            "--output-path",
            help="Path to write model outputs JSON.",
        )
        parser.add_argument(
            "-w",
            "--witness-path",
            help="Path to write witness.",
        )

    @classmethod
    def run(cls: type[WitnessCommand], args: argparse.Namespace) -> None:
        args.circuit_path = args.circuit_path or args.pos_circuit_path
        args.input_path = args.input_path or args.pos_input_path
        args.output_path = args.output_path or args.pos_output_path
        args.witness_path = args.witness_path or args.pos_witness_path

        if not all(
            [args.circuit_path, args.input_path, args.output_path, args.witness_path],
        ):
            msg = (
                "witness requires circuit_path, input_path, output_path, "
                "and witness_path"
            )
            raise ValueError(msg)

        cls._ensure_file_exists(args.circuit_path)
        cls._ensure_file_exists(args.input_path)
        cls._ensure_parent_dir(args.output_path)
        cls._ensure_parent_dir(args.witness_path)

        circuit = cls._build_circuit("cli")

        try:
            circuit.base_testing(
                CircuitExecutionConfig(
                    run_type=RunType.GEN_WITNESS,
                    circuit_path=args.circuit_path,
                    input_file=args.input_path,
                    output_file=args.output_path,
                    witness_file=args.witness_path,
                ),
            )
        except CircuitRunError as e:
            raise RuntimeError(e) from e

        print(  # noqa: T201
            f"[witness] wrote witness → {args.witness_path} "
            f"and outputs → {args.output_path}",
        )

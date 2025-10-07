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
    @BaseCommand.validate_required(
        "circuit_path",
        "input_path",
        "output_path",
        "witness_path",
    )
    @BaseCommand.validate_paths("circuit_path", "input_path")
    @BaseCommand.validate_parent_paths("output_path", "witness_path")
    def run(cls: type[WitnessCommand], args: argparse.Namespace) -> None:
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

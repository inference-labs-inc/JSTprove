from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.core.circuits.errors import CircuitRunError
from python.core.utils.helper_functions import CircuitExecutionConfig, RunType
from python.frontend.commands.args import (
    CIRCUIT_PATH,
    INPUT_PATH,
    OUTPUT_PATH,
    WITNESS_PATH,
)
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
        CIRCUIT_PATH.add_to_parser(parser)
        INPUT_PATH.add_to_parser(parser)
        OUTPUT_PATH.add_to_parser(parser)
        WITNESS_PATH.add_to_parser(parser)

    @classmethod
    @BaseCommand.validate_required(
        CIRCUIT_PATH,
        INPUT_PATH,
        OUTPUT_PATH,
        WITNESS_PATH,
    )
    @BaseCommand.validate_paths(CIRCUIT_PATH, INPUT_PATH)
    @BaseCommand.validate_parent_paths(OUTPUT_PATH, WITNESS_PATH)
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

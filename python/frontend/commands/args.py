"""Argument specifications for CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

from dataclasses import dataclass


@dataclass(frozen=True)
class ArgSpec:
    """Specification for a command-line argument."""

    name: str
    flag: str
    short: str
    help_text: str

    @property
    def positional(self) -> str:
        """Return the positional argument name."""
        return f"pos_{self.name}"

    def add_to_parser(
        self,
        parser: argparse.ArgumentParser,
        help_override: str | None = None,
    ) -> None:
        """Add both positional and flag arguments to the parser."""
        help_text = help_override or self.help_text
        parser.add_argument(
            self.positional,
            nargs="?",
            metavar=self.name,
            help=help_text,
        )
        parser.add_argument(
            self.short,
            self.flag,
            help=help_text,
        )


MODEL_PATH = ArgSpec(
    name="model_path",
    flag="--model-path",
    short="-m",
    help_text="Path to the original ONNX model.",
)

CIRCUIT_PATH = ArgSpec(
    name="circuit_path",
    flag="--circuit-path",
    short="-c",
    help_text="Path to the compiled circuit.",
)

INPUT_PATH = ArgSpec(
    name="input_path",
    flag="--input-path",
    short="-i",
    help_text="Path to input JSON.",
)

OUTPUT_PATH = ArgSpec(
    name="output_path",
    flag="--output-path",
    short="-o",
    help_text="Path to write model outputs JSON.",
)

WITNESS_PATH = ArgSpec(
    name="witness_path",
    flag="--witness-path",
    short="-w",
    help_text="Path to write witness.",
)

PROOF_PATH = ArgSpec(
    name="proof_path",
    flag="--proof-path",
    short="-p",
    help_text="Path to write proof.",
)

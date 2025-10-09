from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.frontend.commands.base import BaseCommand
from python.frontend.commands.bench.list import ListCommand
from python.frontend.commands.bench.model import ModelCommand
from python.frontend.commands.bench.sweep import SweepCommand


class BenchCommand(BaseCommand):
    """Benchmark JSTprove models with various configurations."""

    name: ClassVar[str] = "bench"
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str] = "Benchmark JSTprove models with various configurations."

    SUBCOMMANDS: ClassVar[list[type[BaseCommand]]] = [
        ListCommand,
        ModelCommand,
        SweepCommand,
    ]

    @classmethod
    def configure_parser(
        cls: type[BenchCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        subparsers = parser.add_subparsers(
            dest="bench_subcommand",
            required=True,
            help="Benchmark subcommands",
        )

        for subcommand_cls in cls.SUBCOMMANDS:
            subparser = subparsers.add_parser(
                subcommand_cls.name,
                help=subcommand_cls.help,
                aliases=subcommand_cls.aliases,
            )
            subcommand_cls.configure_parser(subparser)

    @classmethod
    def run(cls: type[BenchCommand], args: argparse.Namespace) -> None:
        for subcommand_cls in cls.SUBCOMMANDS:
            if args.bench_subcommand in [subcommand_cls.name, *subcommand_cls.aliases]:
                subcommand_cls.run(args)
                return

        msg = f"Unknown bench subcommand: {args.bench_subcommand}"
        raise ValueError(msg)

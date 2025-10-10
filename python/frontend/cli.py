# python/frontend/cli.py
"""JSTprove CLI."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.frontend.commands import BaseCommand

from python.frontend.commands import (
    BenchCommand,
    CompileCommand,
    ModelCheckCommand,
    ProveCommand,
    VerifyCommand,
    WitnessCommand,
)
from python.frontend.commands.base import HiddenPositionalHelpFormatter

BANNER_TITLE = r"""
  888888  .d8888b. 88888888888
    "88b d88P  Y88b    888
     888 Y88b.         888
     888  "Y888b.      888  88888b.  888d888 .d88b.  888  888  .d88b.
     888     "Y88b.    888  888 "88b 888P"  d88""88b 888  888 d8P  Y8b
     888       "888    888  888  888 888    888  888 Y88  88P 88888888
     88P Y88b  d88P    888  888 d88P 888    Y88..88P  Y8bd8P  Y8b.
     888  "Y8888P"     888  88888P"  888     "Y88P"    Y88P    "Y8888
   .d88P                    888
 .d88P"                     888
888P"                       888
"""

COMMANDS: list[type[BaseCommand]] = [
    ModelCheckCommand,
    CompileCommand,
    WitnessCommand,
    ProveCommand,
    VerifyCommand,
    BenchCommand,
]


def print_header() -> None:
    """Print the CLI banner (no side-effects at import time)."""
    print(  # noqa: T201
        BANNER_TITLE
        + "\n"
        + "JSTprove — Verifiable ML by Inference Labs\n"
        + "Based on Polyhedra Network's Expander (GKR-based proving system)\n",
    )


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the JSTprove CLI.

    Returns:
      0 on success, 1 on error.
    """
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(
        prog="jst",
        description="ZKML CLI (compile, witness, prove, verify).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner.",
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    command_map = {}
    for command_cls in COMMANDS:
        cmd_parser = subparsers.add_parser(
            command_cls.name,
            aliases=command_cls.aliases,
            help=command_cls.help,
            allow_abbrev=False,
            formatter_class=HiddenPositionalHelpFormatter,
        )
        command_cls.configure_parser(cmd_parser)
        command_map[command_cls.name] = command_cls
        for alias in command_cls.aliases:
            command_map[alias] = command_cls

    args = parser.parse_args(argv)

    if not args.no_banner and not os.environ.get("JSTPROVE_NO_BANNER"):
        print_header()

    try:
        command_cls = command_map[args.cmd]
        command_cls.run(args)
    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        return 1
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

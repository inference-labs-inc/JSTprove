from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar


class BaseCommand(ABC):
    """Base class for CLI commands."""

    name: ClassVar[str]
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str]

    @classmethod
    @abstractmethod
    def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Configure the argument parser for this command."""

    @classmethod
    @abstractmethod
    def run(cls, args: argparse.Namespace) -> None:
        """Execute the command."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse


class BaseCommand(ABC):
    """Base class for CLI commands."""

    name: ClassVar[str]
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str]

    @classmethod
    @abstractmethod
    def configure_parser(
        cls: type[BaseCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        """Configure the argument parser for this command."""

    @classmethod
    @abstractmethod
    def run(cls: type[BaseCommand], args: argparse.Namespace) -> None:
        """Execute the command."""

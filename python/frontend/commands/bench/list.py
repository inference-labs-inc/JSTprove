from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import argparse

from python.frontend.commands.args import ArgSpec
from python.frontend.commands.base import BaseCommand

LIST_MODELS = ArgSpec(
    name="list_models",
    flag="--list-models",
    help_text="List all available circuit models.",
    extra_kwargs={"action": "store_true", "default": False},
)


class ListCommand(BaseCommand):
    """List all available circuit models for benchmarking."""

    name: ClassVar[str] = "list"
    aliases: ClassVar[list[str]] = []
    help: ClassVar[str] = "List all available circuit models for benchmarking."

    @classmethod
    def configure_parser(
        cls: type[ListCommand],
        parser: argparse.ArgumentParser,
    ) -> None:
        LIST_MODELS.add_to_parser(parser)

    @classmethod
    def run(cls: type[ListCommand], args: argparse.Namespace) -> None:  # noqa: ARG003
        from python.core.utils.model_registry import (  # noqa: PLC0415
            list_available_models,
        )

        available_models = list_available_models()
        print("\nAvailable Circuit Models:")  # noqa: T201
        for model in available_models:
            print(f"- {model}")  # noqa: T201

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import argparse

DEFAULT_CIRCUIT_MODULE = "python.core.circuit_models.generic_onnx"
DEFAULT_CIRCUIT_CLASS = "GenericModelONNX"


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

    @staticmethod
    def _ensure_file_exists(path: str) -> None:
        p = Path(path)
        if not p.is_file():
            msg = f"Required file not found: {path}"
            raise FileNotFoundError(msg)
        if not p.stat().st_mode & 0o444:
            msg = f"Cannot read file: {path}"
            raise PermissionError(msg)

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_circuit(model_name_hint: str | None = None) -> Any:  # noqa: ANN401
        mod = importlib.import_module(DEFAULT_CIRCUIT_MODULE)
        try:
            cls = getattr(mod, DEFAULT_CIRCUIT_CLASS)
        except AttributeError as e:
            msg = (
                f"Default circuit class '{DEFAULT_CIRCUIT_CLASS}' "
                f"not found in '{DEFAULT_CIRCUIT_MODULE}'"
            )
            raise RuntimeError(msg) from e

        name = model_name_hint or "cli"

        for attempt in (
            lambda: cls(model_name=name),
            lambda: cls(name=name),
            lambda: cls(name),
            lambda: cls(),
        ):
            try:
                return attempt()
            except TypeError:  # noqa: PERF203
                continue

        msg = f"Could not construct {cls.__name__} with/without name '{name}'"
        raise RuntimeError(msg)

from __future__ import annotations

import argparse
import functools
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from python.frontend.commands.args import ArgSpec
from python.frontend.commands.constants import (
    DEFAULT_CIRCUIT_CLASS,
    DEFAULT_CIRCUIT_MODULE,
)


class HiddenPositionalHelpFormatter(argparse.HelpFormatter):
    def _format_usage(
        self,
        usage: str | None,
        actions: list,
        groups: list,
        prefix: str | None,
    ) -> str:
        filtered_actions = [
            action
            for action in actions
            if not (
                isinstance(action, argparse._StoreAction)  # noqa: SLF001
                and action.dest.startswith("pos_")
            )
        ]
        return super()._format_usage(usage, filtered_actions, groups, prefix)

    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(
            action,
            argparse._StoreAction,  # noqa: SLF001
        ) and action.dest.startswith(
            "pos_",
        ):
            return ""
        return super()._format_action(action)


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
    def validate_required(*required: ArgSpec) -> Callable:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(cls: type[BaseCommand], args: argparse.Namespace) -> None:
                for arg_spec in required:
                    flag_val = getattr(args, arg_spec.name, None)
                    pos_val = getattr(args, arg_spec.positional, None)
                    merged = flag_val if flag_val is not None else pos_val
                    if not merged:
                        msg = f"Missing required argument: {arg_spec.name}"
                        raise ValueError(msg)
                    setattr(args, arg_spec.name, merged)
                return func(cls, args)

            return wrapper

        return decorator

    @staticmethod
    def validate_paths(*paths: ArgSpec) -> Callable:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(cls: type[BaseCommand], args: argparse.Namespace) -> None:
                for arg_spec in paths:
                    cls._ensure_file_exists(getattr(args, arg_spec.name))
                return func(cls, args)

            return wrapper

        return decorator

    @staticmethod
    def validate_parent_paths(*paths: ArgSpec) -> Callable:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(cls: type[BaseCommand], args: argparse.Namespace) -> None:
                for arg_spec in paths:
                    cls._ensure_parent_dir(getattr(args, arg_spec.name))
                return func(cls, args)

            return wrapper

        return decorator

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

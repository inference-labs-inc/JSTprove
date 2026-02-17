import importlib
import pkgutil
from pathlib import Path

package_name = __name__
package_dir = Path(__file__).parent.as_posix()

__all__: list[str] = []

_loaded = False


def load_custom_ops() -> None:
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
        if not is_pkg and (module_name != "custom_helpers"):
            importlib.import_module(f"{package_name}.{module_name}")
            __all__.append(module_name)  # noqa: PYI056
    _loaded = True

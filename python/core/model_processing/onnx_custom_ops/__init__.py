import importlib
import pkgutil
from pathlib import Path

# Get the package name of the current module
package_name = __name__

# Dynamically import all .py files in this package directory (except __init__.py)
package_dir = Path(__file__).parent.as_posix()


__all__: list[str] = []

for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    if not is_pkg and (module_name != "custom_helpers"):
        importlib.import_module(f"{package_name}.{module_name}")
        __all__.append(module_name)  # noqa: PYI056

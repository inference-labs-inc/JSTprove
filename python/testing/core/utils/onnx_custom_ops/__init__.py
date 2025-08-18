import importlib
import pkgutil
import os

# Get the package name of the current module
package_name = __name__

# Dynamically import all .py files in this package directory (except __init__.py)
package_dir = os.path.dirname(__file__)

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    if not is_pkg and (module_name != "custom_helpers"):
        importlib.import_module(f"{package_name}.{module_name}")
        __all__.append(module_name)

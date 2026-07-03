
# from .honey_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


# from .utils import *


# # Expose NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS to parent folder
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


# # Optional: Debug message to confirm loading
# print('\033[36m')
# print('Modules Init Loaded')
# print('\033[0m')


import importlib
from pathlib import Path


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


_MODULE_DIR = Path(__file__).parent

for path in _MODULE_DIR.glob("*.py"):
    if path.name == "__init__.py":
        continue

    module_name = path.stem

    try:
        module = importlib.import_module(f".{module_name}", package=__name__)

        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        print(f"[HoneyWorld] Loaded module: {module_name}")

    except Exception as e:
        print(f"[HoneyWorld] Failed to load module {module_name}: {e}")


print(f"[HoneyWorld] Loaded {len(NODE_CLASS_MAPPINGS)} node classes total")
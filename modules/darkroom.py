from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(
    r"H:\Comfy_UI_V78\ComfyUI\custom_nodes\comfyui-darkroom\data\spectral_luts"
)
MANIFEST_PATH = DATA_DIR / "manifest.json"


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.is_file():
        print(
            f"[Honey] Spectral Film Selector: manifest not found at {MANIFEST_PATH}. "
            "Run Darkroom tools/bake_spectral_luts.py --all to generate LUTs."
        )
        return []

    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Honey] Spectral Film Selector: manifest parse failed: {e}")
        return []

    return data.get("presets", [])


def _build_labels(presets: list[dict]) -> list[str]:
    grouped: dict[str, list[dict]] = {}

    for p in presets:
        grouped.setdefault(p.get("category", "Other"), []).append(p)

    labels: list[str] = []
    category_order = [
        "C41 Still",
        "Cinema",
        "Reversal",
        "Instant",
        "Niche",
        "B&W",
        "Other",
    ]

    seen = set()

    for cat in category_order + sorted(grouped.keys()):
        if cat in seen or cat not in grouped:
            continue

        seen.add(cat)

        for p in sorted(grouped[cat], key=lambda x: x["name"]):
            label = f"{cat} / {p['negative']} -> {p['print']}"
            labels.append(label)

    return labels


_PRESETS = _load_manifest()
SPECTRAL_LABELS = _build_labels(_PRESETS)

if not SPECTRAL_LABELS:
    SPECTRAL_LABELS = ["(no spectral LUTs found - run tools/bake_spectral_luts.py)"]


class HoneySpectralFilmSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    SPECTRAL_LABELS,
                    {
                        "default": SPECTRAL_LABELS[0],
                        "control_after_generate": True,
                        "tooltip": "Spectral Film Stock preset selector with matching label output",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("preset", "label")
    FUNCTION = "select"
    CATEGORY = "Honey/Darkroom"

    def select(self, preset):
        return (preset, preset)


##############################################################################################################################
##############################################################################################################################

import importlib.util
from pathlib import Path


def _load_darkroom_color_stock_names():
    """
    Loads COLOR_STOCK_NAMES from ComfyUI-Darkroom's data/color_stocks.py.
    """
    path = Path(r"H:\Comfy_UI_V78\ComfyUI\custom_nodes\comfyui-darkroom\data\color_stocks.py")

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Darkroom color_stocks.py at:\n{path}"
        )

    spec = importlib.util.spec_from_file_location("darkroom_color_stocks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return list(module.COLOR_STOCK_NAMES)


COLOR_STOCK_NAMES = _load_darkroom_color_stock_names()


class HoneyFilmStockSelector:
    @classmethod
    def INPUT_TYPES(cls):
        default_stock = (
            "Aged / Expired Fuji"
            if "Aged / Expired Fuji" in COLOR_STOCK_NAMES
            else COLOR_STOCK_NAMES[0]
        )

        return {
            "required": {
                "film_stock": (
                    COLOR_STOCK_NAMES,
                    {
                        "default": default_stock,
                        "control_after_generate": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("film_stock", "label")
    FUNCTION = "select"
    CATEGORY = "Honey/Darkroom"

    def select(self, film_stock):
        return (film_stock, film_stock)

##############################################################################################################################
##############################################################################################################################


NODE_CLASS_MAPPINGS = {
    "HoneySpectralFilmSelector": HoneySpectralFilmSelector,
    "HoneyFilmStockSelector": HoneyFilmStockSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneySpectralFilmSelector": "Honey Spectral Film Selector",
    "HoneyFilmStockSelector": "Honey Film Stock Selector",
}
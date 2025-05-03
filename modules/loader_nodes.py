import folder_paths
import sys
import os

from .utils import load_and_save_tags

# Determine the ComfyUI root directory (two levels up from the current file)
current_dir = os.path.dirname(__file__)
comfyui_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Add the ComfyUI root to sys.path if it's not already there
if comfyui_root not in sys.path:
    sys.path.append(comfyui_root)

# Now import the required modules from the comfy package
from comfy import utils, sd


## adding a 'none' option:
class LoraLoaderx:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        # Include "None" as a valid selection
        lora_files = folder_paths.get_filename_list("loras")
        lora_files.insert(0, "None")

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (lora_files, {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "How strongly to modify the diffusion model. This value can be negative."
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "How strongly to modify the CLIP model. This value can be negative."
                }),
            },
            "optional": {
                "override_name": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING",)
    RETURN_NAMES = ("M", "C", "tags", "lora")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "Honey"
    DESCRIPTION = "LoRAs are used to make me pull my hair out."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, override_name=""):
        tags_output = ""

        def load_keywords(lora_name):
            """
            Loads the CivitAI keyword tags for a given LoRA.

            Args:
                lora_name (str): Name of the LoRA file.

            Returns:
                list: List of keyword tags.
            """
            if lora_name == "None":
                return []
            return load_and_save_tags(lora_name, force_fetch=False)

        if override_name.strip():
            lora_name = override_name.strip()

        if lora_name == "None":
            return (model, clip, "", "None")

        tags = load_keywords(lora_name)
        tags_output = ", ".join(tags)

        if strength_model == 0 and strength_clip == 0:
            return (model, clip, tags_output, lora_name)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_filename = os.path.basename(lora_path)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        return (model_lora, clip_lora, tags_output, lora_filename)


class SmLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
            },
            "optional": {
                "override_name": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING","STRING",)
    RETURN_NAMES = ("M", "C", "tags", 'lora')
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "Honey"
    DESCRIPTION = "LoRAs are used to make me pull my hair out."

    def load_lora(self, model, clip, lora_name, strength_model=1.0, strength_clip=1.0, override_name=""):
        tags_output = ""
        def load_keywords(lora_name):
                    """
                    Loads the CivitAI keyword tags for a given LoRA.

                    Args:
                        lora_name (str): Name of the LoRA file.

                    Returns:
                        list: List of keyword tags.
                    """
                    if lora_name == "None":
                        return []
                    return load_and_save_tags(lora_name, force_fetch=False)

        if override_name.strip():
            lora_name = override_name.strip()
        
        tags =load_keywords(lora_name)
        tags_output = ", ".join(tags)

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        # Extract just the file name from the full path.
        lora_filename = os.path.basename(lora_path)
        lora = None
        if self.loaded_lora is not None:            
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)


        model_lora, clip_lora = sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)


        return (model_lora, clip_lora, tags_output, lora_filename)

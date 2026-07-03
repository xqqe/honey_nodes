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
                "model": ("MODEL", {
                    "tooltip": "The diffusion model the LoRA will be applied to."
                }),
                "clip": ("CLIP", {
                    "tooltip": "The CLIP model the LoRA will be applied to."
                }),
                "lora_name": (lora_files, {
                    "tooltip": "The name of the LoRA."
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How strongly to modify the diffusion model. This value can be negative."
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How strongly to modify the CLIP model. This value can be negative."
                }),
            },
            "optional": {
                "extra_tags": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Additional text to append to the tags output."
                }),
                "previous_loras": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Previous LoRA filename string to prepend to this LoRA name output."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING",)
    RETURN_NAMES = ("M", "C", "tags", "lora")

    OUTPUT_TOOLTIPS = (
        "The modified diffusion model.",
        "The modified CLIP model.",
        "The keyword tags, plus any extra tags.",
        "Filename-safe LoRA name chain without .safetensors."
    )

    FUNCTION = "load_lora"

    CATEGORY = "Honey"
    DESCRIPTION = "LoRAs are used to make me pull my hair out."

    def load_lora(
        self,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
        extra_tags="",
        previous_loras=""
    ):
        def load_keywords(lora_name_for_tags):
            """
            Loads the CivitAI keyword tags for a given LoRA.
            """
            if lora_name_for_tags == "None":
                return []
            return load_and_save_tags(lora_name_for_tags, force_fetch=False)

        def strip_safetensors(name):
            """
            Removes only the final .safetensors extension.
            """
            name = name or ""

            if name.lower().endswith(".safetensors"):
                return name[:-len(".safetensors")]

            return name

        def make_filename_safe(name):
            """
            Makes a string safer for use in filenames.

            Keeps letters, numbers, underscores, dashes, and periods.
            Converts spaces and common separators to underscores.
            Removes characters that are invalid/problematic in Windows filenames.
            """
            name = name or ""
            name = name.strip()

            # Normalize common separators into underscores
            for char in [" ", ",", ";", "|", "/", "\\"]:
                name = name.replace(char, "_")

            # Remove characters Windows does not allow in filenames
            for char in ['<', '>', ':', '"', '?', '*']:
                name = name.replace(char, "")

            # Collapse repeated underscores
            while "__" in name:
                name = name.replace("__", "_")

            # Avoid leading/trailing separators
            name = name.strip("_-.")

            return name

        def join_lora_names(previous, current, separator="_"):
            """
            Joins previous LoRA names and current LoRA name for filename use.
            """
            previous = make_filename_safe(previous)
            current = make_filename_safe(current)

            if previous and current:
                return f"{previous}{separator}{current}"
            elif previous:
                return previous
            else:
                return current

        # Comfy can pass None for optional unconnected STRING inputs.
        extra_tags = (extra_tags or "").strip()
        previous_loras = (previous_loras or "").strip()

        if lora_name == "None":
            lora_output_name = make_filename_safe(previous_loras)
            return (model, clip, extra_tags, lora_output_name)

        tags = load_keywords(lora_name)
        tags_output = ", ".join(tags)

        if extra_tags:
            if tags_output:
                tags_output = f"{tags_output}, {extra_tags}"
            else:
                tags_output = extra_tags

        current_lora_name = strip_safetensors(os.path.basename(lora_name))
        lora_output_name = join_lora_names(previous_loras, current_lora_name, separator="_")

        if strength_model == 0 and strength_clip == 0:
            return (model, clip, tags_output, lora_output_name)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = sd.load_lora_for_models(
            model,
            clip,
            lora,
            strength_model,
            strength_clip
        )

        return (model_lora, clip_lora, tags_output, lora_output_name)

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


    modes = ["simple", "advanced"]

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        inputs = {
            "required": {
                "input_mode": (cls.modes,),
                "lora_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
            }
        }

        for i in range(1, 51):
            inputs["required"][f"lora_name_{i}"] = (loras,)
            inputs["required"][f"lora_wt_{i}"] = ("FLOAT", {
                "default": 1.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.01
            })
            inputs["required"][f"model_str_{i}"] = ("FLOAT", {
                "default": 1.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.01
            })
            inputs["required"][f"clip_str_{i}"] = ("FLOAT", {
                "default": 1.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.01
            })

        inputs["optional"] = {
            "tags": ("STRING", {
                "default": "",
                "multiline": True,
                "forceInput": True,
                "tooltip": "Existing tags to prepend/continue."
            }),
            "lora_names": ("STRING", {
                "default": "",
                "forceInput": True,
                "tooltip": "Existing LoRA name string to prepend/continue."
            }),
            "lora_stack": ("LORA_STACK",),
        }

        return inputs

    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK",)
    RETURN_NAMES = ("tags", "lora_names", "lora_stack",)

    OUTPUT_TOOLTIPS = (
        "Combined LoRA keyword tags.",
        "Combined filename-safe LoRA names.",
        "Combined LoRA stack."
    )

    FUNCTION = "lora_stacker"
    CATEGORY = "Honey"
    DESCRIPTION = "Stacks LoRAs while also outputting combined tags and filename-safe LoRA names."

    def lora_stacker(
        self,
        input_mode,
        lora_count,
        tags="",
        lora_names="",
        lora_stack=None,
        **kwargs
    ):
        def strip_safetensors(name):
            """
            Removes only the final .safetensors extension.
            """
            name = name or ""

            if name.lower().endswith(".safetensors"):
                return name[:-len(".safetensors")]

            return name

        def make_filename_safe(name):
            """
            Makes a string safer for use in filenames.

            Keeps letters, numbers, underscores, dashes, and periods.
            Converts spaces and common separators to underscores.
            Removes characters that are invalid/problematic in Windows filenames.
            """
            name = name or ""
            name = name.strip()

            for char in [" ", ",", ";", "|", "/", "\\"]:
                name = name.replace(char, "_")

            for char in ['<', '>', ':', '"', '?', '*']:
                name = name.replace(char, "")

            while "__" in name:
                name = name.replace("__", "_")

            name = name.strip("_-.")
            return name

        def clean_lora_name_for_output(lora_name):
            """
            Converts a LoRA path/name into the short filename stem.

            Example:
            Z/illustration/Coloring_Book.safetensors
            becomes:
            Coloring_Book
            """
            name = os.path.basename(lora_name or "")
            name = strip_safetensors(name)
            name = make_filename_safe(name)
            return name

        def join_nonempty(items, separator):
            """
            Joins only non-empty strings.
            """
            cleaned = []

            for item in items:
                item = (item or "").strip()
                if item:
                    cleaned.append(item)

            return separator.join(cleaned)

        def load_keywords_for_lora(lora_name):
            """
            Loads the CivitAI keyword tags for a given LoRA.
            Requires your existing load_and_save_tags() function.
            """
            if not lora_name or lora_name == "None":
                return []

            loaded_tags = load_and_save_tags(lora_name, force_fetch=False)

            if loaded_tags is None:
                return []

            return loaded_tags

        # Comfy can pass None for unconnected optional STRING inputs.
        tags = (tags or "").strip()
        lora_names = (lora_names or "").strip()

        selected_lora_names = [
            kwargs.get(f"lora_name_{i}")
            for i in range(1, lora_count + 1)
        ]

        selected_lora_names = [
            name for name in selected_lora_names
            if name and name != "None"
        ]

        # Build new stack entries from this node.
        if input_mode == "simple":
            weights = [
                kwargs.get(f"lora_wt_{i}")
                for i in range(1, lora_count + 1)
            ]

            new_loras = []

            for lora_name, lora_weight in zip(
                [kwargs.get(f"lora_name_{i}") for i in range(1, lora_count + 1)],
                weights
            ):
                if lora_name and lora_name != "None":
                    new_loras.append((lora_name, lora_weight, lora_weight))

        else:
            model_strs = [
                kwargs.get(f"model_str_{i}")
                for i in range(1, lora_count + 1)
            ]
            clip_strs = [
                kwargs.get(f"clip_str_{i}")
                for i in range(1, lora_count + 1)
            ]

            new_loras = []

            for lora_name, model_str, clip_str in zip(
                [kwargs.get(f"lora_name_{i}") for i in range(1, lora_count + 1)],
                model_strs,
                clip_strs
            ):
                if lora_name and lora_name != "None":
                    new_loras.append((lora_name, model_str, clip_str))

        # Start with the incoming stack, then append this node's LoRAs.
        combined_loras = []

        if lora_stack is not None:
            combined_loras.extend([
                l for l in lora_stack
                if l and len(l) > 0 and l[0] != "None"
            ])

        combined_loras.extend(new_loras)

        # Build tags for LoRAs selected in this node.
        new_tag_chunks = []

        for lora_name in selected_lora_names:
            lora_tags = load_keywords_for_lora(lora_name)

            if lora_tags:
                new_tag_chunks.append(", ".join(lora_tags))

        new_tags = join_nonempty(new_tag_chunks, ", ")
        combined_tags = join_nonempty([tags, new_tags], ", ")

        # Build filename-safe LoRA name string for LoRAs selected in this node.
        new_lora_name_chunks = [
            clean_lora_name_for_output(lora_name)
            for lora_name in selected_lora_names
        ]

        new_lora_names = join_nonempty(new_lora_name_chunks, "_")
        combined_lora_names = join_nonempty([lora_names, new_lora_names], "_")

        return (combined_tags, combined_lora_names, combined_loras)
    
NODE_CLASS_MAPPINGS = {
    "HoneyLoraLoader": LoraLoaderx,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyLoraLoader": "Honey Lora Loader",
}

import folder_paths
import os
import random

# from utils import *
from .utils import load_and_save_tags, append_lora_name_if_empty


class HoneyPowerLoraTagsFromPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "force_fetch": ("BOOLEAN", {"default": False}),
                "append_loraname_if_empty": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "tags", "lora_names")
    FUNCTION = "extract_tags"
    CATEGORY = "Honey"

    def extract_tags(self, model, force_fetch, append_loraname_if_empty, prompt=None, unique_id=None):
        def clean_lora_name_for_output(lora_name):
            import os

            name = os.path.basename(lora_name or "")

            if name.lower().endswith(".safetensors"):
                name = name[:-len(".safetensors")]

            for char in [" ", ",", ";", "|", "/", "\\"]:
                name = name.replace(char, "_")

            for char in ['<', '>', ':', '"', '?', '*']:
                name = name.replace(char, "")

            while "__" in name:
                name = name.replace("__", "_")

            return name.strip("_-.")

        def find_upstream_node_id(current_node_id, input_name):
            """
            Finds the node connected to current_node_id[input_name].
            In Comfy prompt JSON, links usually appear as [source_node_id, output_index].
            """
            if prompt is None or current_node_id is None:
                return None

            current_node = prompt.get(str(current_node_id))
            if not current_node:
                return None

            inputs = current_node.get("inputs", {})
            link = inputs.get(input_name)

            if isinstance(link, list) and len(link) >= 1:
                return str(link[0])

            return None

        def get_enabled_loras_from_prompt_node(prompt_node):
            """
            Extracts enabled rgthree Power Lora Loader widget values.
            Expected rgthree inputs look like:
              lora_1: {"on": true, "lora": "...", "strength": 1}
            """
            result = []

            if not prompt_node:
                return result

            inputs = prompt_node.get("inputs", {})

            for input_name, value in inputs.items():
                if not input_name.startswith("lora_"):
                    continue

                if not isinstance(value, dict):
                    continue

                if not value.get("on"):
                    continue

                lora_name = value.get("lora")

                if not lora_name or lora_name == "None":
                    continue

                result.append(lora_name)

            return result

        all_tags = []
        lora_name_outputs = []

        source_node_id = find_upstream_node_id(unique_id, "model")

        if source_node_id is not None:
            source_node = prompt.get(str(source_node_id), {})
            lora_names = get_enabled_loras_from_prompt_node(source_node)
        else:
            lora_names = []

        for lora_name in lora_names:
            civitai_tags_list = load_and_save_tags(lora_name, force_fetch)

            if civitai_tags_list is None:
                civitai_tags_list = []

            civitai_tags_list = append_lora_name_if_empty(
                civitai_tags_list,
                lora_name,
                append_loraname_if_empty
            )

            all_tags.extend(civitai_tags_list)

            cleaned_name = clean_lora_name_for_output(lora_name)
            if cleaned_name:
                lora_name_outputs.append(cleaned_name)

        all_tags = sorted(set(all_tags), key=str.lower)

        tags_output = ", ".join(all_tags)
        lora_names_output = "_".join(lora_name_outputs)

        return (model, tags_output, lora_names_output)

class HoneyLoraStackTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_stack": ("LORA_STACK",),
                "force_fetch": ("BOOLEAN", {"default": False}),
                "append_loraname_if_empty": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "override_lora_name": ("STRING", {"forceInput": True}),
                "previous_loras": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("concatenated_tags", "lora_names",)
    FUNCTION = "process_lora_stack"
    CATEGORY = "Honey"

    def process_lora_stack(
        self,
        lora_stack,
        force_fetch,
        append_loraname_if_empty,
        override_lora_name="",
        previous_loras=""
    ):
        """
        Processes a LoRA stack, extracts tags for each LoRA, concatenates all tags
        into a single string, and outputs filename-safe LoRA names.
        """

        def strip_safetensors(name):
            name = name or ""

            if name.lower().endswith(".safetensors"):
                return name[:-len(".safetensors")]

            return name

        def make_filename_safe(name):
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

            return name.strip("_-.")

        def clean_lora_name_for_output(lora_name):
            name = os.path.basename(lora_name or "")
            name = strip_safetensors(name)
            name = make_filename_safe(name)
            return name

        def join_nonempty(items, separator):
            cleaned = []

            for item in items:
                item = (item or "").strip()
                if item:
                    cleaned.append(item)

            return separator.join(cleaned)

        all_tags = []
        lora_name_chunks = []

        override_lora_name = (override_lora_name or "").strip()
        previous_loras = (previous_loras or "").strip()

        for lora in lora_stack:
            lora_name = lora[0]

            if not lora_name or lora_name == "None":
                continue

            tag_lookup_name = override_lora_name if override_lora_name else lora_name

            civitai_tags_list = load_and_save_tags(tag_lookup_name, force_fetch)

            if civitai_tags_list is None:
                civitai_tags_list = []

            civitai_tags_list = append_lora_name_if_empty(
                civitai_tags_list,
                tag_lookup_name,
                append_loraname_if_empty
            )

            all_tags.extend(civitai_tags_list)

            clean_name = clean_lora_name_for_output(lora_name)
            if clean_name:
                lora_name_chunks.append(clean_name)

        # Remove duplicate tags and sort
        all_tags = sorted(set(all_tags), key=str.lower)

        # Match LoraLoaderx formatting
        concatenated_tags = ", ".join(all_tags)

        current_lora_names = join_nonempty(lora_name_chunks, "_")
        lora_names_output = join_nonempty([previous_loras, current_lora_names], "_")

        return (concatenated_tags, lora_names_output)
class ExtractLoRAName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK",),  # Input: LoRA stack
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_string",)
    FUNCTION = "extract_lora_string"

    CATEGORY = "Honey/Utilities"

    def extract_lora_string(self, lora_stack):
        """
        Extracts LoRA names from the given stack and formats them as a string.

        Args:
            lora_stack (list): A list of tuples where the first element is the LoRA name.

        Returns:
            tuple: A single-element tuple containing the formatted LoRA string.
        """
        if not isinstance(lora_stack, list):
            return ("",)  # Return an empty string if the input is invalid

        # Extract LoRA names and format them
        lora_string = "".join(
            f"<lora:{lora[0]}>" for lora in lora_stack if isinstance(lora, tuple) and len(lora) > 0 and lora[0] != "None"
        )
        return (lora_string,)

class Honey_LoRAStackRandom:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                # LoRA 1
                "T1": ("BOOLEAN", {"default": True}),  # Toggle 1
                "L1": (loras,),                        # LoRA name 1
                "W1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 1
                "K1": ("BOOLEAN", {"default": True}),  # Include keyword toggle for L1

                # LoRA 2
                "T2": ("BOOLEAN", {"default": True}),  # Toggle 2
                "L2": (loras,),                        # LoRA name 2
                "W2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 2
                "K2": ("BOOLEAN", {"default": True}),  # Include keyword toggle for L2

                # LoRA 3
                "T3": ("BOOLEAN", {"default": True}),  # Toggle 3
                "L3": (loras,),                        # LoRA name 3
                "W3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 3
                "K3": ("BOOLEAN", {"default": True}),  # Include keyword toggle for L3

                # LoRA 4
                "T4": ("BOOLEAN", {"default": True}),  # Toggle 4
                "L4": (loras,),                        # LoRA name 4
                "W4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 4
                "K4": ("BOOLEAN", {"default": True}),  # Include keyword toggle for L4

                # LoRA 5
                "T5": ("BOOLEAN", {"default": True}),  # Toggle 5
                "L5": (loras,),                        # LoRA name 5
                "W5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 5
                "K5": ("BOOLEAN", {"default": True}),  # Include keyword toggle for L5

                # Random Selection Toggle and Seed
                "random_toggle": ("BOOLEAN", {"default": False}),  # Enable random selection
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  # Random seed
            },
            "optional": {
                "stack": ("LORA_STACK",),  # LoRA stack
            },
        }

    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("stack", "tags", "lora_names")
    FUNCTION = "stacker"
    CATEGORY = "Honey"

    def stacker(
        self,
        T1, L1, W1, K1,
        T2, L2, W2, K2,
        T3, L3, W3, K3,
        T4, L4, W4, K4,
        T5, L5, W5, K5,
        random_toggle, seed, stack=None
    ):
        # Initialize LoRA stack
        lora_list = []

        # Extend with existing stack if provided
        if stack is not None:
            lora_list.extend([l for l in stack if l[0] != "None"])

        # Add LoRAs (including "None" if toggles are on)
        if T1:
            lora_list.append((L1, W1, W1, K1) if L1 != "None" else ("None", 0, 0, False))
        if T2:
            lora_list.append((L2, W2, W2, K2) if L2 != "None" else ("None", 0, 0, False))
        if T3:
            lora_list.append((L3, W3, W3, K3) if L3 != "None" else ("None", 0, 0, False))
        if T4:
            lora_list.append((L4, W4, W4, K4) if L4 != "None" else ("None", 0, 0, False))
        if T5:
            lora_list.append((L5, W5, W5, K5) if L5 != "None" else ("None", 0, 0, False))

        # Random selection logic
        if random_toggle and lora_list:
            random.seed(seed)
            selected_lora = random.choice(lora_list)

            # If the selected LoRA is "None", return an empty stack
            if selected_lora[0] == "None":
                return [], "", ""

            # Otherwise, keep only the selected LoRA
            lora_list = [selected_lora]

        # Extract and concatenate CivitAI tags
        all_tags = []
        lora_names = []
        for lora in lora_list:
            lora_name, _, _, include_keyword = lora
            if lora_name != "None":
                lora_names.append(lora_name)
            if lora_name != "None" and include_keyword:
                # Fetch CivitAI tags for each LoRA
                civitai_tags = load_and_save_tags(lora_name, False)
                all_tags.extend(civitai_tags)

        # Remove duplicates, sort, and concatenate tags
        all_tags = sorted(set(all_tags), key=str.lower)
        tags_output = ", ".join(all_tags)

        # Concatenate LoRA names
        lora_names_output = ", ".join(lora_names)

        return lora_list, tags_output, lora_names_output

    @classmethod
    def IS_CHANGED(cls, random_toggle, seed, **kwargs):
        """
        Ensures re-execution of the node each time the seed or random_toggle changes.
        """
        import hashlib
        m = hashlib.sha256()
        m.update(str(random_toggle).encode('utf-8') + str(seed).encode('utf-8'))
        return m.digest().hex()

class Honey_LoRATags:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                # Single LoRA input
                "L": (loras,),  # LoRA name
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "get_keyword_tags"

    CATEGORY = "Honey"

    def get_keyword_tags(self, L):
        """
        Extracts and concatenates CivitAI keyword tags for the selected LoRA.

        Args:
            L: LoRA name (string).

        Returns:
            tuple: A single-element tuple containing the concatenated keyword tags string.
        """
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

        # Fetch CivitAI keyword tags for the single LoRA
        all_tags = load_keywords(L)

        # Remove duplicates and concatenate tags
        unique_tags = sorted(set(all_tags), key=str.lower)
        tags_output = ", ".join(unique_tags)

        return (tags_output,)

class HoneyTextConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'optional': {
                'LORA_tag': ('STRING', {'default': ''}),
                'LORA_toggle': ('BOOLEAN', {'default': True}),
                'Prefix': ('STRING', {'default': 'analog film photograph'}),
                'Prefix_t': ('BOOLEAN', {'default': True}),
                'Main_Prompt': ('STRING', {'default': 'beautiful woman wearing a sweater'}),
                'Main_t': ('BOOLEAN', {'default': True}),
                'extra': ('STRING', {'default': ''}),
                'extra_t': ('BOOLEAN', {'default': True}),
                'Suffix': ('STRING', {'default': '(8k, RAW photo, best quality, masterpiece:1.2), best quality, sharp focus, (Masterpiece), (Best Quality), extremely detailed, intricate, hyper detailed portrait.'}),
                'Suffix_t': ('BOOLEAN', {'default': True}),
            },
            'required': {
                'delimiter': ('STRING', {'default': ', '})
            }
        }

    RETURN_TYPES = ('STRING', 'STRING')  # Outputs both concatenated text and concatenated text excluding LORA_tag
    FUNCTION = 'concat'
    CATEGORY = 'Honey'

    def concat(
        self, 
        LORA_tag, LORA_toggle, 
        Prefix, Prefix_t, 
        Main_Prompt, Main_t, 
        extra, extra_t, 
        Suffix, Suffix_t, 
        delimiter
    ):
        # Concatenate the strings with toggles and delimiter
        texts = [
            Prefix if Prefix_t and Prefix.strip() else None,
            Main_Prompt if Main_t and Main_Prompt.strip() else None,
            extra if extra_t and extra.strip() else None,
            Suffix if Suffix_t and Suffix.strip() else None,
            LORA_tag if LORA_toggle and LORA_tag.strip() else None,
        ]

        # Filter out None or empty strings
        texts = [t for t in texts if t is not None]

        # Concatenate all text
        result = delimiter.join(texts)

        # Correctly remove only the LORA_tag if it's in the list
        texts_without_lora = [t for t in texts if t != LORA_tag]
        result_without_lora = delimiter.join(texts_without_lora)

        return result, result_without_lora

#---------------------------------------------------------------------------------------------------------------------#
import torch
class HoneyBatchAspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
    
        aspect_ratios = ["custom",
                                  "1:1 square 1024x1024",
                                  "1:1 square 1152x1152",
                                  "3:4 portrait 1152x1536",
                                  "3:4 portrait 896x1152",
                                  "5:8 portrait 832x1216",
                                  "9:16 portrait 768x1344",
                                  "9:21 portrait 640x1536",
                                  "4:3 landscape 1152x896",
                                  "3:2 landscape 1216x832",
                                  "16:9 landscape 1344x768",
                                  "21:9 landscape 1536x640"]
        
        return {
            "required": {
                "width": ("INT", {"default": 1152, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1152, "min": 64, "max": 8192}),
                "aspect_ratio": (aspect_ratios,),
                "swap_dimensions": (["Off", "On"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
            }
        }
    RETURN_TYPES = ("INT", "INT", "INT", "LATENT" )
    RETURN_NAMES = ("width", "height", "batch_size", "empty_latent" )
    FUNCTION = "Aspect_Ratio"
    CATEGORY = "Honey"

    def Aspect_Ratio(self, width, height, aspect_ratio, swap_dimensions, batch_size):
        if aspect_ratio == "1:1 square 1024x1024":
            width, height = 1024, 1024
        elif aspect_ratio == "1:1 square 1152x1152":
            width, height = 1152, 1152
        elif aspect_ratio == "3:4 portrait 1152x1536":
            width, height = 1152, 1536
        elif aspect_ratio == "3:4 portrait 896x1152":
            width, height = 896, 1152
        elif aspect_ratio == "5:8 portrait 832x1216":
            width, height = 832, 1216
        elif aspect_ratio == "9:16 portrait 768x1344":
            width, height = 768, 1344
        elif aspect_ratio == "9:21 portrait 640x1536":
            width, height = 640, 1536
        elif aspect_ratio == "4:3 landscape 1152x896":
            width, height = 1152, 896
        elif aspect_ratio == "3:2 landscape 1216x832":
            width, height = 1216, 832
        elif aspect_ratio == "16:9 landscape 1344x768":
            width, height = 1344, 768
        elif aspect_ratio == "21:9 landscape 1536x640":
            width, height = 1536, 640

        if swap_dimensions == "On":
            width, height = height, width
             
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])

           
        return(width, height, batch_size, {"samples":latent} )  




###############################################################


##############################################################################################################################
##############################################################################################################################





# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "HoneyLoraStackTags": HoneyLoraStackTags,
    'HoneyBatchAspectRatio':HoneyBatchAspectRatio,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyLoraStackTags": "Honey LoRA Stack Tags",
    "HoneyBatchAspectRatio": "Honey Batch AspectRatio",
}


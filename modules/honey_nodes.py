
import folder_paths

import random

# from utils import *
from .utils import load_and_save_tags, append_lora_name_if_empty

class HoneyLoraStackTags:
    @classmethod
    def INPUT_TYPES(s):
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {
            "required": {
                "lora_stack": ("LORA_STACK",),
                "force_fetch": ("BOOLEAN", {"default": False}),
                "append_loraname_if_empty": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "override_lora_name": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("concatenated_tags",)
    FUNCTION = "process_lora_stack"
    CATEGORY = "Honey"

    def process_lora_stack(self, lora_stack, force_fetch, append_loraname_if_empty, override_lora_name=""):
        """
        Processes a LoRA stack, extracts tags for each LoRA, and concatenates all tags into a single list.

        Args:
            lora_stack (list): A list of LoRA tuples, each containing (name, weight1, weight2).
            force_fetch (bool): Whether to force-fetch tags.
            append_loraname_if_empty (bool): Whether to append the LoRA name if tags are empty.
            override_lora_name (str): An override name for the LoRA being processed.

        Returns:
            tuple: A single-element tuple containing the concatenated tags list.
        """
        all_tags = []

        for lora in lora_stack:
            lora_name = lora[0]
            if override_lora_name:
                lora_name = override_lora_name

            # Fetch meta and civitai tags
            # meta_tags_list = sort_tags_by_frequency(get_metadata(lora_name, "loras"))
            civitai_tags_list = load_and_save_tags(lora_name, force_fetch)

            # Append LoRA name if tags are empty
            # meta_tags_list = append_lora_name_if_empty(meta_tags_list, lora_name, append_loraname_if_empty)
            civitai_tags_list = append_lora_name_if_empty(civitai_tags_list, lora_name, append_loraname_if_empty)
            

            # Concatenate tags
            # all_tags.extend(meta_tags_list)
            all_tags.extend(civitai_tags_list)


        # Remove duplicates and sort
        all_tags = sorted(set(all_tags), key=str.lower)
        # Convert to a single space-separated string
        concatenated_tags = " ".join(all_tags)


        return (concatenated_tags,)
    
class Honey_LoRAStack:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        return {
            "required": {
                # LoRA 1
                "T1": ("BOOLEAN", {"default": True}),  # Toggle 1
                "L1": (loras,),                        # LoRA name 1
                "W1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 1

                # LoRA 2
                "T2": ("BOOLEAN", {"default": True}),  # Toggle 2
                "L2": (loras,),                        # LoRA name 2
                "W2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 2

                # LoRA 3
                "T3": ("BOOLEAN", {"default": True}),  # Toggle 3
                "L3": (loras,),                        # LoRA name 3
                "W3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 3

                # LoRA 4
                "T4": ("BOOLEAN", {"default": True}),  # Toggle 4
                "L4": (loras,),                        # LoRA name 4
                "W4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 4

                # LoRA 5
                "T5": ("BOOLEAN", {"default": True}),  # Toggle 5
                "L5": (loras,),                        # LoRA name 5
                "W5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),  # Weight 5
            },
            "optional": {
                "stack": ("LORA_STACK",),  # LoRA stack
            },
        }

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("stack", "tags")
    FUNCTION = "stacker"
    CATEGORY = "Honey"

    def stacker(
        self,
        T1, L1, W1,
        T2, L2, W2,
        T3, L3, W3,
        T4, L4, W4,
        T5, L5, W5,
        stack=None
    ):
        # Initialize LoRA stack
        lora_list = []

        # Extend with existing stack if provided
        if stack is not None:
            lora_list.extend([l for l in stack if l[0] != "None"])

        # Add LoRAs based on toggles
        if T1 and L1 != "None":
            lora_list.append((L1, W1, W1))
        if T2 and L2 != "None":
            lora_list.append((L2, W2, W2))
        if T3 and L3 != "None":
            lora_list.append((L3, W3, W3))
        if T4 and L4 != "None":
            lora_list.append((L4, W4, W4))
        if T5 and L5 != "None":
            lora_list.append((L5, W5, W5))

        # Extract and concatenate CivitAI tags
        all_tags = []
        for lora in lora_list:
            lora_name = lora[0]

            # Fetch CivitAI tags for each LoRA
            civitai_tags = load_and_save_tags(lora_name, False)
            all_tags.extend(civitai_tags)

        # Remove duplicates, sort, and concatenate tags
        all_tags = sorted(set(all_tags), key=str.lower)
        tags_output = ", ".join(all_tags)

        return lora_list, tags_output

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


class Honey_LoRATags3:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                # LoRA 1
                "L1": (loras,),  # LoRA name 1
                # LoRA 2
                "L2": (loras,),  # LoRA name 2
                # LoRA 3
                "L3": (loras,),  # LoRA name 3
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "get_keyword_tags"

    CATEGORY = "Honey"

    def get_keyword_tags(self, L1, L2, L3):
        """
        Extracts and concatenates CivitAI keyword tags for selected LoRAs.

        Args:
            L1, L2, L3: LoRA names (strings).

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

        # Collect LoRA names
        loras = [L1, L2, L3]

        # Fetch CivitAI keyword tags
        all_tags = []
        for lora in loras:
            all_tags.extend(load_keywords(lora))

        # Remove duplicates and concatenate tags
        unique_tags = sorted(set(all_tags), key=str.lower)
        tags_output = ", ".join(unique_tags)

        return (tags_output,)


#---------------------------------------------------------------------------------------------------------------------#
class Honey_AspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([ 
                    "1:1 square 1024x1024",
                    "3:4 portrait 896x1152",
                    "5:8 portrait 832x1216",
                    "9:16 portrait 768x1344",
                    "9:21 portrait 640x1536",
                    "4:3 landscape 1152x896",
                    "3:2 landscape 1216x832",
                    "16:9 landscape 1344x768",
                    "21:9 landscape 1536x640"
                ],),
                "swap_dimensions": (["Off", "On"],),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "Aspect_Ratio"

    CATEGORY = "Honey"

    def Aspect_Ratio(self, aspect_ratio, swap_dimensions):
        # Define aspect ratios
        aspect_ratios = {
            "1:1 square 1024x1024": (1024, 1024),
            "3:4 portrait 896x1152": (896, 1152),
            "5:8 portrait 832x1216": (832, 1216),
            "9:16 portrait 768x1344": (768, 1344),
            "9:21 portrait 640x1536": (640, 1536),
            "4:3 landscape 1152x896": (1152, 896),
            "3:2 landscape 1216x832": (1216, 832),
            "16:9 landscape 1344x768": (1344, 768),
            "21:9 landscape 1536x640": (1536, 640),
        }

        # Get width and height
        width, height = aspect_ratios.get(aspect_ratio, (1024, 1024))

        # Swap dimensions if toggle is On
        if swap_dimensions == "On":
            width, height = height, width

        return width, height

#---------------------------------------------------------------------------------------------------------------------------------------------------#
class HoneyTextConcat2:
    # Takes up to 5 text inputs and concatenates them into a single string
    # with an option for delimiter and toggles for each input

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

    RETURN_TYPES = ('STRING',)
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
            LORA_tag if LORA_toggle and LORA_tag.strip() else None,
            Prefix if Prefix_t and Prefix.strip() else None,
            Main_Prompt if Main_t and Main_Prompt.strip() else None,
            extra if extra_t and extra.strip() else None,
            Suffix if Suffix_t and Suffix.strip() else None,
        ]
        # Filter out None or empty strings
        texts = [t for t in texts if t is not None]

        # Join the enabled texts with the specified delimiter
        result = delimiter.join(texts)
        return (result,)

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
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
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

#---------------------------------------------------------------------------------------------------------------------#



###############################################################


##############################################################################################################################
##############################################################################################################################

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "HoneyLoraStackTags": HoneyLoraStackTags,
    "Honey_LoRAStackRandom":Honey_LoRAStackRandom,
    'Honey_LoRATags':Honey_LoRATags,
    "Honey_LoRAStack": Honey_LoRAStack,
    "HoneyTextConcat":HoneyTextConcat,
    'ExtractLoRAName':ExtractLoRAName,
    'Honey_AspectRatio':Honey_AspectRatio,
    'HoneyBatchAspectRatio':HoneyBatchAspectRatio,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyLoraStackTags": "Honey LoRA Stack Tags",
    "Honey_LoRAStack": "Honey LoRA Stack",
    "Honey_LoRAStackRandom": "Honey LoRA Stack Random",
    'Honey_LoRATags':'Honey LoRATags',
    "HoneyTextConcat":"Honey TextConcat",
    'ExtractLoRAName':'Extract LoRAName',
    'Honey_AspectRatio':'Honey AspectRatio',
    'HoneyBatchAspectRatio':'Honey Batch AspectRatio',
}


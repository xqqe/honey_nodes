import os
import folder_paths

from typing import Union

from nodes import LoraLoader


from .power_prompt_utils import get_lora_by_filename
from .rgutils import FlexibleOptionalInputType, any_type
from .utils import load_and_save_tags
from .log import log_node_warn


NODE_NAME = "Honey Power Lora Loader"


class HoneyPowerLoraLoader:
  """Power LoRA Loader variant that also outputs combined tags and filename-safe LoRA names."""

  NAME = NODE_NAME
  CATEGORY = "Honey"

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {},
      "optional": FlexibleOptionalInputType(type=any_type, data={
        "model": ("MODEL",),
        "clip": ("CLIP",),

        # Optional chain inputs. These can be connected from an earlier node.
        "tags": ("STRING", {
          "default": "",
          "multiline": True,
          "forceInput": True,
        }),
        "lora_names": ("STRING", {
          "default": "",
          "forceInput": True,
        }),
      }),
      "hidden": {},
    }

  RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
  RETURN_NAMES = ("MODEL", "CLIP", "tags", "lora_names")
  FUNCTION = "load_loras"

  def load_loras(self, model=None, clip=None, tags="", lora_names="", **kwargs):
    """Loops over provided LoRAs, applies valid ones, and outputs tags/name strings."""

    tags = (tags or "").strip()
    lora_names = (lora_names or "").strip()

    new_tag_chunks = []
    new_lora_name_chunks = []

    for key, value in kwargs.items():
      key_upper = key.upper()

      if not (
        key_upper.startswith('LORA_')
        and isinstance(value, dict)
        and 'on' in value
        and 'lora' in value
        and 'strength' in value
      ):
        continue

      strength_model = value['strength']

      # If only one strength value is passed, use it for both model and clip.
      # If strengthTwo exists, strength is model strength and strengthTwo is clip strength.
      strength_clip = value['strengthTwo'] if 'strengthTwo' in value else None

      if clip is None:
        if strength_clip is not None and strength_clip != 0:
          log_node_warn(NODE_NAME, 'Received clip strength even though no clip supplied!')
        strength_clip = 0
      else:
        strength_clip = strength_clip if strength_clip is not None else strength_model

      if not value['on']:
        continue

      if strength_model == 0 and strength_clip == 0:
        continue

      lora_display_name = value['lora']

      lora_file = get_lora_by_filename(lora_display_name, log_node=self.NAME)

      if lora_file is None:
        continue

      # Apply LoRA.
      if model is not None:
        model, clip = LoraLoader().load_lora(
          model,
          clip,
          lora_file,
          strength_model,
          strength_clip
        )

      # Add trigger/tag words from rgthree info file.
      lora_tags = self.get_triggers_for_lora(lora_display_name, max_each=None)
      if lora_tags:
        new_tag_chunks.append(", ".join(lora_tags))

      # Add short filename-safe LoRA name.
      clean_name = self.clean_lora_name_for_output(lora_file)
      if clean_name:
        new_lora_name_chunks.append(clean_name)

    new_tags = self.join_nonempty(new_tag_chunks, ", ")
    combined_tags = self.join_nonempty([tags, new_tags], ", ")

    new_lora_names = self.join_nonempty(new_lora_name_chunks, "_")
    combined_lora_names = self.join_nonempty([lora_names, new_lora_names], "_")

    return (model, clip, combined_tags, combined_lora_names)

  @staticmethod
  def strip_safetensors(name):
    """Removes only the final .safetensors extension."""
    name = name or ""

    if name.lower().endswith(".safetensors"):
      return name[:-len(".safetensors")]

    return name

  @staticmethod
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

  @classmethod
  def clean_lora_name_for_output(cls, lora_name):
    """
    Converts a LoRA path/name into the short filename stem.

    Example:
      Z/illustration/Coloring_Book.safetensors
    becomes:
      Coloring_Book
    """
    name = os.path.basename(lora_name or "")
    name = cls.strip_safetensors(name)
    name = cls.make_filename_safe(name)
    return name

  @staticmethod
  def join_nonempty(items, separator):
    """Joins only non-empty strings."""
    cleaned = []

    for item in items:
      item = (item or "").strip()
      if item:
        cleaned.append(item)

    return separator.join(cleaned)


  @classmethod
  def get_triggers_for_lora(cls, lora_name: str, max_each=None):
    """
    Gets tags for a single LoRA using Honey's existing tag cache/fetch function.

    max_each=None means all tags.
    max_each=1 means one tag.
    """
    if not lora_name or lora_name == "None":
      return []

    try:
      tags = load_and_save_tags(lora_name, force_fetch=False)
    except Exception as e:
      print(f"[Honey Power Lora Loader] Could not load tags for {lora_name}: {e}")
      return []

    if not tags:
      return []

    if max_each is not None:
      return tags[:max_each]

    return tags
  @classmethod
  def get_enabled_loras_from_prompt_node(cls,
                                         prompt_node: dict) -> list[dict[str, Union[str, float]]]:
    """Gets enabled LoRAs of a node within a server prompt."""
    result = []

    for name, lora in prompt_node['inputs'].items():
      if name.startswith('lora_') and lora['on']:
        lora_file = get_lora_by_filename(lora['lora'], log_node=cls.NAME)

        if lora_file is not None:
          lora_dict = {
            'name': lora['lora'],
            'strength': lora['strength'],
            'path': folder_paths.get_full_path("loras", lora_file)
          }

          if 'strengthTwo' in lora:
            lora_dict['strength_clip'] = lora['strengthTwo']

          result.append(lora_dict)

    return result

  @classmethod
  def get_enabled_triggers_from_prompt_node(cls, prompt_node: dict, max_each: int = 1):
    """Gets trigger words up to max_each for enabled LoRAs of a node within a server prompt."""
    loras = [l['name'] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
    trained_words = []

    for lora in loras:
      trained_words += cls.get_triggers_for_lora(lora, max_each=max_each)

    return trained_words
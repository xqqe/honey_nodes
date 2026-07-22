from pathlib import Path
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths


class HoneyLoadImageListFromDirectory:
    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("images", "masks", "filenames", "count")
    OUTPUT_IS_LIST = (True, True, True, False)

    FUNCTION = "load_images"
    CATEGORY = "Honey/Image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run when queued
        return float("nan")

    def _natural_key(self, text):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)
        ]

    def _resolve_directory(self, directory):
        directory = str(directory).strip()
        if not directory:
            raise ValueError("Directory input is empty.")

        p = Path(directory)

        if p.is_absolute():
            return p

        candidates = [
            Path(folder_paths.get_input_directory()) / p,
            Path(folder_paths.get_output_directory()) / p,
            Path(os.getcwd()) / p,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def _find_files(self, directory_path, recursive):
        if recursive:
            files = [
                p for p in directory_path.rglob("*")
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        else:
            files = [
                p for p in directory_path.iterdir()
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

        files.sort(key=lambda p: self._natural_key(p.name))
        return files

    def _load_single_image(self, image_path):
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        rgb = img.convert("RGB")
        rgb_np = np.array(rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(rgb_np)[None, ...]   # [1, H, W, 3]

        if "A" in img.getbands():
            alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            # Match ComfyUI Load Image behavior:
            # transparent areas -> 1 in mask
            mask_tensor = 1.0 - torch.from_numpy(alpha)
        else:
            h, w = rgb_np.shape[:2]
            mask_tensor = torch.zeros((h, w), dtype=torch.float32)

        mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

        return image_tensor, mask_tensor

    def load_images(self, directory, recursive):
        directory_path = self._resolve_directory(directory)

        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        files = self._find_files(directory_path, recursive)

        if not files:
            raise ValueError(
                f"No supported image files found in directory: {directory_path}"
            )

        image_list = []
        mask_list = []
        filename_list = []

        for file_path in files:
            image_tensor, mask_tensor = self._load_single_image(file_path)
            image_list.append(image_tensor)
            mask_list.append(mask_tensor)
            filename_list.append(file_path.name)

        count = len(filename_list)

        print(f"[Honey Load Image List] Loaded {count} images from: {directory_path}")

        return (image_list, mask_list, filename_list, count)


NODE_CLASS_MAPPINGS = {
    "HoneyLoadImageListFromDirectory": HoneyLoadImageListFromDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyLoadImageListFromDirectory": "Load Image List From Directory",
}
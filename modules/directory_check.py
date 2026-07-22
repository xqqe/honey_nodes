from pathlib import Path
import os

import folder_paths


class HoneyEnsureDirectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "forceInput": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = (
        "directory_path",
        "already_existed",
        "created",
    )

    FUNCTION = "ensure_directory"
    CATEGORY = "Honey/File"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def ensure_directory(self, directory):
        directory = str(directory).strip().strip('"')

        if not directory:
            raise ValueError("Directory input is empty.")

        requested_path = Path(directory)

        # Absolute paths are used directly.
        # Relative paths are placed under ComfyUI's output directory.
        if requested_path.is_absolute():
            directory_path = requested_path
        else:
            directory_path = (
                Path(folder_paths.get_output_directory())
                / requested_path
            )

        if directory_path.exists():
            if not directory_path.is_dir():
                raise ValueError(
                    f"The path exists, but it is not a directory: "
                    f"{directory_path}"
                )

            already_existed = True
            created = False

        else:
            directory_path.mkdir(parents=True, exist_ok=True)

            already_existed = False
            created = True

        resolved_path = str(directory_path.resolve())

        print(
            f"[Honey Ensure Directory] "
            f"{'Already existed' if already_existed else 'Created'}: "
            f"{resolved_path}"
        )

        return (
            resolved_path,
            already_existed,
            created,
        )


NODE_CLASS_MAPPINGS = {
    "HoneyEnsureDirectory": HoneyEnsureDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyEnsureDirectory": "Ensure Directory Exists",
}
# from pathlib import Path
# import os

# import folder_paths


# class HoneyFilenameContainsExists:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "search_string": (
#                     "STRING",
#                     {
#                         "default": "",
#                         "forceInput": True,
#                     },
#                 ),
#                 "directory": (
#                     "STRING",
#                     {
#                         "default": "",
#                         "multiline": False,
#                     },
#                 ),
#                 "recursive": (
#                     "BOOLEAN",
#                     {
#                         "default": False,
#                     },
#                 ),
#             }
#         }

#     RETURN_TYPES = ("BOOLEAN", "STRING")
#     RETURN_NAMES = ("exists", "matched_filename")
#     FUNCTION = "check_exists"
#     CATEGORY = "Honey/File"

#     def check_exists(self, search_string: str, directory: str, recursive: bool):
#         search_string = str(search_string).strip()
#         directory = str(directory).strip()

#         if not search_string:
#             return (False, "")

#         if not directory:
#             base_dir = Path(folder_paths.get_output_directory())
#         elif os.path.isabs(directory):
#             base_dir = Path(directory)
#         else:
#             base_dir = Path(folder_paths.get_output_directory()) / directory

#         if not base_dir.exists() or not base_dir.is_dir():
#             return (False, "")

#         if recursive:
#             iterator = base_dir.rglob("*")
#         else:
#             iterator = base_dir.iterdir()

#         for path in iterator:
#             if path.is_file() and search_string in path.name:
#                 return (True, path.name)

#         return (False, "")


# NODE_CLASS_MAPPINGS = {
#     "HoneyFilenameContainsExists": HoneyFilenameContainsExists,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "HoneyFilenameContainsExists": "Filename Contains Exists",
# }

from pathlib import Path
import os

import folder_paths


class HoneyFilenameContainsExists:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_strings": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                    },
                ),
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

    INPUT_IS_LIST = True

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("matches", "non_matches")
    OUTPUT_IS_LIST = (True, True)

    FUNCTION = "check_exists"
    CATEGORY = "Honey/File"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _resolve_directory(self, directory):
        # INPUT_IS_LIST makes every input arrive as a list.
        if isinstance(directory, list):
            directory = directory[0] if directory else ""

        directory = str(directory).strip()

        if not directory:
            return Path(folder_paths.get_output_directory())

        path = Path(directory)

        if path.is_absolute():
            return path

        return Path(folder_paths.get_output_directory()) / path

    def check_exists(self, search_strings, directory, recursive):
        base_dir = self._resolve_directory(directory)

        if isinstance(recursive, list):
            recursive = recursive[0] if recursive else False

        recursive = bool(recursive)

        if not base_dir.exists():
            raise ValueError(f"Directory does not exist: {base_dir}")

        if not base_dir.is_dir():
            raise ValueError(f"Path is not a directory: {base_dir}")

        if recursive:
            files = [
                path.name
                for path in base_dir.rglob("*")
                if path.is_file()
            ]
        else:
            files = [
                path.name
                for path in base_dir.iterdir()
                if path.is_file()
            ]

        matches = []
        non_matches = []

        for search_string in search_strings:
            search_string = str(search_string).strip()

            found = bool(search_string) and any(
                search_string in filename
                for filename in files
            )

            matches.append(found)
            non_matches.append(not found)

        print(
            f"[Filename Contains Exists] Checked {len(search_strings)} strings "
            f"against {len(files)} files in: {base_dir}"
        )

        return (matches, non_matches)


NODE_CLASS_MAPPINGS = {
    "HoneyFilenameContainsExists": HoneyFilenameContainsExists,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyFilenameContainsExists": "Filename Contains Exists",
}
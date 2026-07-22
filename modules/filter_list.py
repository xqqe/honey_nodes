class HoneyFilterImageListByBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "filenames": ("STRING",),
                "keep": ("BOOLEAN",),
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = (
        "filtered_images",
        "filtered_masks",
        "filtered_filenames",
        "count",
    )

    OUTPUT_IS_LIST = (True, True, True, False)

    FUNCTION = "filter_lists"
    CATEGORY = "Honey/List"

    def filter_lists(self, images, masks, filenames, keep):
        lengths = {
            "images": len(images),
            "masks": len(masks),
            "filenames": len(filenames),
            "keep": len(keep),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                "All input lists must have the same length. "
                f"Received: {lengths}"
            )

        filtered_images = []
        filtered_masks = []
        filtered_filenames = []

        for image, mask, filename, should_keep in zip(
            images,
            masks,
            filenames,
            keep,
        ):
            if bool(should_keep):
                filtered_images.append(image)
                filtered_masks.append(mask)
                filtered_filenames.append(filename)

        count = len(filtered_filenames)

        print(
            f"[Honey Filter Image List] Kept {count} "
            f"of {len(filenames)} items."
        )

        return (
            filtered_images,
            filtered_masks,
            filtered_filenames,
            count,
        )


NODE_CLASS_MAPPINGS = {
    "HoneyFilterImageListByBoolean": HoneyFilterImageListByBoolean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyFilterImageListByBoolean": "Filter Image List By Boolean",
}
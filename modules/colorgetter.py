import torch

class PadToSquareCornerColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "corner": (["top_left", "top_right", "bottom_left", "bottom_right"], {"default": "top_left"}),
                "sample_size": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pad_to_square"
    CATEGORY = "image"

    def pad_to_square(self, image, corner, sample_size):
        # image shape is [B, H, W, C]
        b, h, w, c = image.shape

        # Since you're using a loop, this should usually be one image,
        # but we'll still handle B cleanly.
        out = []

        for i in range(b):
            img = image[i]  # [H, W, C]
            hh, ww, cc = img.shape
            s = min(sample_size, hh, ww)

            if corner == "top_left":
                patch = img[0:s, 0:s, :]
            elif corner == "top_right":
                patch = img[0:s, ww-s:ww, :]
            elif corner == "bottom_left":
                patch = img[hh-s:hh, 0:s, :]
            elif corner == "bottom_right":
                patch = img[hh-s:hh, ww-s:ww, :]
            else:
                patch = img[0:s, 0:s, :]

            # Average the sampled patch to get the background color
            fill_color = patch.mean(dim=(0, 1))  # [C]

            square_size = max(hh, ww)

            # Create square canvas filled with sampled color
            canvas = torch.ones((square_size, square_size, cc), dtype=img.dtype, device=img.device)
            canvas = canvas * fill_color.view(1, 1, cc)

            # Center original image
            y = (square_size - hh) // 2
            x = (square_size - ww) // 2
            canvas[y:y+hh, x:x+ww, :] = img

            out.append(canvas)

        out = torch.stack(out, dim=0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "PadToSquareCornerColor": PadToSquareCornerColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PadToSquareCornerColor": "Pad To Square (Corner Color)"
}
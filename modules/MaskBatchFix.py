import torch

# fixes the mask output for the load images for loop node; they come out the wrong shape or something


class EnsureMaskBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "fix_mask"
    CATEGORY = "Honey/Mask"

    def fix_mask(self, mask):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(mask).__name__}")

        # Unbatched grayscale mask: [H, W] -> [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Image-style single-channel tensor: [B, H, W, 1] -> [B, H, W]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        if mask.ndim != 3:
            raise ValueError(
                f"Expected MASK shape [H,W], [B,H,W], or [B,H,W,1]; "
                f"received {tuple(mask.shape)}"
            )

        return (mask,)


NODE_CLASS_MAPPINGS = {
    "EnsureMaskBatch": EnsureMaskBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnsureMaskBatch": "Ensure Mask Batch",
}
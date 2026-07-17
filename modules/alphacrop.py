from __future__ import annotations

import torch


class CropImageToAlphaBounds:
    """
    Crop IMAGE inputs based on their alpha channel.

    Designed for images whose background has already been removed.
    Works on RGBA images and returns an IMAGE list so each output can
    have its own dimensions.

    Optional features:
      - add margin around the detected content
      - center the result on a square transparent canvas
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_threshold": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "margin": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "square_pad": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "square_margin": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "crop_images"
    CATEGORY = "Honey/Image"

    @staticmethod
    def _ensure_rgba(img: torch.Tensor) -> torch.Tensor:
        """
        Ensure the input image has 4 channels.
        ComfyUI IMAGE shape is [H, W, C] per item here.
        """
        if img.ndim != 3:
            raise ValueError(
                f"Expected one image with shape [H, W, C], got {tuple(img.shape)}"
            )

        channels = img.shape[2]

        if channels == 4:
            return img

        if channels == 3:
            raise ValueError(
                "This node needs an RGBA image with transparency. "
                "The provided image only has 3 channels (RGB)."
            )

        raise ValueError(
            f"Unsupported channel count: {channels}. Expected 4-channel RGBA."
        )

    @staticmethod
    def _crop_one(
        img: torch.Tensor,
        alpha_threshold: float,
        margin: int,
    ) -> torch.Tensor:
        """
        Crop one RGBA image to the bounds of visible alpha.
        Input shape: [H, W, 4]
        Output shape: [H2, W2, 4]
        """
        img = CropImageToAlphaBounds._ensure_rgba(img)

        alpha = img[:, :, 3]
        ys, xs = torch.where(alpha > alpha_threshold)

        # If nothing is visible, return the original image unchanged.
        if xs.numel() == 0 or ys.numel() == 0:
            return img

        x1 = int(xs.min().item())
        x2 = int(xs.max().item()) + 1
        y1 = int(ys.min().item())
        y2 = int(ys.max().item()) + 1

        if margin > 0:
            height, width = img.shape[:2]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(width, x2 + margin)
            y2 = min(height, y2 + margin)

        return img[y1:y2, x1:x2, :]

    @staticmethod
    def _pad_to_square(
        img: torch.Tensor,
        square_margin: int,
    ) -> torch.Tensor:
        """
        Center one RGBA image on a square transparent canvas.
        No resizing. No cropping.
        Input shape: [H, W, 4]
        Output shape: [S, S, 4]
        """
        img = CropImageToAlphaBounds._ensure_rgba(img)

        height, width, channels = img.shape
        side = max(height, width) + square_margin * 2

        if height == width and square_margin == 0:
            return img

        square = torch.zeros(
            (side, side, channels),
            dtype=img.dtype,
            device=img.device,
        )

        x_offset = (side - width) // 2
        y_offset = (side - height) // 2

        square[
            y_offset:y_offset + height,
            x_offset:x_offset + width,
            :
        ] = img

        return square

    def crop_images(
        self,
        image: torch.Tensor,
        alpha_threshold: float,
        margin: int,
        square_pad: bool,
        square_margin: int,
    ):
        """
        image arrives as a ComfyUI IMAGE batch: [B, H, W, C]
        Returns a Python list of IMAGE tensors, each shaped [1, H, W, C]
        """
        if image.ndim != 4:
            raise ValueError(
                f"Expected IMAGE batch [B, H, W, C], got {tuple(image.shape)}"
            )

        results = []

        for i in range(image.shape[0]):
            img = image[i]

            cropped = self._crop_one(
                img,
                alpha_threshold=alpha_threshold,
                margin=margin,
            )

            if square_pad:
                cropped = self._pad_to_square(
                    cropped,
                    square_margin=square_margin,
                )

            # Return each result as a single-item IMAGE tensor.
            results.append(cropped.unsqueeze(0))

        if not results:
            results.append(
                torch.zeros(
                    (1, 64, 64, 4),
                    dtype=torch.float32,
                    device=image.device,
                )
            )

        return (results,)


NODE_CLASS_MAPPINGS = {
    "CropImageToAlphaBounds": CropImageToAlphaBounds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropImageToAlphaBounds": "Crop Image to Alpha Bounds",
}
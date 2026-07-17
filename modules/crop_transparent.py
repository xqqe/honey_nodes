from __future__ import annotations

import torch
import torch.nn.functional as F


class CropTransparentImageToContent:
    """
    Crop an IMAGE using a MASK (typically from Load Image on a transparent PNG),
    then optionally center the result on a square canvas.

    This is ideal for images whose background was already removed in Photoshop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
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
                        "default": True,
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

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_image", "cropped_mask")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "crop"
    CATEGORY = "Honey/Image"

    @staticmethod
    def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize to [B, H, W].
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.ndim == 4:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.shape[1] == 1:
                mask = mask[:, 0, :, :]
            else:
                raise ValueError(
                    f"Unexpected mask shape: {tuple(mask.shape)}"
                )

        if mask.ndim != 3:
            raise ValueError(
                f"Expected MASK [B,H,W], got {tuple(mask.shape)}"
            )

        return mask.float().clamp(0.0, 1.0)

    @staticmethod
    def _auto_fix_mask_polarity(mask_1hw: torch.Tensor) -> torch.Tensor:
        """
        If the border is mostly white, assume the mask is inverted and flip it.
        Expects shape [1, H, W].
        """
        border_pixels = torch.cat(
            [
                mask_1hw[:, 0, :].reshape(-1),
                mask_1hw[:, -1, :].reshape(-1),
                mask_1hw[:, :, 0].reshape(-1),
                mask_1hw[:, :, -1].reshape(-1),
            ]
        )

        if float(border_pixels.mean().item()) > 0.5:
            return 1.0 - mask_1hw

        return mask_1hw

    @staticmethod
    def _pad_to_square(
        image: torch.Tensor,
        mask: torch.Tensor,
        square_margin: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Center one image + mask on a square canvas.
        image: [1,H,W,C]
        mask:  [1,H,W]
        """
        height = image.shape[1]
        width = image.shape[2]
        channels = image.shape[3]

        side = max(height, width) + square_margin * 2

        if height == width and square_margin == 0:
            return image, mask

        square_image = torch.zeros(
            (1, side, side, channels),
            dtype=image.dtype,
            device=image.device,
        )

        square_mask = torch.zeros(
            (1, side, side),
            dtype=mask.dtype,
            device=mask.device,
        )

        x_offset = (side - width) // 2
        y_offset = (side - height) // 2

        square_image[
            :,
            y_offset:y_offset + height,
            x_offset:x_offset + width,
            :
        ] = image

        square_mask[
            :,
            y_offset:y_offset + height,
            x_offset:x_offset + width
        ] = mask

        return square_image, square_mask

    def crop(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_threshold: float,
        margin: int,
        square_pad: bool,
        square_margin: int,
    ):
        if image.ndim != 4:
            raise ValueError(
                f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}"
            )

        mask = self._normalize_mask(mask)

        image_batch = image.shape[0]
        mask_batch = mask.shape[0]
        output_count = max(image_batch, mask_batch)

        images_out: list[torch.Tensor] = []
        masks_out: list[torch.Tensor] = []

        for index in range(output_count):
            image_index = min(index, image_batch - 1)
            mask_index = min(index, mask_batch - 1)

            current_image = image[
                image_index:image_index + 1
            ]

            current_mask = mask[
                mask_index:mask_index + 1
            ]

            current_mask = self._auto_fix_mask_polarity(
                current_mask
            )

            image_height = current_image.shape[1]
            image_width = current_image.shape[2]

            if (
                current_mask.shape[1] != image_height
                or current_mask.shape[2] != image_width
            ):
                current_mask = F.interpolate(
                    current_mask.unsqueeze(1),
                    size=(image_height, image_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            ys, xs = torch.where(
                current_mask[0] > mask_threshold
            )

            if xs.numel() == 0 or ys.numel() == 0:
                cropped_image = current_image
                cropped_mask = current_mask
            else:
                x1 = int(xs.min().item())
                x2 = int(xs.max().item()) + 1
                y1 = int(ys.min().item())
                y2 = int(ys.max().item()) + 1

                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image_width, x2 + margin)
                y2 = min(image_height, y2 + margin)

                cropped_image = current_image[
                    :,
                    y1:y2,
                    x1:x2,
                    :
                ]

                cropped_mask = current_mask[
                    :,
                    y1:y2,
                    x1:x2
                ]

            if square_pad:
                cropped_image, cropped_mask = self._pad_to_square(
                    cropped_image,
                    cropped_mask,
                    square_margin=square_margin,
                )

            images_out.append(cropped_image)
            masks_out.append(cropped_mask)

        return images_out, masks_out


NODE_CLASS_MAPPINGS = {
    "CropTransparentImageToContent": CropTransparentImageToContent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropTransparentImageToContent": "Crop Transparent Image to Content",
}
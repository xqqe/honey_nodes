from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SEGSToMaskedImageList:
    """
    Convert Impact Pack SEGS into cropped images.

    Features:
      - uses each SEG's cropped image and cropped mask
      - optionally trims transparent/empty border
      - optionally centers result on a square canvas
      - can output either:
          * transparent background
          * white background
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "trim_to_alpha": (
                    "BOOLEAN",
                    {
                        "default": True,
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
                "alpha_threshold": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "background_mode": (
                    ["transparent", "white"],
                    {
                        "default": "transparent",
                    },
                ),
            },
            "optional": {
                "fallback_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("masked_images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "Honey/SEGS"

    @staticmethod
    def _crop_image_from_source(
        image: torch.Tensor,
        crop_region: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Crop a ComfyUI IMAGE tensor using an Impact Pack crop region.

        ComfyUI IMAGE shape:
            [batch, height, width, channels]

        crop_region:
            (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = crop_region
        return image[:, y1:y2, x1:x2, :]

    @staticmethod
    def _normalize_image(
        image: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.float()

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.ndim != 4:
            raise ValueError(
                f"Unexpected image shape: {tuple(image.shape)}"
            )

        return image

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """
        Normalize mask to shape [B, H, W].
        """
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.ndim == 4:
            # Handle [B,H,W,1] or [B,1,H,W]
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.shape[1] == 1:
                mask = mask[:, 0, :, :]
            else:
                raise ValueError(
                    f"Unexpected 4D mask shape: {tuple(mask.shape)}"
                )

        if mask.ndim != 3:
            raise ValueError(
                f"Unexpected mask shape: {tuple(mask.shape)}"
            )

        # If somehow there are multiple masks, combine them.
        if mask.shape[0] > 1:
            mask = torch.amax(mask, dim=0, keepdim=True)

        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _match_mask_to_image(
        mask: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Resize the mask only if it does not match the cropped image size.
        """
        target_height = image.shape[1]
        target_width = image.shape[2]

        if (
            mask.shape[1] != target_height
            or mask.shape[2] != target_width
        ):
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return mask

    @staticmethod
    def _make_rgba(
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine RGB image + mask into RGBA.
        """
        mask = mask.to(
            device=image.device,
            dtype=image.dtype,
        )

        alpha = mask.unsqueeze(-1)

        if alpha.shape[0] != image.shape[0]:
            alpha = alpha.expand(
                image.shape[0],
                -1,
                -1,
                -1,
            )

        rgb = image[..., :3]

        return torch.cat(
            [
                rgb,
                alpha,
            ],
            dim=-1,
        )

    @staticmethod
    def _trim_to_alpha_bounds_one(
        rgba: torch.Tensor,
        alpha_threshold: float,
    ) -> torch.Tensor:
        """
        Trim one RGBA image to the bounding box of visible alpha.
        Input shape: [1, H, W, 4]
        """
        if rgba.ndim != 4 or rgba.shape[0] != 1:
            raise ValueError(
                "_trim_to_alpha_bounds_one expects [1, H, W, 4], got "
                f"{tuple(rgba.shape)}"
            )

        alpha = rgba[0, :, :, 3]
        ys, xs = torch.where(alpha > alpha_threshold)

        if xs.numel() == 0 or ys.numel() == 0:
            return rgba

        x1 = int(xs.min().item())
        x2 = int(xs.max().item()) + 1
        y1 = int(ys.min().item())
        y2 = int(ys.max().item()) + 1

        return rgba[:, y1:y2, x1:x2, :]

    @staticmethod
    def _pad_to_square_one(
        rgba: torch.Tensor,
        square_margin: int,
    ) -> torch.Tensor:
        """
        Center one RGBA image on a transparent square canvas.
        No resizing.
        Input shape: [1, H, W, 4]
        """
        if rgba.ndim != 4 or rgba.shape[0] != 1:
            raise ValueError(
                "_pad_to_square_one expects [1, H, W, 4], got "
                f"{tuple(rgba.shape)}"
            )

        height = rgba.shape[1]
        width = rgba.shape[2]
        channels = rgba.shape[3]

        side = max(height, width) + square_margin * 2

        if height == width and square_margin == 0:
            return rgba

        square = torch.zeros(
            (
                1,
                side,
                side,
                channels,
            ),
            dtype=rgba.dtype,
            device=rgba.device,
        )

        x_offset = (side - width) // 2
        y_offset = (side - height) // 2

        square[
            :,
            y_offset:y_offset + height,
            x_offset:x_offset + width,
            :,
        ] = rgba

        return square

    @staticmethod
    def _apply_background_mode_one(
        rgba: torch.Tensor,
        background_mode: str,
    ) -> torch.Tensor:
        """
        Convert one RGBA image to either:
          - transparent RGBA
          - RGB flattened on white
        Input shape: [1, H, W, 4]
        Output shape:
          - [1, H, W, 4] for transparent
          - [1, H, W, 3] for white
        """
        if rgba.ndim != 4 or rgba.shape[0] != 1 or rgba.shape[-1] != 4:
            raise ValueError(
                "_apply_background_mode_one expects [1, H, W, 4], got "
                f"{tuple(rgba.shape)}"
            )

        if background_mode == "transparent":
            return rgba

        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4].clamp(0.0, 1.0)

        if background_mode == "white":
            bg = torch.ones_like(rgb)
        else:
            raise ValueError(
                f"Unsupported background_mode: {background_mode}"
            )

        flattened = rgb * alpha + bg * (1.0 - alpha)
        return flattened

    def convert(
        self,
        segs,
        trim_to_alpha: bool,
        square_pad: bool,
        square_margin: int,
        alpha_threshold: float,
        background_mode: str,
        fallback_image: torch.Tensor | None = None,
    ):
        results: list[torch.Tensor] = []

        # Impact Pack SEGS structure is typically: (header, seg_list)
        for seg in segs[1]:
            cropped_image = seg.cropped_image

            if cropped_image is not None:
                cropped_image = self._normalize_image(
                    cropped_image
                )
            elif fallback_image is not None:
                cropped_image = self._crop_image_from_source(
                    fallback_image,
                    seg.crop_region,
                ).clone()
                cropped_image = self._normalize_image(
                    cropped_image
                )
            else:
                continue

            mask = self._normalize_mask(
                seg.cropped_mask
            )

            mask = self._match_mask_to_image(
                mask,
                cropped_image,
            )

            rgba = self._make_rgba(
                cropped_image,
                mask,
            )

            # Split accidental batches so each output list item is one image.
            for batch_index in range(rgba.shape[0]):
                single_rgba = rgba[
                    batch_index:batch_index + 1
                ]

                if trim_to_alpha:
                    single_rgba = self._trim_to_alpha_bounds_one(
                        single_rgba,
                        alpha_threshold=alpha_threshold,
                    )

                if square_pad:
                    single_rgba = self._pad_to_square_one(
                        single_rgba,
                        square_margin=square_margin,
                    )

                single_output = self._apply_background_mode_one(
                    single_rgba,
                    background_mode=background_mode,
                )

                results.append(single_output)

        if not results:
            # Fallback empty image
            if background_mode == "transparent":
                results.append(
                    torch.zeros(
                        (1, 64, 64, 4),
                        dtype=torch.float32,
                    )
                )
            else:
                results.append(
                    torch.ones(
                        (1, 64, 64, 3),
                        dtype=torch.float32,
                    )
                )

        return (results,)


NODE_CLASS_MAPPINGS = {
    "SEGSToMaskedImageList": SEGSToMaskedImageList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SEGSToMaskedImageList": "SEGS to Masked Centered Square Images",
}
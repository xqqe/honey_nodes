import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class HoneyTextBanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "Hello world",
                        "multiline": True,
                        "forceInput": True,
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "font_size": (
                    "INT",
                    {
                        "default": 72,
                        "min": 8,
                        "max": 512,
                        "step": 1,
                    },
                ),
                "margin": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                    },
                ),
                "wrap_text": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "auto_fit": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "bold_strength": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "make_banner"
    CATEGORY = "Honey/Text"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # rerender every run
        return float("nan")

    def _get_font(self, size):
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\seguisb.ttf",
            "arialbd.ttf",
            "DejaVuSans-Bold.ttf",
        ]

        for path in candidates:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue

        # last resort
        return ImageFont.load_default()

    def _measure_text(self, draw, text, font, spacing):
        if not text:
            text = " "
        bbox = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            spacing=spacing,
            align="center",
        )
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h, bbox

    def _wrap_paragraph(self, draw, paragraph, font, max_width, spacing):
        words = paragraph.split()
        if not words:
            return [""]

        lines = []
        current = words[0]

        for word in words[1:]:
            test = current + " " + word
            test_w, _, _ = self._measure_text(draw, test, font, spacing)

            if test_w <= max_width:
                current = test
            else:
                lines.append(current)
                current = word

        lines.append(current)
        return lines

    def _wrap_text(self, draw, text, font, max_width, spacing):
        paragraphs = text.splitlines() if text else [""]
        wrapped_lines = []

        for para in paragraphs:
            wrapped_lines.extend(
                self._wrap_paragraph(draw, para, font, max_width, spacing)
            )

        return "\n".join(wrapped_lines)

    def make_banner(
        self,
        text,
        width,
        height,
        font_size,
        margin,
        wrap_text,
        auto_fit,
        bold_strength,
    ):
        text = str(text)

        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        max_width = max(1, width - 2 * margin)
        max_height = max(1, height - 2 * margin)

        chosen_font = None
        chosen_text = text
        chosen_spacing = max(4, int(font_size * 0.15))
        chosen_bbox = (0, 0, 0, 0)

        min_font_size = 8
        test_size = font_size

        while test_size >= min_font_size:
            font = self._get_font(test_size)
            spacing = max(2, int(test_size * 0.15))

            candidate_text = text
            if wrap_text:
                candidate_text = self._wrap_text(
                    draw, text, font, max_width, spacing
                )

            tw, th, bbox = self._measure_text(draw, candidate_text, font, spacing)

            if not auto_fit or (tw <= max_width and th <= max_height):
                chosen_font = font
                chosen_text = candidate_text
                chosen_spacing = spacing
                chosen_bbox = bbox
                break

            test_size -= 1

        if chosen_font is None:
            chosen_font = self._get_font(min_font_size)
            chosen_spacing = max(2, int(min_font_size * 0.15))
            chosen_text = text
            if wrap_text:
                chosen_text = self._wrap_text(
                    draw, text, chosen_font, max_width, chosen_spacing
                )
            _, _, chosen_bbox = self._measure_text(
                draw, chosen_text, chosen_font, chosen_spacing
            )

        x0, y0, x1, y1 = chosen_bbox
        text_w = x1 - x0
        text_h = y1 - y0

        x = (width - text_w) / 2 - x0
        y = (height - text_h) / 2 - y0

        offsets = [(0, 0)]
        for d in range(1, bold_strength + 1):
            offsets.extend([
                ( d,  0),
                (-d,  0),
                ( 0,  d),
                ( 0, -d),
                ( d,  d),
                (-d, -d),
                ( d, -d),
                (-d,  d),
            ])

        for ox, oy in offsets:
            draw.multiline_text(
                (x + ox, y + oy),
                chosen_text,
                font=chosen_font,
                fill=(255, 255, 255),
                spacing=chosen_spacing,
                align="center",
            )

        np_img = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_img).unsqueeze(0)  # [1, H, W, 3]

        return (tensor,)




class AnyType(str):
    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


class HoneyBannerDisplay:
    # Fixed appearance settings.
    WIDTH = 1024
    HEIGHT = 96
    MARGIN_X = 20
    MARGIN_Y = 10
    START_FONT_SIZE = 72
    MIN_FONT_SIZE = 10
    LINE_SPACING_FACTOR = 0.15
    BOLD_STRENGTH = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY_TYPE,),
            }
        }

    # This makes the node receive the whole value/list directly,
    # instead of being auto-run item-by-item.
    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("banner",)
    FUNCTION = "render_banner"
    CATEGORY = "Honey/Text"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _get_font(self, size):
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\seguisb.ttf",
            "arialbd.ttf",
            "DejaVuSans-Bold.ttf",
        ]

        for path in candidates:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue

        return ImageFont.load_default()

    def _stringify(self, value):
        # Because INPUT_IS_LIST = True, the input usually arrives as a list.
        if isinstance(value, list):
            if len(value) == 0:
                return ""
            if len(value) == 1:
                return self._stringify(value[0])
            return "\n".join(self._stringify(v) for v in value)

        try:
            return str(value)
        except Exception:
            return repr(value)

    def _measure_text(self, draw, text, font, spacing):
        if not text:
            text = " "
        bbox = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            spacing=spacing,
            align="center",
        )
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h, bbox

    def _wrap_line(self, draw, line, font, max_width, spacing):
        if not line.strip():
            return [""]

        words = line.split()
        if not words:
            return [""]

        lines = []
        current = words[0]

        for word in words[1:]:
            trial = current + " " + word
            trial_w, _, _ = self._measure_text(draw, trial, font, spacing)
            if trial_w <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word

        lines.append(current)
        return lines

    def _wrap_text(self, draw, text, font, max_width, spacing):
        paragraphs = text.splitlines() if text else [""]
        wrapped = []

        for para in paragraphs:
            wrapped.extend(self._wrap_line(draw, para, font, max_width, spacing))

        return "\n".join(wrapped)

    def render_banner(self, value):
        text = self._stringify(value)

        width = self.WIDTH
        height = self.HEIGHT
        max_width = max(1, width - 2 * self.MARGIN_X)
        max_height = max(1, height - 2 * self.MARGIN_Y)

        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        chosen_font = None
        chosen_text = text
        chosen_spacing = 4
        chosen_bbox = (0, 0, 0, 0)

        for font_size in range(self.START_FONT_SIZE, self.MIN_FONT_SIZE - 1, -1):
            font = self._get_font(font_size)
            spacing = max(2, int(font_size * self.LINE_SPACING_FACTOR))
            wrapped_text = self._wrap_text(draw, text, font, max_width, spacing)

            tw, th, bbox = self._measure_text(draw, wrapped_text, font, spacing)

            if tw <= max_width and th <= max_height:
                chosen_font = font
                chosen_text = wrapped_text
                chosen_spacing = spacing
                chosen_bbox = bbox
                break

        if chosen_font is None:
            chosen_font = self._get_font(self.MIN_FONT_SIZE)
            chosen_spacing = max(2, int(self.MIN_FONT_SIZE * self.LINE_SPACING_FACTOR))
            chosen_text = self._wrap_text(
                draw, text, chosen_font, max_width, chosen_spacing
            )
            _, _, chosen_bbox = self._measure_text(
                draw, chosen_text, chosen_font, chosen_spacing
            )

        x0, y0, x1, y1 = chosen_bbox
        text_w = x1 - x0
        text_h = y1 - y0

        x = (width - text_w) / 2 - x0
        y = (height - text_h) / 2 - y0

        offsets = [(0, 0)]
        for d in range(1, self.BOLD_STRENGTH + 1):
            offsets.extend([
                ( d,  0),
                (-d,  0),
                ( 0,  d),
                ( 0, -d),
                ( d,  d),
                (-d, -d),
                ( d, -d),
                (-d,  d),
            ])

        for ox, oy in offsets:
            draw.multiline_text(
                (x + ox, y + oy),
                chosen_text,
                font=chosen_font,
                fill=(255, 255, 255),
                spacing=chosen_spacing,
                align="center",
            )

        np_img = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_img).unsqueeze(0)  # [1, H, W, 3]

        return (tensor,)



class AnyType(str):
    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


def stringify_value(value):
    if value is None:
        return "None"

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {stringify_value(item)}"
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"

        return "\n".join(
            stringify_value(item)
            for item in value
        )

    try:
        return str(value)
    except Exception:
        return repr(value)


class HoneyShowBanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY_TYPE,),
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "show"
    CATEGORY = "Honey/Display"
    OUTPUT_NODE = True

    def show(self, value):
        # INPUT_IS_LIST means the complete incoming value/list is received.
        display_text = stringify_value(value)

        return {
            "ui": {
                "banner_text": [display_text],
            },
            "result": (value,),
        }


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "HoneyBannerDisplay": HoneyBannerDisplay,
    "HoneyTextBanner": HoneyTextBanner,
    "HoneyShowBanner": HoneyShowBanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HoneyBannerDisplay": "Banner Display",
    "HoneyTextBanner": "Text Banner",
    "HoneyShowBanner": "Show Banner",
}



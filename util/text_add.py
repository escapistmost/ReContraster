import os
import textwrap

from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "CreatiLayout", "utils", "arial.ttf")
)


def hex_to_rgb(color, fallback=(255, 255, 255)):
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return tuple(int(max(0, min(255, value))) for value in color[:3])
    if not isinstance(color, str):
        return fallback
    color = color.strip()
    if color.lower() == "auto":
        return fallback
    if color.startswith("#"):
        color = color[1:]
    if len(color) == 3:
        color = "".join(char * 2 for char in color)
    if len(color) != 6:
        return fallback
    try:
        return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))
    except ValueError:
        return fallback


def _font(font_size, font_path=None):
    font_path = font_path or DEFAULT_FONT
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def _text_bbox(draw, text, font, spacing):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_words(draw, text, font, max_width):
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    wrapped = []
    for line in lines:
        width = draw.textbbox((0, 0), line, font=font)[2]
        if width <= max_width:
            wrapped.append(line)
            continue
        approx_chars = max(1, int(len(line) * max_width / max(width, 1)))
        wrapped.extend(textwrap.wrap(line, width=approx_chars) or [line])
    return wrapped


def _fit_text(draw, text, font_path, box_width, box_height, requested_size=None):
    max_size = int(requested_size) if requested_size else int(min(box_height * 0.72, box_width * 0.22))
    max_size = max(10, min(max_size, 220))

    for size in range(max_size, 7, -2):
        font = _font(size, font_path)
        spacing = max(2, int(size * 0.12))
        lines = _wrap_words(draw, text, font, box_width)
        rendered = "\n".join(lines)
        width, height = _text_bbox(draw, rendered, font, spacing)
        if width <= box_width and height <= box_height:
            return rendered, font, spacing, size

    size = 8
    font = _font(size, font_path)
    return str(text), font, 1, size


def _box_to_pixels(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = int(max(0.0, min(1.0, x1)) * width)
    y1 = int(max(0.0, min(1.0, y1)) * height)
    x2 = int(max(0.0, min(1.0, x2)) * width)
    y2 = int(max(0.0, min(1.0, y2)) * height)
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _auto_colors(image, box):
    crop = image.crop(box).convert("RGB")
    pixels = list(crop.resize((1, 1)).getdata())[0]
    luminance = 0.2126 * pixels[0] + 0.7152 * pixels[1] + 0.0722 * pixels[2]
    if luminance < 128:
        return (255, 255, 255), (0, 0, 0)
    return (0, 0, 0), (255, 255, 255)


def add_text_blocks(image_path, text_blocks, output_path=None, font_path=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for block in text_blocks or []:
        text = str(block.get("text", "")).strip()
        box = block.get("box")
        if not text or not box:
            continue

        x1, y1, x2, y2 = _box_to_pixels(box, width, height)
        pad = int(min(x2 - x1, y2 - y1) * float(block.get("padding", 0.06)))
        inner = (x1 + pad, y1 + pad, x2 - pad, y2 - pad)
        if inner[2] <= inner[0] or inner[3] <= inner[1]:
            inner = (x1, y1, x2, y2)

        auto_fill, auto_stroke = _auto_colors(image, inner)
        fill = auto_fill if str(block.get("color", "auto")).lower() == "auto" else hex_to_rgb(block.get("color"), auto_fill)
        stroke_fill = (
            auto_stroke
            if str(block.get("stroke_color", "auto")).lower() == "auto"
            else hex_to_rgb(block.get("stroke_color"), auto_stroke)
        )

        rendered, font, spacing, size = _fit_text(
            draw,
            text,
            block.get("font_path") or font_path,
            inner[2] - inner[0],
            inner[3] - inner[1],
            block.get("font_size"),
        )
        text_width, text_height = _text_bbox(draw, rendered, font, spacing)

        align = str(block.get("align", "center")).lower()
        valign = str(block.get("valign", "center")).lower()
        if align == "left":
            x = inner[0]
        elif align == "right":
            x = inner[2] - text_width
        else:
            x = inner[0] + (inner[2] - inner[0] - text_width) / 2

        if valign == "top":
            y = inner[1]
        elif valign == "bottom":
            y = inner[3] - text_height
        else:
            y = inner[1] + (inner[3] - inner[1] - text_height) / 2

        stroke_width = block.get("stroke_width")
        if stroke_width is None:
            stroke_width = max(1, int(size * 0.06))

        draw.multiline_text(
            (int(x), int(y)),
            rendered,
            font=font,
            fill=fill,
            spacing=spacing,
            align=align if align in {"left", "center", "right"} else "center",
            stroke_width=int(stroke_width),
            stroke_fill=stroke_fill,
        )

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)
    return image


def add_text(img, text, box, color, output_path, thick=6, font=None, width=1024, height=1024):
    stroke_width = max(1, int(thick))
    return add_text_blocks(
        img,
        [
            {
                "text": text,
                "box": box,
                "color": color,
                "stroke_width": stroke_width,
                "align": "center",
                "valign": "center",
            }
        ],
        output_path,
    )

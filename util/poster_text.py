import ast
import copy
import json
import re

from .agent_tree import single_chat


TEXT_LAYOUT_KEYS = {
    "text_layout",
    "text_areas",
    "typography",
    "visual_texts",
    "reserved_text_regions",
}

TEXT_ELEMENT_PATTERNS = (
    "text:",
    "text style",
    "font",
    "lettering",
    "typography",
    "slogan",
    "title",
    "caption",
    "headline",
    "visual text",
    "words reading",
    "text reading",
    "reading '",
    "reading \"",
)


def parse_structured_response(response):
    """Parse JSON-like LLM output while tolerating Python booleans."""
    if not isinstance(response, str):
        return response

    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
    if starts:
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end > start:
            cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return ast.literal_eval(cleaned)


def fallback_visual_texts(prompt):
    lower = prompt.lower()
    if "save water" in lower or "saving water" in lower:
        text = "SAVE WATER"
    elif "peace" in lower:
        text = "CHOOSE PEACE"
    elif "environment" in lower or "climate" in lower:
        text = "PROTECT OUR FUTURE"
    else:
        text = re.sub(
            r"^\s*design\s+(?:a\s+)?(?:contrast\s+)?poster\s+(?:to|for|calling for|about)?\s*",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip(" .")
        if not text:
            text = "MAKE THE CONTRAST COUNT"
        text = text[:48].upper()
    return [{"label": "title", "text": text}]


def normalize_visual_texts(value, prompt):
    if value is None:
        return fallback_visual_texts(prompt)

    if isinstance(value, str):
        value = [value]

    if isinstance(value, dict):
        if any(key in value for key in ("text", "content", "title", "slogan")):
            value = [value]
        else:
            value = [
                {"label": label, "text": text}
                for label, text in value.items()
                if isinstance(text, str) and text.strip()
            ]

    texts = []
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if isinstance(item, str):
                text = item.strip()
                label = "title" if idx == 0 else "subtitle"
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("visual_text")
                    or item.get("title")
                    or item.get("slogan")
                    or ""
                )
                label = item.get("label") or item.get("type") or ("title" if idx == 0 else "subtitle")
                text = str(text).strip()
            else:
                continue

            if text:
                texts.append({"label": str(label), "text": text})

    return texts or fallback_visual_texts(prompt)


def normalize_text_extraction(data, prompt):
    if not isinstance(data, dict):
        return {"theme": prompt, "visual_texts": fallback_visual_texts(prompt)}

    theme = data.get("theme") or data.get("poster_theme") or data.get("image_theme") or prompt
    theme = str(theme).strip() or prompt
    visual_texts = normalize_visual_texts(
        data.get("visual_texts") or data.get("texts") or data.get("text"),
        prompt,
    )
    return {"theme": theme, "visual_texts": visual_texts}


def extract_theme_and_visual_texts(api, url, prompt, system_message=None):
    if not system_message:
        return normalize_text_extraction({}, prompt), ""

    response = single_chat(
        api,
        url,
        f"Poster requirement:\n{prompt}",
        role="user",
        system_message=system_message,
    )
    return normalize_text_extraction(parse_structured_response(response), prompt), response


def visual_text_context(visual_texts):
    return json.dumps(visual_texts, ensure_ascii=False)


def build_agent_prompt(theme, visual_texts, feedback=None):
    parts = [
        f"Poster theme: {theme}",
        "Visual texts for final rendering only. Reserve clear poster space for these texts, but do not describe them as visual objects for the diffusion model:",
        visual_text_context(visual_texts),
    ]
    if feedback:
        parts.append(f"Feedback: {feedback}")
    return "\n".join(parts)


def _looks_like_text_element(description):
    text = str(description).lower()
    return any(pattern in text for pattern in TEXT_ELEMENT_PATTERNS)


def _valid_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        coords = [float(value) for value in box]
    except (TypeError, ValueError):
        return False
    return coords[2] > coords[0] and coords[3] > coords[1]


def _clip_box(box):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _strip_text_sentences(description):
    if not isinstance(description, str):
        return description
    sentences = re.split(r"(?<=[.!?])\s+", description)
    kept = [sentence for sentence in sentences if not _looks_like_text_element(sentence)]
    if kept:
        return " ".join(kept).strip()
    cleaned = re.sub(
        r"\b(?:with|featuring|including|showing|containing)?\s*"
        r"(?:central|glowing|large|readable|poster|sharp)?\s*"
        r"(?:text|slogan|caption|title|lettering|typography|words)\b[^.!,;]*",
        "",
        description,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
    return cleaned or "A regional poster scene without readable text."


def split_poster_design_for_generation(poster_design):
    design = copy.deepcopy(poster_design)
    if not isinstance(design, dict):
        return design, []

    text_layout = []
    for key in list(TEXT_LAYOUT_KEYS):
        value = design.pop(key, None)
        if value and key in {"text_layout", "text_areas", "reserved_text_regions"}:
            text_layout = value

    for region_key in ("region1", "region2"):
        if region_key in design:
            design[region_key] = _strip_text_sentences(design[region_key])

    box_data = design.get("box")
    if isinstance(box_data, dict):
        clean_box = {}
        for region_key in ("region1", "region2"):
            region_boxes = box_data.get(region_key, {})
            if not isinstance(region_boxes, dict):
                continue
            clean_region = {}
            for description, box in region_boxes.items():
                clipped = _clip_box(box) if _valid_box(box) else None
                if clipped is None or _looks_like_text_element(description):
                    continue
                clean_region[description] = clipped
            clean_box[region_key] = clean_region
        design["box"] = clean_box

    return design, text_layout


def default_text_layout(visual_texts):
    boxes = (
        [0.08, 0.05, 0.92, 0.18],
        [0.12, 0.82, 0.88, 0.94],
    )
    blocks = []
    for idx, item in enumerate(visual_texts[:2]):
        blocks.append(
            {
                "label": item.get("label", "title" if idx == 0 else "subtitle"),
                "text": item["text"],
                "box": boxes[min(idx, len(boxes) - 1)],
                "align": "center",
                "valign": "center",
                "color": "auto",
                "stroke_color": "auto",
            }
        )
    return blocks


def normalize_text_layout(layout, visual_texts):
    if isinstance(layout, dict):
        if any(key in layout for key in ("items", "texts", "blocks")):
            layout = layout.get("items") or layout.get("texts") or layout.get("blocks")
        else:
            layout = [layout]

    if not isinstance(layout, list):
        return default_text_layout(visual_texts)

    blocks = []
    for idx, item in enumerate(layout):
        if not isinstance(item, dict):
            continue
        box = item.get("box") or item.get("bbox") or item.get("position")
        clipped = _clip_box(box) if _valid_box(box) else None
        if clipped is None:
            continue

        fallback_text = visual_texts[idx]["text"] if idx < len(visual_texts) else ""
        text = item.get("text") or item.get("content") or fallback_text
        text = str(text).strip()
        if not text:
            continue

        blocks.append(
            {
                "label": item.get("label") or item.get("type") or ("title" if idx == 0 else "subtitle"),
                "text": text,
                "box": clipped,
                "align": item.get("align", "center"),
                "valign": item.get("valign", "center"),
                "color": item.get("color", "auto"),
                "stroke_color": item.get("stroke_color", "auto"),
                "stroke_width": item.get("stroke_width"),
                "font_size": item.get("font_size"),
            }
        )

    return blocks or default_text_layout(visual_texts)

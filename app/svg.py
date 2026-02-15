import re

from fastapi import HTTPException

SVG_QUALITY_HINTS: dict[str, str] = {
    "low": (
        "Style: clean flat design. Use simple geometric shapes, solid fills, minimal paths. "
        "No gradients or filters. Think app-icon or logo level simplicity."
    ),
    "medium": (
        "Style: polished vector illustration. Use layered shapes with gradients for depth, "
        "highlights, and shadows. Build complex objects by composing many precise smaller shapes "
        "(e.g. a shoe = sole shape + upper shape + lace loops + tongue + stitching lines). "
        "Each distinct part of the subject should be its own shape with correct proportions."
    ),
    "high": (
        "Style: detailed, near-realistic vector illustration. Construct the subject from many "
        "precise, anatomically/structurally correct sub-shapes. Every distinct part must be its "
        "own carefully shaped path (e.g. a running shoe needs: outsole, midsole, upper panel, "
        "toe box, heel counter, tongue, lace eyelets, individual laces, swoosh/logo area, "
        "pull tab, stitching lines — each as separate paths with correct proportions). "
        "Use linear and radial gradients for realistic shading and material texture. "
        "Add subtle highlights, shadow layers, and edge detail. "
        "Stay under ~1000 path elements for performance."
    ),
}

SVG_MAX_TOKENS = 16000

SVG_SYSTEM_PROMPT = (
    "You are an expert SVG illustrator who creates structurally accurate vector art. "
    "Output ONLY raw SVG markup — no markdown fences, no explanation, no extra text.\n\n"
    "Technical requirements:\n"
    '- xmlns="http://www.w3.org/2000/svg" attribute, viewBox "0 0 {width} {height}"\n'
    "- No external resources (images, fonts, stylesheets) — inline styles only\n"
    "- Web-safe fonts only (Arial, Helvetica, Georgia, Verdana, sans-serif, serif)\n\n"
    "Critical illustration rules:\n"
    "- BEFORE writing SVG, mentally decompose the subject into its real structural parts. "
    "A shoe is not a dome — it has a flat sole, a low-profile upper, laces, a tongue, etc. "
    "A car is not a blob — it has wheels, windows, a hood, doors, etc.\n"
    "- Get the PROPORTIONS and SILHOUETTE right first. The overall outline must be "
    "recognizable as the subject even without color or detail.\n"
    "- Build from back to front using layered shapes — background elements first, "
    "foreground details on top.\n"
    "- Use <path> with precise d attributes for organic curves. Use basic shapes "
    "(rect, circle, ellipse) where geometrically appropriate.\n\n"
    "{quality_hint}\n\n"
    "If reference images are provided, study their structure and proportions carefully."
)


def parse_svg_dimensions(size: str) -> tuple[int, int]:
    parts = size.split("x")
    return int(parts[0]), int(parts[1])


def extract_svg(text: str) -> str:
    match = re.search(r"<svg[\s\S]*?</svg>", text, re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=502, detail="Model did not return valid SVG markup.")
    svg = match.group(0)
    if "xmlns" not in svg:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg

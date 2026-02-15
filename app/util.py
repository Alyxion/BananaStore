import re
from unicodedata import normalize

from fastapi import UploadFile


def fallback_filename(description: str) -> str:
    base = normalize("NFKD", description).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^a-zA-Z0-9\s-]", "", base).strip().lower()
    base = re.sub(r"[-\s]+", "-", base)
    if not base:
        return "generated-image"
    return base[:80].strip("-") or "generated-image"


def sanitize_filename(raw: str, fallback: str) -> str:
    cleaned = normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.strip().lower().replace(".png", "").replace(".svg", "")
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", cleaned)
    cleaned = re.sub(r"[-\s]+", "-", cleaned).strip("-")
    return (cleaned or fallback)[:80]


async def read_reference_images(
    reference_images: list[UploadFile] | None,
) -> tuple[list[tuple[str, bytes, str]], list[str]]:
    parsed: list[tuple[str, bytes, str]] = []
    svg_sources: list[str] = []
    for file in reference_images or []:
        content = await file.read()
        if not content:
            continue
        mime = file.content_type or "application/octet-stream"
        name = file.filename or "reference-image"
        if mime == "image/svg+xml" or name.endswith(".svg"):
            try:
                svg_sources.append(content.decode("utf-8", errors="ignore"))
            except Exception:
                pass
            continue
        parsed.append((name, content, mime))
    return parsed, svg_sources

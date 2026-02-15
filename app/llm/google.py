import base64
from typing import Any

import httpx
from fastapi import HTTPException

from app.costs import record_google_image, tracker
from app.llm import ensure_api_key, safe_provider_error, to_data_url


async def generate_image(
    description: str,
    size: str,
    quality: str,
    ratio: str,
    reference_images: list[tuple[str, bytes, str]],
    svg_sources: list[str] | None = None,
) -> str:
    api_key = ensure_api_key("GOOGLE_API_KEY", "Google")
    tracker.check_limit(0.04)

    prompt_text = (
        f"Generate one high quality image. Prompt: {description}. "
        f"Requested size: {size}. Requested quality: {quality}. Requested aspect ratio: {ratio}."
    )
    if svg_sources:
        for i, src in enumerate(svg_sources, 1):
            prompt_text += f"\n\nReference SVG {i} source (use as visual inspiration):\n{src}"

    parts: list[dict[str, Any]] = [{"text": prompt_text}]

    for _, content, mime_type in reference_images:
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(content).decode("utf-8"),
                }
            }
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "responseModalities": ["IMAGE", "TEXT"],
                },
            },
        )

    if response.status_code >= 400:
        raise safe_provider_error("Google", response)

    payload = response.json()
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        candidate_parts = ((candidate.get("content") or {}).get("parts")) or []
        for part in candidate_parts:
            inline_data = part.get("inline_data") or part.get("inlineData")
            if inline_data and inline_data.get("data"):
                mime = inline_data.get("mime_type") or inline_data.get("mimeType") or "image/png"
                record_google_image()
                return to_data_url(inline_data["data"], mime)

    raise HTTPException(status_code=502, detail=f"Google returned no image payload: {payload}")

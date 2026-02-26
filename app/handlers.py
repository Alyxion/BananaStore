"""Core handler functions — no FastAPI types. Used by both REST routes and WebSocket."""

import base64
from dataclasses import asdict
from typing import Any

from fastapi import HTTPException

from app.config import settings
from app.costs import CostTracker
from app.llm import anthropic as llm_anthropic
from app.llm import azure_openai as llm_azure
from app.llm import google as llm_google
from app.llm import openai as llm_openai
from app.providers import PROVIDER_CAPABILITIES
from app.util import fallback_filename


def _azure_available() -> bool:
    return bool(settings.get("AZURE_OPENAI_API_KEY"))


async def handle_providers() -> dict[str, Any]:
    azure_active = _azure_available()
    providers = {}
    for provider_id, details in PROVIDER_CAPABILITIES.items():
        # When Azure is configured, hide the direct OpenAI provider
        if provider_id == "openai" and azure_active:
            continue
        if provider_id == "azure_openai" and not azure_active:
            continue
        key_name = details["requiresKey"]
        providers[provider_id] = {
            **details,
            "hasKey": bool(settings.get(key_name)),
        }
    return {"providers": providers}


async def handle_suggest_filename(description: str) -> dict:
    description = description.strip()
    if not description:
        return {"filename": "generated-image"}
    fallback = fallback_filename(description)
    filename = await llm_openai.suggest_filename(description, fallback)
    return {"filename": filename}


async def handle_transcribe(audio_bytes: bytes, filename: str, content_type: str) -> dict:
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio payload provided.")
    if len(audio_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file is too large. Keep it under 8MB.")
    transcribe = llm_azure.transcribe_audio if _azure_available() else llm_openai.transcribe_audio
    text = await transcribe(
        content=audio_bytes,
        filename=filename,
        content_type=content_type,
    )
    return {"text": text}


async def handle_describe_image(image_data_url: str, source_text: str, language: str) -> dict:
    image_data_url = image_data_url.strip()
    if not image_data_url.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image_data_url must be a valid data URL.")
    description = await llm_openai.describe_image(
        image_data_url=image_data_url,
        source_text=source_text.strip(),
        language=language.strip(),
    )
    return {"description": description}


async def handle_tts(text: str, language: str) -> bytes:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required.")
    synthesize = llm_azure.synthesize_speech if _azure_available() else llm_openai.synthesize_speech
    return await synthesize(text=text, language=language.strip())


async def handle_generate(
    provider: str,
    description: str,
    quality: str,
    ratio: str,
    format: str,
    reference_images: list[dict[str, Any]],
    tracker: CostTracker,
) -> dict:
    """Generate an image.

    reference_images: list of {"name": str, "data_b64": str, "content_type": str}
    """
    if provider not in PROVIDER_CAPABILITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    options = PROVIDER_CAPABILITIES[provider]
    format_qualities = (options.get("formatQualities") or {}).get(format)
    allowed_qualities = format_qualities or options["qualities"]
    if quality not in allowed_qualities:
        raise HTTPException(status_code=400, detail=f"Unsupported quality '{quality}' for provider '{provider}'")
    if ratio not in options["ratios"]:
        raise HTTPException(status_code=400, detail=f"Unsupported ratio '{ratio}' for provider '{provider}'")
    if format not in options["formats"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{format}' for provider '{provider}'")

    size = options["ratioSizes"].get(ratio, "1024x1024")

    # Parse reference images from WS payload format
    parsed_reference_images: list[tuple[str, bytes, str]] = []
    svg_sources: list[str] = []
    for ref in reference_images:
        name = ref.get("name", "reference-image")
        data_b64 = ref.get("data_b64", "")
        ct = ref.get("content_type", "image/png")
        if not data_b64:
            continue
        raw = base64.b64decode(data_b64)
        if ct == "image/svg+xml" or name.endswith(".svg"):
            try:
                svg_sources.append(raw.decode("utf-8", errors="ignore"))
            except Exception:
                pass
            continue
        parsed_reference_images.append((name, raw, ct))

    reference_count = len(parsed_reference_images) + len(svg_sources)
    cost_before = tracker.total_usd

    if format == "Vector" and provider == "openai":
        image_data_url = await llm_openai.generate_svg(
            description=description, size=size, quality=quality, ratio=ratio,
            reference_images=parsed_reference_images, svg_sources=svg_sources,
        )
    elif format == "Vector" and provider == "anthropic":
        image_data_url = await llm_anthropic.generate_svg(
            description=description, size=size, quality=quality, ratio=ratio,
            reference_images=parsed_reference_images, svg_sources=svg_sources,
        )
    elif provider == "openai":
        image_data_url = await llm_openai.generate_image(
            description=description, size=size, quality=quality,
            reference_images=parsed_reference_images, svg_sources=svg_sources,
        )
    elif provider == "google":
        image_data_url = await llm_google.generate_image(
            description=description, size=size, quality=quality, ratio=ratio,
            reference_images=parsed_reference_images, svg_sources=svg_sources,
        )
    elif provider == "azure_openai":
        image_data_url = await llm_azure.generate_image(
            description=description, size=size, quality=quality,
            reference_images=parsed_reference_images, svg_sources=svg_sources,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    cost_usd = tracker.total_usd - cost_before
    return {
        "provider": provider,
        "size": size,
        "quality": quality,
        "ratio": ratio,
        "format": format,
        "image_data_url": image_data_url,
        "used_reference_images": reference_count,
        "cost_usd": round(cost_usd, 6) if cost_usd > 0 else None,
    }


async def handle_costs(tracker: CostTracker) -> dict:
    return {
        "total_usd": tracker.total_usd,
        "limit_usd": tracker.limit_usd,
        "by_category": tracker.totals_by_category(),
        "by_provider": tracker.totals_by_provider(),
        "entry_count": len(tracker.entries),
    }


async def handle_costs_history(tracker: CostTracker) -> list[dict]:
    return [asdict(e) for e in tracker.entries]


async def handle_costs_limit(tracker: CostTracker, limit_usd: float | None) -> dict:
    tracker.limit_usd = limit_usd
    return {"limit_usd": tracker.limit_usd}

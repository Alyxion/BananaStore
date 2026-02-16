from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.costs import tracker
from app.llm import anthropic as llm_anthropic
from app.llm import google as llm_google
from app.llm import openai as llm_openai
from app.providers import PROVIDER_CAPABILITIES
from app.schemas import (
    CostLimitRequest,
    CostSummary,
    DescribeImageRequest,
    DescribeImageResponse,
    FilenameRequest,
    FilenameResponse,
    GenerateResponse,
    TranscriptionResponse,
    TtsRequest,
)
from app.util import fallback_filename, read_reference_images

router = APIRouter()


@router.get("/api/providers")
async def get_providers() -> dict[str, Any]:
    providers = {}
    for provider_id, details in PROVIDER_CAPABILITIES.items():
        key_name = details["requiresKey"]
        providers[provider_id] = {
            **details,
            "hasKey": bool(settings.get(key_name)),
        }
    return {"providers": providers}


@router.post("/api/suggest-filename", response_model=FilenameResponse)
async def suggest_filename(payload: FilenameRequest) -> FilenameResponse:
    description = payload.description.strip()
    if not description:
        return FilenameResponse(filename="generated-image")

    fallback = fallback_filename(description)
    filename = await llm_openai.suggest_filename(description, fallback)
    return FilenameResponse(filename=filename)


@router.post("/api/transcribe-openai", response_model=TranscriptionResponse)
async def transcribe_openai(audio: UploadFile = File(...)) -> TranscriptionResponse:
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="No audio payload provided.")

    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file is too large. Keep it under 8MB.")

    text = await llm_openai.transcribe_audio(
        content=content,
        filename=audio.filename or "voice.webm",
        content_type=audio.content_type or "audio/webm",
    )
    return TranscriptionResponse(text=text)


@router.post("/api/describe-image", response_model=DescribeImageResponse)
async def describe_image(payload: DescribeImageRequest) -> DescribeImageResponse:
    image_data_url = payload.image_data_url.strip()
    if not image_data_url.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image_data_url must be a valid data URL.")

    description = await llm_openai.describe_image(
        image_data_url=image_data_url,
        source_text=payload.source_text.strip(),
        language=payload.language.strip(),
    )
    return DescribeImageResponse(description=description)


@router.post("/api/tts-openai")
async def tts_openai(payload: TtsRequest) -> Response:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required.")

    audio_bytes = await llm_openai.synthesize_speech(text=text, language=payload.language.strip())
    return Response(content=audio_bytes, media_type="audio/mpeg")


@router.post("/api/generate", response_model=GenerateResponse)
async def generate_image(
    provider: str = Form(...),
    description: str = Form(...),
    quality: str = Form(...),
    ratio: str = Form(...),
    format: str = Form("Photo"),
    reference_images: list[UploadFile] | None = File(default=None),
) -> GenerateResponse:
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
    parsed_reference_images, svg_sources = await read_reference_images(reference_images)
    reference_count = len(parsed_reference_images) + len(svg_sources)

    cost_before = tracker.total_usd
    if format == "Vector" and provider == "openai":
        image_data_url = await llm_openai.generate_svg(
            description=description,
            size=size,
            quality=quality,
            ratio=ratio,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    elif format == "Vector" and provider == "anthropic":
        image_data_url = await llm_anthropic.generate_svg(
            description=description,
            size=size,
            quality=quality,
            ratio=ratio,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    elif provider == "openai":
        image_data_url = await llm_openai.generate_image(
            description=description,
            size=size,
            quality=quality,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    elif provider == "google":
        image_data_url = await llm_google.generate_image(
            description=description,
            size=size,
            quality=quality,
            ratio=ratio,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    cost_usd = tracker.total_usd - cost_before
    return GenerateResponse(
        provider=provider,
        size=size,
        quality=quality,
        ratio=ratio,
        format=format,
        image_data_url=image_data_url,
        used_reference_images=reference_count,
        cost_usd=round(cost_usd, 6) if cost_usd > 0 else None,
    )


@router.get("/api/costs", response_model=CostSummary)
async def get_costs() -> CostSummary:
    return CostSummary(
        total_usd=tracker.total_usd,
        limit_usd=tracker.limit_usd,
        by_category=tracker.totals_by_category(),
        by_provider=tracker.totals_by_provider(),
        entry_count=len(tracker.entries),
    )


@router.get("/api/costs/history")
async def get_costs_history() -> list[dict]:
    return [asdict(e) for e in tracker.entries]


@router.post("/api/costs/limit")
async def set_cost_limit(payload: CostLimitRequest) -> dict:
    tracker.limit_usd = payload.limit_usd
    return {"limit_usd": tracker.limit_usd}

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.costs import SpendingLimitExceeded, tracker
from app.llm import openai as llm_openai
from app.llm import google as llm_google
from app.llm import anthropic as llm_anthropic
from app.util import fallback_filename, read_reference_images

PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "qualities": ["auto", "low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "ratioSizes": {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"},
        "formats": ["Photo", "Vector"],
        "formatQualities": {"Photo": ["auto", "low", "medium", "high"], "Vector": ["low", "medium", "high"]},
        "requiresKey": "OPENAI_API_KEY",
    },
    "google": {
        "label": "Google",
        "qualities": ["standard", "hd"],
        "ratios": ["1:1", "16:9", "9:16"],
        "ratioSizes": {"1:1": "1024x1024", "16:9": "1280x720", "9:16": "720x1280"},
        "formats": ["Photo"],
        "requiresKey": "GOOGLE_API_KEY",
    },
    "anthropic": {
        "label": "Anthropic",
        "qualities": ["low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "ratioSizes": {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"},
        "formats": ["Vector"],
        "requiresKey": "ANTHROPIC_API_KEY",
    },
}


# --- Pydantic models ---

class GenerateResponse(BaseModel):
    provider: str
    size: str
    quality: str
    ratio: str
    format: str
    image_data_url: str
    used_reference_images: int
    cost_usd: float | None = None


class FilenameRequest(BaseModel):
    description: str


class FilenameResponse(BaseModel):
    filename: str


class TranscriptionResponse(BaseModel):
    text: str


class DescribeImageRequest(BaseModel):
    image_data_url: str
    source_text: str = ""
    language: str = ""


class DescribeImageResponse(BaseModel):
    description: str


class TtsRequest(BaseModel):
    text: str
    language: str = ""


class CostLimitRequest(BaseModel):
    limit_usd: float | None = None


class CostSummary(BaseModel):
    total_usd: float
    limit_usd: float | None
    by_category: dict[str, float]
    by_provider: dict[str, float]
    entry_count: int


# --- App setup ---

@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    limit = os.getenv("COST_LIMIT_USD")
    if limit:
        tracker.limit_usd = float(limit)
    yield


app = FastAPI(title="BananaStore", lifespan=lifespan)


@app.exception_handler(SpendingLimitExceeded)
async def spending_limit_handler(_request: Request, exc: SpendingLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc), "limit": exc.limit, "current": exc.current, "attempted": exc.attempted},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Routes ---

@app.get("/api/providers")
async def get_providers() -> dict[str, Any]:
    providers = {}
    for provider_id, details in PROVIDER_CAPABILITIES.items():
        key_name = details["requiresKey"]
        providers[provider_id] = {
            **details,
            "hasKey": bool(os.getenv(key_name)),
        }
    return {"providers": providers}


@app.post("/api/suggest-filename", response_model=FilenameResponse)
async def suggest_filename(payload: FilenameRequest) -> FilenameResponse:
    description = payload.description.strip()
    if not description:
        return FilenameResponse(filename="generated-image")

    fallback = fallback_filename(description)
    filename = await llm_openai.suggest_filename(description, fallback)
    return FilenameResponse(filename=filename)


@app.post("/api/transcribe-openai", response_model=TranscriptionResponse)
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


@app.post("/api/describe-image", response_model=DescribeImageResponse)
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


@app.post("/api/tts-openai")
async def tts_openai(payload: TtsRequest) -> Response:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required.")

    audio_bytes = await llm_openai.synthesize_speech(text=text, language=payload.language.strip())
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/api/generate", response_model=GenerateResponse)
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


@app.get("/api/costs", response_model=CostSummary)
async def get_costs() -> CostSummary:
    return CostSummary(
        total_usd=tracker.total_usd,
        limit_usd=tracker.limit_usd,
        by_category=tracker.totals_by_category(),
        by_provider=tracker.totals_by_provider(),
        entry_count=len(tracker.entries),
    )


@app.get("/api/costs/history")
async def get_costs_history() -> list[dict]:
    return [asdict(e) for e in tracker.entries]


@app.post("/api/costs/limit")
async def set_cost_limit(payload: CostLimitRequest) -> dict:
    tracker.limit_usd = payload.limit_usd
    return {"limit_usd": tracker.limit_usd}


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

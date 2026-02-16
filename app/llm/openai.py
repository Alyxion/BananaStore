import base64
from typing import Any

import httpx
from fastapi import HTTPException

from app.config import settings
from app.costs import (
    record_openai_chat,
    record_openai_image,
    record_openai_transcription,
    record_openai_tts,
    tracker,
)
from app.llm import ensure_api_key, safe_provider_error, to_data_url
from app.util import sanitize_filename
from app.svg import (
    SVG_MAX_TOKENS,
    SVG_QUALITY_HINTS,
    SVG_SYSTEM_PROMPT,
    extract_svg,
    parse_svg_dimensions,
)


async def generate_image(
    description: str,
    size: str,
    quality: str,
    reference_images: list[tuple[str, bytes, str]],
    svg_sources: list[str] | None = None,
) -> str:
    api_key = ensure_api_key("OPENAI_API_KEY", "OpenAI")
    effective_q = quality if quality != "auto" else "medium"
    from app.costs import OPENAI_IMAGE_PRICING
    est_key = (effective_q, size if size == "1024x1024" else "other")
    tracker.check_limit(OPENAI_IMAGE_PRICING.get(est_key, 0.063))
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = description
    if svg_sources:
        for i, src in enumerate(svg_sources, 1):
            prompt += f"\n\nReference SVG {i} source (use as visual inspiration):\n{src}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        if reference_images:
            data = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "n": "1",
                "output_format": "png",
            }
            files = [
                ("image[]", (file_name, content, mime_type))
                for file_name, content, mime_type in reference_images
            ]
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                data=data,
                files=files,
            )
        else:
            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "output_format": "png",
                    "n": 1,
                },
            )

    if response.status_code >= 400:
        raise safe_provider_error("OpenAI", response)

    payload = response.json()
    image_b64 = ((payload.get("data") or [{}])[0]).get("b64_json")
    if not image_b64:
        raise HTTPException(status_code=502, detail=f"OpenAI returned no image payload: {payload}")

    record_openai_image(quality, size)
    return to_data_url(image_b64)


async def generate_svg(
    description: str,
    size: str,
    quality: str,
    ratio: str,
    reference_images: list[tuple[str, bytes, str]],
    svg_sources: list[str] | None = None,
) -> str:
    api_key = ensure_api_key("OPENAI_API_KEY", "OpenAI")
    num_images = len(reference_images)
    tracker.check_limit(0.01)  # conservative estimate
    width, height = parse_svg_dimensions(size)
    quality_hint = SVG_QUALITY_HINTS.get(quality, SVG_QUALITY_HINTS["medium"])
    system_prompt = SVG_SYSTEM_PROMPT.format(width=width, height=height, quality_hint=quality_hint)

    user_content: list[dict[str, Any]] = []
    for _, content, mime_type in reference_images:
        b64 = base64.b64encode(content).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        })
    prompt_text = f"Create an SVG illustration: {description}. Target size: {size}, aspect ratio: {ratio}."
    if svg_sources:
        for i, src in enumerate(svg_sources, 1):
            prompt_text += f"\n\nReference SVG {i} source (adjust as needed):\n{src}"
    user_content.append({"type": "text", "text": prompt_text})

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5.2",
                "max_completion_tokens": SVG_MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
        )

    if response.status_code >= 400:
        raise safe_provider_error("OpenAI", response)

    payload = response.json()
    usage = payload.get("usage") or {}
    record_openai_chat("gpt-5.2", "generate_svg", usage, num_images=num_images)
    text = (((payload.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    svg = extract_svg(text)
    svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{svg_b64}"


async def suggest_filename(description: str, fallback: str) -> str:
    api_key = settings.get("OPENAI_API_KEY")
    if not api_key:
        return fallback

    payload = {
        "model": "gpt-4.1-nano",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return one short kebab-case filename only (no extension), max 8 words, "
                    "for the user's image request. No punctuation except hyphens."
                ),
            },
            {"role": "user", "content": description},
        ],
        "max_tokens": 30,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        return fallback

    body = response.json()
    usage = body.get("usage") or {}
    record_openai_chat("gpt-4.1-nano", "suggest_filename", usage)
    content = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
    if not content:
        return fallback

    return sanitize_filename(content, fallback)


async def describe_image(image_data_url: str, source_text: str, language: str) -> str:
    api_key = ensure_api_key("OPENAI_API_KEY", "OpenAI")
    tracker.check_limit(0.001)
    language_instruction = (
        f"Use language '{language}'."
        if language
        else "Use the same language as SOURCE_TEXT."
    )
    payload = {
        "model": "gpt-4.1-nano",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly artist commenting on an image you just created for someone. "
                    "Speak naturally in first person — mention what you did with their idea, "
                    "highlight a detail you're proud of, or note a creative choice you made. "
                    "Keep it to one or two sentences, 25–40 words. Be warm but not over the top. "
                    f"{language_instruction} Avoid markdown, lists, and preambles."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"SOURCE_TEXT: {source_text or 'N/A'}"},
                    {"type": "text", "text": "Comment on this image you just created for the user."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "max_tokens": 120,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=18.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        raise safe_provider_error("OpenAI", response)

    body = response.json()
    usage = body.get("usage") or {}
    record_openai_chat("gpt-4.1-nano", "describe_image", usage, num_images=1)
    description = ((((body.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
    if not description:
        raise HTTPException(status_code=502, detail="OpenAI returned no image description.")

    return description


async def transcribe_audio(content: bytes, filename: str, content_type: str) -> str:
    api_key = ensure_api_key("OPENAI_API_KEY", "OpenAI")
    tracker.check_limit(0.001)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": "gpt-4o-mini-transcribe", "response_format": "json"},
            files={"file": (filename, content, content_type)},
        )

    if response.status_code >= 400:
        raise safe_provider_error("OpenAI", response)

    payload = response.json()
    duration = payload.get("duration", 0.0)
    record_openai_transcription(duration)
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="OpenAI returned no transcript text.")

    return text


async def synthesize_speech(text: str, language: str) -> bytes:
    api_key = ensure_api_key("OPENAI_API_KEY", "OpenAI")
    tracker.check_limit(len(text) * 0.00003)
    _ = language

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "tts-1-hd",
                "voice": "nova",
                "input": text,
                "response_format": "mp3",
            },
        )

    if response.status_code >= 400:
        raise safe_provider_error("OpenAI", response)

    audio_bytes = response.content
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="OpenAI returned no audio content.")

    record_openai_tts(len(text))
    return audio_bytes

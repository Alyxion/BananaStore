import httpx
from fastapi import HTTPException

from app.config import settings


def ensure_api_key(env_name: str, provider_label: str) -> str:
    api_key = settings.get(env_name)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"{provider_label} API key not found in {env_name}")
    return api_key


def to_data_url(image_b64: str, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{image_b64}"


def safe_provider_error(provider_name: str, response: httpx.Response) -> HTTPException:
    try:
        payload = response.json()
    except ValueError:
        return HTTPException(
            status_code=502,
            detail=f"{provider_name} returned an unexpected error ({response.status_code}).",
        )

    error_obj = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error_obj, dict):
        message = error_obj.get("message", "")
        code = error_obj.get("code", "")
        if "moderation" in code or "safety" in message.lower():
            return HTTPException(
                status_code=422,
                detail=f"{provider_name}: Your prompt was blocked by the safety filter. Try rephrasing your description.",
            )
        if message:
            return HTTPException(
                status_code=502,
                detail=f"{provider_name}: {message}",
            )

    return HTTPException(
        status_code=502,
        detail=f"{provider_name} error ({response.status_code}). Please try again.",
    )

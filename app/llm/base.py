"""Base class for LLM provider integrations."""

from abc import ABC

import httpx
from fastapi import HTTPException

from app.config import settings


class LLMProvider(ABC):
    """Shared infrastructure for all LLM providers."""

    provider_name: str  # "OpenAI", "Google", "Anthropic"
    api_key_env: str    # "OPENAI_API_KEY", etc.
    default_timeout: float = 120.0

    def get_api_key(self, *, required: bool = True) -> str | None:
        """Return the API key from config/env, or raise if required and missing."""
        api_key = settings.get(self.api_key_env)
        if not api_key and required:
            raise HTTPException(
                status_code=400,
                detail=f"{self.provider_name} API key not found in {self.api_key_env}",
            )
        return api_key or None

    def check_cost(self, tracker, estimated: float) -> None:
        """Pre-flight cost check against spending limit."""
        tracker.check_limit(estimated)

    def make_client(self, timeout: float | None = None) -> httpx.AsyncClient:
        """Create an httpx async client with the provider's default timeout."""
        return httpx.AsyncClient(timeout=timeout or self.default_timeout)

    def auth_headers(self, api_key: str) -> dict[str, str]:
        """Return auth headers. Default: Bearer token. Override per provider."""
        return {"Authorization": f"Bearer {api_key}"}

    def auth_params(self, api_key: str) -> dict[str, str]:
        """Return query params for auth. Default: empty. Override per provider."""
        return {}

    def raise_on_error(self, response: httpx.Response) -> None:
        """Raise HTTPException if response indicates an error."""
        if response.status_code < 400:
            return
        try:
            payload = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail=f"{self.provider_name} returned an unexpected error ({response.status_code}).",
            )

        error_obj = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error_obj, dict):
            message = error_obj.get("message", "")
            code = error_obj.get("code", "")
            if "moderation" in code or "safety" in message.lower():
                raise HTTPException(
                    status_code=422,
                    detail=f"{self.provider_name}: Your prompt was blocked by the safety filter. Try rephrasing your description.",
                )
            if message:
                raise HTTPException(
                    status_code=502,
                    detail=f"{self.provider_name}: {message}",
                )

        raise HTTPException(
            status_code=502,
            detail=f"{self.provider_name} error ({response.status_code}). Please try again.",
        )

    @staticmethod
    def to_data_url(image_b64: str, mime_type: str = "image/png") -> str:
        return f"data:{mime_type};base64,{image_b64}"

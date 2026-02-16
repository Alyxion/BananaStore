import base64
from typing import Any

import httpx

from app.costs import record_anthropic_chat, tracker
from app.llm.base import LLMProvider
from app.svg import (
    SVG_MAX_TOKENS,
    SVG_QUALITY_HINTS,
    SVG_SYSTEM_PROMPT,
    extract_svg,
    parse_svg_dimensions,
)


class AnthropicProvider(LLMProvider):
    provider_name = "Anthropic"
    api_key_env = "ANTHROPIC_API_KEY"

    def auth_headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    async def generate_svg(
        self,
        description: str,
        size: str,
        quality: str,
        ratio: str,
        reference_images: list[tuple[str, bytes, str]],
        svg_sources: list[str] | None = None,
    ) -> str:
        api_key = self.get_api_key()
        num_images = len(reference_images)
        self.check_cost(tracker, 0.01)
        width, height = parse_svg_dimensions(size)
        quality_hint = SVG_QUALITY_HINTS.get(quality, SVG_QUALITY_HINTS["medium"])
        system_prompt = SVG_SYSTEM_PROMPT.format(width=width, height=height, quality_hint=quality_hint)

        ANTHROPIC_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
        user_content: list[dict[str, Any]] = []
        for _, content, mime_type in reference_images:
            if mime_type not in ANTHROPIC_IMAGE_MIMES:
                continue
            b64 = base64.b64encode(content).decode("utf-8")
            user_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": b64},
            })
        prompt_text = f"Create an SVG illustration: {description}. Target size: {size}, aspect ratio: {ratio}."
        if svg_sources:
            for i, src in enumerate(svg_sources, 1):
                prompt_text += f"\n\nReference SVG {i} source (adjust as needed):\n{src}"
        user_content.append({"type": "text", "text": prompt_text})

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    **self.auth_headers(api_key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-opus-4-6",
                    "max_tokens": SVG_MAX_TOKENS,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_content},
                    ],
                },
            )

        self.raise_on_error(response)

        payload = response.json()
        usage = payload.get("usage") or {}
        record_anthropic_chat("claude-opus-4-6", "generate_svg", usage, num_images=num_images)
        text_parts = [
            block.get("text", "")
            for block in (payload.get("content") or [])
            if block.get("type") == "text"
        ]
        full_text = "\n".join(text_parts)
        svg = extract_svg(full_text)
        svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{svg_b64}"


# Module-level singleton
_provider = AnthropicProvider()

# Backward-compatible shim
async def generate_svg(*a, **kw): return await _provider.generate_svg(*a, **kw)

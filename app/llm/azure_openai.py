import httpx
from fastapi import HTTPException

from app.config import settings
from app.costs import record_azure_image, record_openai_transcription, record_openai_tts, tracker
from app.llm.base import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    provider_name = "Azure OpenAI"
    api_key_env = "AZURE_OPENAI_API_KEY"

    def _endpoint(self) -> str:
        endpoint = settings.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise HTTPException(status_code=400, detail="AZURE_OPENAI_ENDPOINT not configured")
        return endpoint.rstrip("/")

    def _api_version(self) -> str:
        return settings.get("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview"

    def _deployment(self) -> str:
        deployment = settings.get("AZURE_OPENAI_DEPLOYMENT_IMAGE")
        if not deployment:
            raise HTTPException(status_code=400, detail="AZURE_OPENAI_DEPLOYMENT_IMAGE not configured")
        return deployment

    def _deployment_tts(self) -> str:
        deployment = settings.get("AZURE_OPENAI_DEPLOYMENT_TTS")
        if not deployment:
            raise HTTPException(status_code=400, detail="AZURE_OPENAI_DEPLOYMENT_TTS not configured")
        return deployment

    def _deployment_stt(self) -> str:
        deployment = settings.get("AZURE_OPENAI_DEPLOYMENT_STT")
        if not deployment:
            raise HTTPException(status_code=400, detail="AZURE_OPENAI_DEPLOYMENT_STT not configured")
        return deployment

    def auth_headers(self, api_key: str) -> dict[str, str]:
        return {"api-key": api_key}

    async def generate_image(
        self,
        description: str,
        size: str,
        quality: str,
        reference_images: list[tuple[str, bytes, str]],
        svg_sources: list[str] | None = None,
    ) -> str:
        api_key = self.get_api_key()
        effective_q = quality if quality != "auto" else "medium"
        from app.costs import OPENAI_IMAGE_PRICING
        est_key = (effective_q, size if size == "1024x1024" else "other")
        self.check_cost(tracker, OPENAI_IMAGE_PRICING.get(est_key, 0.063))

        endpoint = self._endpoint()
        deployment = self._deployment()
        api_version = self._api_version()
        headers = self.auth_headers(api_key)
        prompt = description
        if svg_sources:
            for i, src in enumerate(svg_sources, 1):
                prompt += f"\n\nReference SVG {i} source (use as visual inspiration):\n{src}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            if reference_images:
                url = f"{endpoint}/openai/deployments/{deployment}/images/edits?api-version={api_version}"
                data = {
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
                    url,
                    headers=headers,
                    data=data,
                    files=files,
                )
            else:
                url = f"{endpoint}/openai/deployments/{deployment}/images/generations?api-version={api_version}"
                response = await client.post(
                    url,
                    headers={**headers, "Content-Type": "application/json"},
                    json={
                        "prompt": prompt,
                        "size": size,
                        "quality": quality,
                        "output_format": "png",
                        "n": 1,
                    },
                )

        self.raise_on_error(response)

        payload = response.json()
        image_b64 = ((payload.get("data") or [{}])[0]).get("b64_json")
        if not image_b64:
            raise HTTPException(status_code=502, detail=f"Azure OpenAI returned no image payload: {payload}")

        record_azure_image(quality, size)
        return self.to_data_url(image_b64)

    async def transcribe_audio(self, content: bytes, filename: str, content_type: str) -> str:
        api_key = self.get_api_key()
        self.check_cost(tracker, 0.001)

        endpoint = self._endpoint()
        deployment = self._deployment_stt()
        api_version = self._api_version()

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{endpoint}/openai/deployments/{deployment}/audio/transcriptions?api-version={api_version}",
                headers=self.auth_headers(api_key),
                data={"response_format": "json"},
                files={"file": (filename, content, content_type)},
            )

        self.raise_on_error(response)

        payload = response.json()
        duration = payload.get("duration", 0.0)
        record_openai_transcription(duration)
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=502, detail="Azure OpenAI returned no transcript text.")

        return text

    async def synthesize_speech(self, text: str, language: str) -> bytes:
        api_key = self.get_api_key()
        self.check_cost(tracker, len(text) * 0.00003)
        _ = language

        endpoint = self._endpoint()
        deployment = self._deployment_tts()
        api_version = self._api_version()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{endpoint}/openai/deployments/{deployment}/audio/speech?api-version={api_version}",
                headers={
                    **self.auth_headers(api_key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": deployment,
                    "voice": "nova",
                    "input": text,
                    "response_format": "mp3",
                },
            )

        self.raise_on_error(response)

        audio_bytes = response.content
        if not audio_bytes:
            raise HTTPException(status_code=502, detail="Azure OpenAI returned no audio content.")

        record_openai_tts(len(text))
        return audio_bytes


# Module-level singleton
_provider = AzureOpenAIProvider()

# Backward-compatible shims
async def generate_image(*a, **kw): return await _provider.generate_image(*a, **kw)
async def transcribe_audio(*a, **kw): return await _provider.transcribe_audio(*a, **kw)
async def synthesize_speech(*a, **kw): return await _provider.synthesize_speech(*a, **kw)

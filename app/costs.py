"""Cost tracking for LLM API calls."""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CostEntry:
    category: str  # "prompt" | "image_generation" | "image_input" | "voice_input" | "voice_output"
    provider: str  # "openai" | "google" | "anthropic"
    model: str
    function: str
    cost_usd: float
    detail: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SpendingLimitExceeded(Exception):
    def __init__(self, limit: float, current: float, attempted: float):
        self.limit = limit
        self.current = current
        self.attempted = attempted
        super().__init__(
            f"Spending limit ${limit:.4f} would be exceeded: "
            f"current ${current:.4f} + attempted ${attempted:.4f} = ${current + attempted:.4f}"
        )


# --- Pricing tables ---

# gpt-image-1: {(quality, size): cost}
OPENAI_IMAGE_PRICING: dict[tuple[str, str], float] = {
    ("low", "1024x1024"): 0.011,
    ("low", "other"): 0.016,
    ("medium", "1024x1024"): 0.042,
    ("medium", "other"): 0.063,
    ("high", "1024x1024"): 0.167,
    ("high", "other"): 0.250,
}

# Chat models: {model: (input_per_token, output_per_token)}
CHAT_TOKEN_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.2": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4.1-nano": (0.10 / 1_000_000, 0.40 / 1_000_000),
    "claude-opus-4-6": (15.00 / 1_000_000, 75.00 / 1_000_000),
}

# Audio pricing
OPENAI_TRANSCRIPTION_PER_MINUTE = 0.003  # gpt-4o-mini-transcribe
OPENAI_TTS_PER_CHAR = 0.015 / 1000  # gpt-4o-mini-tts per character

# Google flat image cost
GOOGLE_IMAGE_COST = 0.04


class CostTracker:
    def __init__(self) -> None:
        self._entries: list[CostEntry] = []
        self._limit_usd: float | None = None
        self._lock = threading.Lock()

    def record(self, entry: CostEntry) -> None:
        with self._lock:
            if self._limit_usd is not None:
                if self.total_usd + entry.cost_usd > self._limit_usd:
                    raise SpendingLimitExceeded(self._limit_usd, self.total_usd, entry.cost_usd)
            self._entries.append(entry)

    def check_limit(self, estimated_cost: float) -> None:
        with self._lock:
            if self._limit_usd is not None:
                if self.total_usd + estimated_cost > self._limit_usd:
                    raise SpendingLimitExceeded(self._limit_usd, self.total_usd, estimated_cost)

    @property
    def total_usd(self) -> float:
        return sum(e.cost_usd for e in self._entries)

    def totals_by_category(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self._entries:
            result[e.category] = result.get(e.category, 0.0) + e.cost_usd
        return result

    def totals_by_provider(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self._entries:
            result[e.provider] = result.get(e.provider, 0.0) + e.cost_usd
        return result

    @property
    def entries(self) -> list[CostEntry]:
        return list(self._entries)

    @property
    def limit_usd(self) -> float | None:
        return self._limit_usd

    @limit_usd.setter
    def limit_usd(self, value: float | None) -> None:
        self._limit_usd = value

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._limit_usd = None


# Module singleton
tracker = CostTracker()


# --- Record helper functions ---

def record_openai_image(quality: str, size: str, function: str = "generate_image") -> CostEntry:
    effective_quality = quality if quality != "auto" else "medium"
    key = (effective_quality, size if size == "1024x1024" else "other")
    cost = OPENAI_IMAGE_PRICING.get(key, OPENAI_IMAGE_PRICING[("medium", "other")])
    entry = CostEntry(
        category="image_generation",
        provider="openai",
        model="gpt-image-1",
        function=function,
        cost_usd=cost,
        detail={"quality": quality, "effective_quality": effective_quality, "size": size},
    )
    tracker.record(entry)
    return entry


def record_openai_chat(
    model: str,
    function: str,
    usage: dict,
    num_images: int = 0,
) -> list[CostEntry]:
    pricing = CHAT_TOKEN_PRICING.get(model)
    if not pricing:
        return []
    input_price, output_price = pricing
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_cost = prompt_tokens * input_price + completion_tokens * output_price

    entries: list[CostEntry] = []

    if num_images > 0:
        estimated_image_tokens = num_images * 800
        image_token_cost = estimated_image_tokens * input_price
        image_cost = min(image_token_cost, total_cost)
        prompt_cost = total_cost - image_cost
        entries.append(CostEntry(
            category="image_input",
            provider="openai",
            model=model,
            function=function,
            cost_usd=image_cost,
            detail={"num_images": num_images, "estimated_image_tokens": estimated_image_tokens},
        ))
        tracker.record(entries[-1])
        if prompt_cost > 0:
            entries.append(CostEntry(
                category="prompt",
                provider="openai",
                model=model,
                function=function,
                cost_usd=prompt_cost,
                detail={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            ))
            tracker.record(entries[-1])
    else:
        entries.append(CostEntry(
            category="prompt",
            provider="openai",
            model=model,
            function=function,
            cost_usd=total_cost,
            detail={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        ))
        tracker.record(entries[-1])

    return entries


def record_openai_transcription(duration_seconds: float) -> CostEntry:
    cost = (duration_seconds / 60.0) * OPENAI_TRANSCRIPTION_PER_MINUTE
    entry = CostEntry(
        category="voice_input",
        provider="openai",
        model="gpt-4o-mini-transcribe",
        function="transcribe_audio",
        cost_usd=cost,
        detail={"duration_seconds": duration_seconds},
    )
    tracker.record(entry)
    return entry


def record_openai_tts(char_count: int) -> CostEntry:
    cost = char_count * OPENAI_TTS_PER_CHAR
    entry = CostEntry(
        category="voice_output",
        provider="openai",
        model="gpt-4o-mini-tts",
        function="synthesize_speech",
        cost_usd=cost,
        detail={"char_count": char_count},
    )
    tracker.record(entry)
    return entry


def record_azure_image(quality: str, size: str, function: str = "generate_image") -> CostEntry:
    effective_quality = quality if quality != "auto" else "medium"
    key = (effective_quality, size if size == "1024x1024" else "other")
    cost = OPENAI_IMAGE_PRICING.get(key, OPENAI_IMAGE_PRICING[("medium", "other")])
    entry = CostEntry(
        category="image_generation",
        provider="azure_openai",
        model="gpt-image-1",
        function=function,
        cost_usd=cost,
        detail={"quality": quality, "effective_quality": effective_quality, "size": size},
    )
    tracker.record(entry)
    return entry


def record_google_image() -> CostEntry:
    entry = CostEntry(
        category="image_generation",
        provider="google",
        model="gemini-3-pro-image-preview",
        function="generate_image",
        cost_usd=GOOGLE_IMAGE_COST,
        detail={},
    )
    tracker.record(entry)
    return entry


def record_anthropic_chat(
    model: str,
    function: str,
    usage: dict,
    num_images: int = 0,
) -> list[CostEntry]:
    pricing = CHAT_TOKEN_PRICING.get(model)
    if not pricing:
        return []
    input_price, output_price = pricing
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_cost = input_tokens * input_price + output_tokens * output_price

    entries: list[CostEntry] = []

    if num_images > 0:
        estimated_image_tokens = num_images * 1600
        image_token_cost = estimated_image_tokens * input_price
        image_cost = min(image_token_cost, total_cost)
        prompt_cost = total_cost - image_cost
        entries.append(CostEntry(
            category="image_input",
            provider="anthropic",
            model=model,
            function=function,
            cost_usd=image_cost,
            detail={"num_images": num_images, "estimated_image_tokens": estimated_image_tokens},
        ))
        tracker.record(entries[-1])
        if prompt_cost > 0:
            entries.append(CostEntry(
                category="prompt",
                provider="anthropic",
                model=model,
                function=function,
                cost_usd=prompt_cost,
                detail={"input_tokens": input_tokens, "output_tokens": output_tokens},
            ))
            tracker.record(entries[-1])
    else:
        entries.append(CostEntry(
            category="prompt",
            provider="anthropic",
            model=model,
            function=function,
            cost_usd=total_cost,
            detail={"input_tokens": input_tokens, "output_tokens": output_tokens},
        ))
        tracker.record(entries[-1])

    return entries

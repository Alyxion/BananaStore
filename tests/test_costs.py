"""Unit tests for app.costs module."""

import pytest

from app.costs import (
    CHAT_TOKEN_PRICING,
    GOOGLE_IMAGE_COST,
    OPENAI_IMAGE_PRICING,
    OPENAI_TRANSCRIPTION_PER_MINUTE,
    OPENAI_TTS_PER_CHAR,
    CostEntry,
    CostTracker,
    SpendingLimitExceeded,
    record_anthropic_chat,
    record_google_image,
    record_openai_chat,
    record_openai_image,
    record_openai_transcription,
    record_openai_tts,
    tracker,
)


# --- CostTracker ---


class TestCostTracker:
    def test_initial_state(self):
        t = CostTracker()
        assert t.total_usd == 0.0
        assert t.entries == []
        assert t.limit_usd is None

    def test_record_adds_entry(self):
        entry = CostEntry("prompt", "openai", "gpt-5.2", "test", 0.05, {})
        tracker.record(entry)
        assert len(tracker.entries) == 1
        assert tracker.total_usd == 0.05

    def test_record_multiple(self):
        tracker.record(CostEntry("prompt", "openai", "gpt-5.2", "a", 0.10, {}))
        tracker.record(CostEntry("image_generation", "google", "gemini", "b", 0.04, {}))
        assert len(tracker.entries) == 2
        assert abs(tracker.total_usd - 0.14) < 1e-9

    def test_total_usd(self):
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.01, {}))
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.02, {}))
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.03, {}))
        assert abs(tracker.total_usd - 0.06) < 1e-9

    def test_totals_by_category(self):
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.10, {}))
        tracker.record(CostEntry("image_generation", "openai", "m", "f", 0.05, {}))
        tracker.record(CostEntry("prompt", "anthropic", "m", "f", 0.20, {}))
        result = tracker.totals_by_category()
        assert abs(result["prompt"] - 0.30) < 1e-9
        assert abs(result["image_generation"] - 0.05) < 1e-9

    def test_totals_by_provider(self):
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.10, {}))
        tracker.record(CostEntry("prompt", "anthropic", "m", "f", 0.20, {}))
        tracker.record(CostEntry("image_generation", "openai", "m", "f", 0.05, {}))
        result = tracker.totals_by_provider()
        assert abs(result["openai"] - 0.15) < 1e-9
        assert abs(result["anthropic"] - 0.20) < 1e-9

    def test_entries_returns_copy(self):
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.01, {}))
        entries = tracker.entries
        entries.clear()
        assert len(tracker.entries) == 1

    def test_limit_usd_setter(self):
        tracker.limit_usd = 1.00
        assert tracker.limit_usd == 1.00
        tracker.limit_usd = None
        assert tracker.limit_usd is None

    def test_limit_enforcement_on_record(self):
        tracker.limit_usd = 0.10
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.05, {}))
        with pytest.raises(SpendingLimitExceeded):
            tracker.record(CostEntry("prompt", "openai", "m", "f", 0.06, {}))
        # Entry should not have been added
        assert len(tracker.entries) == 1
        assert tracker.total_usd == 0.05

    def test_limit_exact_boundary(self):
        tracker.limit_usd = 0.10
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.10, {}))
        assert tracker.total_usd == 0.10
        with pytest.raises(SpendingLimitExceeded):
            tracker.record(CostEntry("prompt", "openai", "m", "f", 0.001, {}))

    def test_no_limit_allows_any(self):
        tracker.record(CostEntry("prompt", "openai", "m", "f", 999.99, {}))
        assert tracker.total_usd == 999.99

    def test_check_limit_passes(self):
        tracker.limit_usd = 1.00
        tracker.check_limit(0.50)  # should not raise

    def test_check_limit_raises(self):
        tracker.limit_usd = 0.10
        tracker.record(CostEntry("prompt", "openai", "m", "f", 0.09, {}))
        with pytest.raises(SpendingLimitExceeded):
            tracker.check_limit(0.02)

    def test_check_limit_no_limit(self):
        tracker.check_limit(1000.0)  # should not raise

    def test_reset(self):
        tracker.limit_usd = 5.00
        tracker.record(CostEntry("prompt", "openai", "m", "f", 1.00, {}))
        tracker.reset()
        assert tracker.total_usd == 0.0
        assert tracker.entries == []
        assert tracker.limit_usd is None


# --- SpendingLimitExceeded ---


class TestSpendingLimitExceeded:
    def test_attributes(self):
        exc = SpendingLimitExceeded(limit=1.00, current=0.80, attempted=0.30)
        assert exc.limit == 1.00
        assert exc.current == 0.80
        assert exc.attempted == 0.30

    def test_message_format(self):
        exc = SpendingLimitExceeded(limit=1.00, current=0.80, attempted=0.30)
        msg = str(exc)
        assert "$1.00" in msg or "1.0" in msg
        assert "exceeded" in msg.lower()


# --- record_openai_image ---


class TestRecordOpenaiImage:
    def test_low_1024x1024(self):
        entry = record_openai_image("low", "1024x1024")
        assert entry.cost_usd == 0.011
        assert entry.category == "image_generation"
        assert entry.provider == "openai"
        assert entry.model == "gpt-image-1"

    def test_low_other_size(self):
        entry = record_openai_image("low", "1536x1024")
        assert entry.cost_usd == 0.016

    def test_medium_1024x1024(self):
        entry = record_openai_image("medium", "1024x1024")
        assert entry.cost_usd == 0.042

    def test_medium_other(self):
        entry = record_openai_image("medium", "1024x1536")
        assert entry.cost_usd == 0.063

    def test_high_1024x1024(self):
        entry = record_openai_image("high", "1024x1024")
        assert entry.cost_usd == 0.167

    def test_high_other(self):
        entry = record_openai_image("high", "1536x1024")
        assert entry.cost_usd == 0.250

    def test_auto_maps_to_medium(self):
        entry = record_openai_image("auto", "1024x1024")
        assert entry.cost_usd == 0.042
        assert entry.detail["effective_quality"] == "medium"

    def test_detail_fields(self):
        entry = record_openai_image("high", "1024x1024")
        assert "quality" in entry.detail
        assert "size" in entry.detail

    def test_adds_to_tracker(self):
        record_openai_image("low", "1024x1024")
        assert tracker.total_usd == 0.011


# --- record_openai_chat ---


class TestRecordOpenaiChat:
    def test_prompt_only(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 200}
        entries = record_openai_chat("gpt-4.1-nano", "suggest_filename", usage)
        assert len(entries) == 1
        assert entries[0].category == "prompt"
        input_price, output_price = CHAT_TOKEN_PRICING["gpt-4.1-nano"]
        expected = 100 * input_price + 200 * output_price
        assert abs(entries[0].cost_usd - expected) < 1e-12

    def test_with_images_splits_cost(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}
        entries = record_openai_chat("gpt-5.2", "generate_svg", usage, num_images=2)
        categories = {e.category for e in entries}
        assert "image_input" in categories

    def test_unknown_model_returns_empty(self):
        entries = record_openai_chat("unknown-model", "test", {"prompt_tokens": 10, "completion_tokens": 10})
        assert entries == []

    def test_prompt_detail_fields(self):
        usage = {"prompt_tokens": 50, "completion_tokens": 100}
        entries = record_openai_chat("gpt-4.1-nano", "test", usage)
        assert entries[0].detail["prompt_tokens"] == 50
        assert entries[0].detail["completion_tokens"] == 100


# --- record_openai_transcription ---


class TestRecordOpenaiTranscription:
    def test_cost_calculation(self):
        entry = record_openai_transcription(60.0)  # 1 minute
        assert abs(entry.cost_usd - OPENAI_TRANSCRIPTION_PER_MINUTE) < 1e-12
        assert entry.category == "voice_input"
        assert entry.model == "gpt-4o-mini-transcribe"

    def test_partial_minute(self):
        entry = record_openai_transcription(30.0)  # half minute
        expected = 0.5 * OPENAI_TRANSCRIPTION_PER_MINUTE
        assert abs(entry.cost_usd - expected) < 1e-12

    def test_detail_fields(self):
        entry = record_openai_transcription(45.0)
        assert entry.detail["duration_seconds"] == 45.0


# --- record_openai_tts ---


class TestRecordOpenaiTts:
    def test_cost_calculation(self):
        entry = record_openai_tts(1000)
        expected = 1000 * OPENAI_TTS_PER_CHAR
        assert abs(entry.cost_usd - expected) < 1e-12
        assert entry.category == "voice_output"
        assert entry.model == "tts-1-hd"

    def test_short_text(self):
        entry = record_openai_tts(5)
        expected = 5 * OPENAI_TTS_PER_CHAR
        assert abs(entry.cost_usd - expected) < 1e-12

    def test_detail_fields(self):
        entry = record_openai_tts(42)
        assert entry.detail["char_count"] == 42


# --- record_google_image ---


class TestRecordGoogleImage:
    def test_flat_cost(self):
        entry = record_google_image()
        assert entry.cost_usd == GOOGLE_IMAGE_COST
        assert entry.category == "image_generation"
        assert entry.provider == "google"
        assert entry.model == "gemini-3-pro-image-preview"


# --- record_anthropic_chat ---


class TestRecordAnthropicChat:
    def test_prompt_only(self):
        usage = {"input_tokens": 100, "output_tokens": 200}
        entries = record_anthropic_chat("claude-opus-4-6", "generate_svg", usage)
        assert len(entries) == 1
        assert entries[0].category == "prompt"
        input_price, output_price = CHAT_TOKEN_PRICING["claude-opus-4-6"]
        expected = 100 * input_price + 200 * output_price
        assert abs(entries[0].cost_usd - expected) < 1e-12

    def test_with_images(self):
        usage = {"input_tokens": 2000, "output_tokens": 500}
        entries = record_anthropic_chat("claude-opus-4-6", "generate_svg", usage, num_images=1)
        categories = {e.category for e in entries}
        assert "image_input" in categories

    def test_unknown_model(self):
        entries = record_anthropic_chat("unknown", "test", {"input_tokens": 10, "output_tokens": 10})
        assert entries == []

    def test_uses_1600_tokens_per_image(self):
        usage = {"input_tokens": 5000, "output_tokens": 1000}
        entries = record_anthropic_chat("claude-opus-4-6", "generate_svg", usage, num_images=1)
        image_entry = [e for e in entries if e.category == "image_input"][0]
        input_price, _ = CHAT_TOKEN_PRICING["claude-opus-4-6"]
        expected_image_cost = 1600 * input_price
        assert abs(image_entry.cost_usd - expected_image_cost) < 1e-12


# --- CostEntry ---


class TestCostEntry:
    def test_timestamp_auto_set(self):
        entry = CostEntry("prompt", "openai", "m", "f", 0.01, {})
        assert entry.timestamp is not None
        assert "T" in entry.timestamp  # ISO format

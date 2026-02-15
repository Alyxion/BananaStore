import pytest
from fastapi import HTTPException

from app.svg import (
    SVG_MAX_TOKENS,
    SVG_QUALITY_HINTS,
    SVG_SYSTEM_PROMPT,
    extract_svg,
    parse_svg_dimensions,
)


# --- parse_svg_dimensions ---

class TestParseSvgDimensions:
    def test_standard_size(self):
        assert parse_svg_dimensions("1024x1024") == (1024, 1024)

    def test_landscape(self):
        assert parse_svg_dimensions("1536x1024") == (1536, 1024)

    def test_portrait(self):
        assert parse_svg_dimensions("1024x1536") == (1024, 1536)


# --- extract_svg ---

class TestExtractSvg:
    def test_extracts_clean_svg(self):
        text = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect/></svg>'
        result = extract_svg(text)
        assert result.startswith("<svg")
        assert result.endswith("</svg>")

    def test_extracts_from_surrounding_text(self):
        text = 'Here is the SVG:\n<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>\nDone.'
        result = extract_svg(text)
        assert "<circle/>" in result

    def test_extracts_from_markdown_fence(self):
        text = '```xml\n<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>\n```'
        result = extract_svg(text)
        assert "<rect/>" in result

    def test_adds_xmlns_when_missing(self):
        text = "<svg viewBox=\"0 0 100 100\"><rect/></svg>"
        result = extract_svg(text)
        assert 'xmlns="http://www.w3.org/2000/svg"' in result

    def test_preserves_existing_xmlns(self):
        text = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        result = extract_svg(text)
        assert result.count("xmlns") == 1

    def test_raises_on_no_svg(self):
        with pytest.raises(HTTPException) as exc_info:
            extract_svg("no svg here at all")
        assert exc_info.value.status_code == 502

    def test_raises_on_empty_string(self):
        with pytest.raises(HTTPException):
            extract_svg("")

    def test_case_insensitive(self):
        text = '<SVG xmlns="http://www.w3.org/2000/svg"><rect/></SVG>'
        result = extract_svg(text)
        assert "<rect/>" in result


# --- Constants ---

class TestSvgConstants:
    def test_quality_hints_has_all_levels(self):
        assert set(SVG_QUALITY_HINTS.keys()) == {"low", "medium", "high"}

    def test_max_tokens_is_reasonable(self):
        assert SVG_MAX_TOKENS >= 8000

    def test_system_prompt_has_placeholders(self):
        assert "{width}" in SVG_SYSTEM_PROMPT
        assert "{height}" in SVG_SYSTEM_PROMPT
        assert "{quality_hint}" in SVG_SYSTEM_PROMPT

    def test_system_prompt_formats_without_error(self):
        result = SVG_SYSTEM_PROMPT.format(width=1024, height=768, quality_hint="test hint")
        assert "1024" in result
        assert "768" in result
        assert "test hint" in result

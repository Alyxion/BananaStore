import pytest

from app.util import fallback_filename, sanitize_filename, read_reference_images
from tests.conftest import make_upload_file


# --- fallback_filename ---

class TestFallbackFilename:
    def test_simple_description(self):
        assert fallback_filename("A red cat") == "a-red-cat"

    def test_special_characters_stripped(self):
        assert fallback_filename("Hello! World? #1") == "hello-world-1"

    def test_unicode_normalized(self):
        assert fallback_filename("café résumé") == "cafe-resume"

    def test_empty_string(self):
        assert fallback_filename("") == "generated-image"

    def test_only_special_chars(self):
        assert fallback_filename("!!!???") == "generated-image"

    def test_truncated_at_80_chars(self):
        long_desc = "a " * 100
        result = fallback_filename(long_desc)
        assert len(result) <= 80

    def test_multiple_spaces_collapsed(self):
        assert fallback_filename("hello    world") == "hello-world"

    def test_leading_trailing_hyphens_stripped(self):
        assert fallback_filename("- hello -") == "hello"


# --- sanitize_filename ---

class TestSanitizeFilename:
    def test_basic(self):
        assert sanitize_filename("My Cool Image", "fallback") == "my-cool-image"

    def test_strips_png_extension(self):
        assert sanitize_filename("image.png", "fallback") == "image"

    def test_strips_svg_extension(self):
        assert sanitize_filename("drawing.svg", "fallback") == "drawing"

    def test_empty_returns_fallback(self):
        assert sanitize_filename("", "my-fallback") == "my-fallback"

    def test_only_special_chars_returns_fallback(self):
        assert sanitize_filename("@#$%", "my-fallback") == "my-fallback"

    def test_truncated_at_80(self):
        long_name = "a" * 200
        result = sanitize_filename(long_name, "fb")
        assert len(result) <= 80

    def test_unicode_normalized(self):
        assert sanitize_filename("Über Cool.png", "fb") == "uber-cool"


# --- read_reference_images ---

class TestReadReferenceImages:
    @pytest.mark.asyncio
    async def test_none_input(self):
        parsed, svgs = await read_reference_images(None)
        assert parsed == []
        assert svgs == []

    @pytest.mark.asyncio
    async def test_empty_list(self):
        parsed, svgs = await read_reference_images([])
        assert parsed == []
        assert svgs == []

    @pytest.mark.asyncio
    async def test_raster_image(self):
        upload = make_upload_file(b"pngbytes", "photo.png", "image/png")
        parsed, svgs = await read_reference_images([upload])
        assert len(parsed) == 1
        assert parsed[0] == ("photo.png", b"pngbytes", "image/png")
        assert svgs == []

    @pytest.mark.asyncio
    async def test_svg_by_mime(self):
        svg_content = b"<svg>test</svg>"
        upload = make_upload_file(svg_content, "icon.svg", "image/svg+xml")
        parsed, svgs = await read_reference_images([upload])
        assert parsed == []
        assert len(svgs) == 1
        assert svgs[0] == "<svg>test</svg>"

    @pytest.mark.asyncio
    async def test_svg_by_extension(self):
        svg_content = b"<svg>ext</svg>"
        upload = make_upload_file(svg_content, "icon.svg", "application/octet-stream")
        parsed, svgs = await read_reference_images([upload])
        assert parsed == []
        assert len(svgs) == 1

    @pytest.mark.asyncio
    async def test_empty_file_skipped(self):
        upload = make_upload_file(b"", "empty.png", "image/png")
        parsed, svgs = await read_reference_images([upload])
        assert parsed == []
        assert svgs == []

    @pytest.mark.asyncio
    async def test_mixed_files(self):
        png = make_upload_file(b"pngdata", "a.png", "image/png")
        svg = make_upload_file(b"<svg/>", "b.svg", "image/svg+xml")
        jpg = make_upload_file(b"jpgdata", "c.jpg", "image/jpeg")
        parsed, svgs = await read_reference_images([png, svg, jpg])
        assert len(parsed) == 2
        assert len(svgs) == 1

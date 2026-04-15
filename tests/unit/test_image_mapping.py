"""Unit tests for ImageNameMapper.

Covers the specific reason this class exists: filenames sent to a VLM
burn context tokens, so we swap them for '1.png', '2.png', …, run
inference, then map back. Bugs here corrupt every downstream result.
"""
from app.utils.image_mapping import ImageNameMapper


def test_roundtrip_preserves_extension():
    m = ImageNameMapper()
    s = m.add_mapping("IMG_weird_very_long_name.JPEG")
    # Counter-based simple names keep the original lowercase extension
    assert s.endswith(".jpeg")
    assert m.get_original_name(s) == "IMG_weird_very_long_name.JPEG"


def test_unknown_extension_defaults_to_png():
    m = ImageNameMapper()
    s = m.add_mapping("file_no_ext")
    assert s.endswith(".png")


def test_repeat_add_is_idempotent():
    m = ImageNameMapper()
    a = m.add_mapping("a.jpg")
    b = m.add_mapping("a.jpg")
    assert a == b
    assert m.counter == 1


def test_get_original_by_stem_fallback():
    # Model sometimes returns "1" without extension; we still resolve.
    m = ImageNameMapper()
    m.add_mapping("original.png")
    assert m.get_original_name("1") == "original.png"


def test_reset_clears_state():
    m = ImageNameMapper()
    m.add_mapping("a.jpg")
    m.reset()
    assert m.counter == 0
    assert m.original_to_simple == {}

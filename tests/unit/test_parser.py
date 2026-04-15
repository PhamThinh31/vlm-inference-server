"""Unit tests for app.utils.parser.

These cover the parsing edge cases we've actually hit in the wild — not
happy paths. The boolean parser used to be a substring match that
returned True for "Yesterday..." because 'yes' is in 'yesterday'; the
JSON extractor has three fallback strategies and each has failed at
least once against a real VLM response.
"""
import pytest

from app.utils.parser import (
    extract_boolean_from_response,
    extract_json_from_response,
    extract_key_value_pairs,
    safe_parse_json,
)


class TestBoolean:
    def test_yes_plain(self):
        assert extract_boolean_from_response("Yes, the pair matches.") is True

    def test_no_plain(self):
        assert extract_boolean_from_response("No, the angles differ.") is False

    @pytest.mark.parametrize("text", [
        "Yesterday we observed the front view",
        "Yemen was mentioned",
        "yesno",  # adjacent, no boundary
    ])
    def test_yes_word_boundary(self, text):
        # Word boundary must block partial-word false positives.
        assert extract_boolean_from_response(text) is False

    def test_invalid_does_not_match_valid(self):
        assert extract_boolean_from_response("The pair is invalid.") is False

    def test_valid_matches(self):
        assert extract_boolean_from_response("Pair is valid for VTON") is True

    def test_both_yes_no_prefers_first(self):
        # "Yes, this pair works. No issues detected." -> True (yes first)
        assert extract_boolean_from_response(
            "Yes, this pair works. No issues detected."
        ) is True

    def test_empty(self):
        assert extract_boolean_from_response("") is False


class TestExtractJson:
    def test_clean_json(self):
        assert extract_json_from_response('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        blob = 'Here it is:\n```json\n{"x": [1,2,3]}\n```\nthanks'
        assert extract_json_from_response(blob) == {"x": [1, 2, 3]}

    def test_nested_substring_fallback(self):
        # Not fenced, surrounded by prose — substring strategy kicks in.
        blob = 'preamble {"outer": {"inner": true}} trailing'
        assert extract_json_from_response(blob) == {"outer": {"inner": True}}

    def test_returns_none_for_garbage(self):
        assert extract_json_from_response("no json here") is None

    def test_returns_none_for_empty(self):
        assert extract_json_from_response("") is None


class TestSafeParse:
    def test_ok(self):
        assert safe_parse_json('[1,2]') == [1, 2]

    def test_default_on_failure(self):
        assert safe_parse_json("not json", default={}) == {}


class TestKeyValue:
    def test_basic_pairs(self):
        out = extract_key_value_pairs("Category: Shirt\nColor: Red")
        assert out["category"] == "Shirt"
        assert out["color"] == "Red"

    def test_markdown_bold_keys(self):
        out = extract_key_value_pairs("**Category:** Pants")
        assert out.get("category") == "Pants"

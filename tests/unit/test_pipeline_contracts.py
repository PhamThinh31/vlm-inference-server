"""Contract tests: fabricated-data sentinels must not leak downstream.

These are not happy-path tests. They enforce the invariants the
data-quality review identified:

1. Task 2 never returns `category="error"` or `category="unknown"` as a
   masquerade for a failed call. Failure modes populate `error` and
   leave `category=None`.
2. Task 3 never fabricates a `confidence` value. When the model did not
   report one, `confidence is None` — downstream aggregations depend on
   this so they can exclude unreported rows instead of averaging 0.5s.
3. Task 1's response parser does not positionally map classifications
   to image IDs. A response that omits or renames a key produces
   parse_failed rows — it does NOT silently shift labels.
"""
from __future__ import annotations

from app.pipelines.task1_classifier import Task1Classifier
from app.pipelines.task2_attributes import Task2AttributesExtractor
from app.pipelines.task3_validation import Task3Validator
from app.schemas import ErrorCode


class _FakeImg:
    size = (512, 512)


class TestTask1NoPositionalFallback:
    def test_missing_key_marks_parse_failed(self):
        # Response names only image_a; image_b is absent. The previous
        # positional fallback would have assigned image_a's result to
        # image_b. That must not happen.
        classifier = Task1Classifier(version="v1.0")
        response = '{"image_a.jpg": {"class": "garment"}}'
        ids = ["image_a.jpg", "image_b.jpg"]
        imgs = [_FakeImg(), _FakeImg()]
        out = classifier._parse_response(response, ids, imgs)

        by_name = {c.filename: c for c in out}
        assert by_name["image_a.jpg"].img_class.value == "garment"
        assert by_name["image_a.jpg"].parse_failed is False
        assert by_name["image_b.jpg"].parse_failed is True
        assert by_name["image_b.jpg"].img_class.value == "unknown"

    def test_basename_match_still_works(self):
        # Exact match fails but basename match succeeds — legitimate.
        classifier = Task1Classifier(version="v1.0")
        response = '{"042": {"class": "body"}}'
        out = classifier._parse_response(response, ["042.jpg"], [_FakeImg()])
        assert out[0].img_class.value == "body"
        assert out[0].parse_failed is False


class TestTask2NoMasquerade:
    def test_parse_failure_sets_error_not_unknown(self):
        extractor = Task2AttributesExtractor(version="v1.1")
        # No Category field, no JSON Attributes field — pure garbage.
        result = extractor._parse_response("garbage response", "g.jpg", "b.jpg")
        assert result.category is None, "must NOT default to 'unknown'"
        assert result.error is not None
        assert result.error.code == ErrorCode.PARSE_FAILED
        # detail is human-readable; don't substring-match it elsewhere,
        # but the parser SHOULD name which fields were missing.
        assert result.error.detail and "category" in result.error.detail

    def test_successful_parse_has_no_error(self):
        extractor = Task2AttributesExtractor(version="v1.1")
        response = (
            "**Category:** Shirt\n"
            "**Attribute Description:** Red cotton\n"
            "**JSON Attributes:** {\"color\": \"red\"}"
        )
        result = extractor._parse_response(response, "g.jpg", "b.jpg")
        assert result.category == "Shirt"
        assert result.attributes == {"color": "red"}
        assert result.error is None


class TestTask3NoFabricatedConfidence:
    def test_missing_confidence_stays_none(self):
        validator = Task3Validator(version="v1.1")
        # Valid Pair field present but no Pose Quality / Confidence.
        response = "Valid Pair: Yes"
        result = validator._parse_response(response, "g.jpg", "b.jpg")
        assert result.is_valid is True
        assert result.confidence is None, "must NOT fabricate 0.5"
        assert result.parse_failed is False

    def test_parse_failure_has_none_confidence_and_error(self):
        validator = Task3Validator(version="v1.1")
        result = validator._parse_response("no verdict here", "g.jpg", "b.jpg")
        assert result.is_valid is False
        assert result.confidence is None
        assert result.parse_failed is True
        assert result.error is not None
        assert result.error.code == ErrorCode.PARSE_FAILED

    def test_pose_quality_maps_to_confidence(self):
        validator = Task3Validator(version="v1.1")
        response = "Pose Quality: 80\nValid Pair: Yes"
        result = validator._parse_response(response, "g.jpg", "b.jpg")
        assert result.pose_quality == 80.0
        assert result.confidence == 0.8

# app/schemas.py
"""
Pydantic models for request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ImageClass(str, Enum):
    GARMENT = "garment"
    BODY = "body"
    UNKNOWN = "unknown"


class GarmentView(str, Enum):
    FRONT = "front"
    BACK = "back"
    UNKNOWN = "unknown"


class BodyCrop(str, Enum):
    FULL_BODY = "full_body"
    HALF_TOP = "half_top"
    HALF_BOTTOM = "half_bottom"


class BodyAngle(str, Enum):
    BACK = "back"
    BACK_RIGHT = "back-right"
    RIGHT_SIDE = "right side"
    FRONT_RIGHT = "front-right"
    FRONT = "front"
    FRONT_LEFT = "front-left"
    LEFT_SIDE = "left side"
    BACK_LEFT = "back-left"


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

class ErrorCode(str, Enum):
    """Closed set of machine-readable failure modes.

    Downstream consumers branch on `code`, never on `detail`. Adding a
    new code is a deliberate schema change: update this enum, update
    the runbook, and tell the modeling team. Renaming a code is a
    breaking change; don't.
    """
    PARSE_FAILED = "parse_failed"
    MISSING_IMAGE = "missing_image"
    INFERENCE_FAILURE = "inference_failure"
    PROCESSING_ERROR = "processing_error"
    TIMEOUT = "timeout"


class TaskError(BaseModel):
    """Structured error carried on task result rows.

    `code` is the machine-readable bucket; `detail` is a free-text
    human hint for logs. Downstream joins and alerts use `code`; if
    you find yourself substring-matching `detail`, that's a bug — add
    a new ErrorCode instead.
    """
    code: ErrorCode
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Shared image models
# ---------------------------------------------------------------------------

class ImageData(BaseModel):
    """Single image payload (base64-encoded)."""
    filename: str
    data: str


class ImagePairData(BaseModel):
    """Garment + body pair payload (base64-encoded)."""
    garment: ImageData
    body: ImageData


# ---------------------------------------------------------------------------
# Task 1 — Classification
# ---------------------------------------------------------------------------

class ImageClassification(BaseModel):
    """Task 1 output for a single image.

    `parse_failed` is the explicit contract: a classification with
    img_class=UNKNOWN and parse_failed=True is a degraded fallback, not a
    genuine model judgement. Downstream tasks MUST treat these rows as
    missing inputs rather than feeding them into pairing.
    """
    filename: str
    img_class: ImageClass = Field(alias="class")
    view: Optional[GarmentView] = None
    body_crop: Optional[BodyCrop] = None
    angle: Optional[BodyAngle] = None
    resolution: Optional[tuple[int, int]] = None
    priority_score: Optional[float] = None
    simple_name: Optional[str] = None
    parse_failed: bool = False

    model_config = ConfigDict(populate_by_name=True)


class Task1Request(BaseModel):
    """Request for Task 1 classification."""
    session_id: Optional[str] = None
    image_files: Optional[list[str]] = None
    image_paths_list: Optional[list[str]] = None
    image_data_list: Optional[list[ImageData]] = None

    @model_validator(mode="after")
    def _require_at_least_one_input(self) -> "Task1Request":
        # Treat empty lists as "not provided" — a client who posts
        # image_data_list=[] should get a clear error, not a 500 later.
        if not any(bool(x) for x in (self.session_id, self.image_paths_list, self.image_data_list)):
            raise ValueError(
                "Provide at least one of: session_id, image_paths_list, image_data_list"
            )
        return self


class Task1Response(BaseModel):
    """Response for Task 1 classification."""
    session_id: Optional[str] = None
    classifications: list[ImageClassification]
    processing_time: float
    total_images: int
    raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# Task 2 — Attribute Extraction
# ---------------------------------------------------------------------------

class Task2Result(BaseModel):
    """Task 2 output for a garment-body pair.

    `category` is Optional: when the model produced a usable response,
    it is a real string; when extraction or inference failed, it is
    None and `error` describes the failure mode. Downstream code must
    branch on `error is None` rather than treating `category == "error"`
    or `category == "unknown"` as sentinels — both are now explicitly
    forbidden as masquerade values.
    """
    garment_file: str
    body_file: str
    raw_response: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: Optional[TaskError] = None


class Task2Request(BaseModel):
    """Request for Task 2 attribute extraction."""
    session_id: Optional[str] = None
    pairs: Optional[list[tuple[str, str]]] = None
    image_pairs: Optional[list[tuple[str, str]]] = None
    image_data_pairs: Optional[list[ImagePairData]] = None

    @model_validator(mode="after")
    def _require_at_least_one_input(self) -> "Task2Request":
        if not any(bool(x) for x in (self.session_id, self.image_pairs, self.image_data_pairs)):
            raise ValueError(
                "Provide at least one of: session_id, image_pairs, image_data_pairs"
            )
        return self


class Task2Response(BaseModel):
    """Response for Task 2."""
    session_id: Optional[str] = None
    results: list[Task2Result]
    processing_time: float
    pairs_processed: int


# ---------------------------------------------------------------------------
# Task 3 — Pair Validation
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Task 3 validation result.

    `parse_failed` distinguishes a real "invalid pair" judgement from a
    parser fallback. An aggregate validation_rate computed without
    filtering parse_failed=True rows will understate quality issues.

    `confidence` is Optional: None means "the model did not express a
    confidence." Previously we defaulted to 0.5 which was indistinguishable
    from a genuine 50% — a fabricated probability that flowed into fleet
    metrics as if real.
    """
    garment_file: str
    body_file: str
    is_valid: bool
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    pose_quality: Optional[float] = None
    parse_failed: bool = False
    error: Optional[TaskError] = None


class Task3Request(BaseModel):
    """Request for Task 3 validation."""
    session_id: Optional[str] = None
    pairs: Optional[list[tuple[str, str]]] = None
    image_pairs: Optional[list[tuple[str, str]]] = None
    image_data_pairs: Optional[list[ImagePairData]] = None
    max_pairs: int = 10

    @model_validator(mode="after")
    def _require_at_least_one_input(self) -> "Task3Request":
        if not any(bool(x) for x in (self.session_id, self.image_pairs, self.image_data_pairs)):
            raise ValueError(
                "Provide at least one of: session_id, image_pairs, image_data_pairs"
            )
        return self


class Task3Response(BaseModel):
    """Response for Task 3."""
    session_id: Optional[str] = None
    validations: list[ValidationResult]
    processing_time: float
    total_validated: int
    valid_count: int
    invalid_count: int


# ---------------------------------------------------------------------------
# GIS Pipeline
# ---------------------------------------------------------------------------

class GISPipelineRequest(BaseModel):
    """Request for complete GIS pipeline."""
    session_id: str
    max_pairs_task2: int = 5
    max_pairs_task3: int = 10
    use_cache: bool = True
    priority_weights: Optional[dict[str, float]] = None


class GISPipelineResponse(BaseModel):
    """Complete GIS pipeline response.

    `cache_hit_rate` is the gpu_prefix_cache_hit_rate vLLM reports
    after the pipeline finishes, in [0.0, 1.0]. None means the engine
    did not expose the metric (older vLLM, stats probe failed) — never
    a fabricated zero. Downstream dashboards must treat None as
    "unreported" and exclude it from aggregates.
    """
    session_id: str
    task1_results: list[ImageClassification]
    task2_results: list[Task2Result]
    task3_results: list[ValidationResult]
    summary: dict[str, Any]
    total_processing_time: float
    cache_hit_rate: Optional[float] = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    """Create a new session."""
    session_name: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    image_count: int
    status: Literal["created", "processing", "completed", "error"]
    metadata: Optional[dict[str, Any]] = None


class ImageUpload(BaseModel):
    """Image upload response."""
    session_id: str
    uploaded_files: list[str]
    failed_files: list[str]
    total_uploaded: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthStatus(BaseModel):
    """System health status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    engine_initialized: bool
    gpu_available: bool
    gpu_memory_usage: Optional[str] = None
    active_sessions: int
    uptime_seconds: float

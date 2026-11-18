"""Request and response bodies for the review API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TEXT = 20_000
MAX_BATCH = 32


class TextBody(BaseModel):
    """One review body."""

    model_config = ConfigDict(json_schema_extra={"example": {"text": "کیفیتش خوب بود"}})

    text: str = Field(min_length=1, max_length=MAX_TEXT)


class TextsBody(BaseModel):
    """Several review bodies, scored in one call."""

    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)

    model_config = ConfigDict(
        json_schema_extra={"example": {"texts": ["کیفیتش خوب بود", "اصلا راضی نیستم"]}}
    )

    @field_validator("texts")
    @classmethod
    def _limit_each_text(cls, texts: list[str]) -> list[str]:
        for text in texts:
            if len(text) > MAX_TEXT:
                raise ValueError(f"each text must be at most {MAX_TEXT} characters")
        return texts


class TokenizeBody(TextBody):
    """One text, with an optional token cap."""

    max_length: int | None = Field(default=None, ge=8, le=512)


class TokenizeBatchBody(TextsBody):
    """Several texts, with one shared token cap."""

    max_length: int | None = Field(default=None, ge=8, le=512)


class HealthBody(BaseModel):
    status: str
    models_on_disk: int
    models_loaded: int
    tokenizers_loaded: int


class ModelInfo(BaseModel):
    model_id: str
    task: str
    kind: str
    trained: bool
    loaded: bool
    labels: list[str]
    tokenizer_id: str
    preprocess_id: str
    max_length: int
    path: str
    description: str


class LoadResult(BaseModel):
    model_id: str
    loaded: bool
    seconds: float
    task: str
    trained: bool
    labels: list[str]
    device: str


class PipelineInfo(BaseModel):
    pipeline_id: str
    description: str


class TokenizerInfo(BaseModel):
    tokenizer_id: str
    max_length: int
    loaded: bool
    source: str
    description: str


class PreprocessResult(BaseModel):
    pipeline: str
    text: str
    empty: bool


class PreprocessBatchResult(BaseModel):
    pipeline: str
    items: list[PreprocessResult]


class TokenizeItem(BaseModel):
    input_ids: list[int]
    tokens: list[str]
    truncated: bool


class TokenizeResult(TokenizeItem):
    tokenizer_id: str
    max_length: int


class TokenizeBatchResult(BaseModel):
    tokenizer_id: str
    max_length: int
    items: list[TokenizeItem]


class Weight(BaseModel):
    token: str
    value: float


class EmbeddingItem(BaseModel):
    text: str
    dimensions: int
    vector: list[float] | None = None
    weights: list[Weight] | None = None
    error: str | None = None


class EmbeddingResult(EmbeddingItem):
    model_id: str
    kind: str


class EmbeddingBatchResult(BaseModel):
    model_id: str
    kind: str
    items: list[EmbeddingItem]


class PredictionItem(BaseModel):
    text: str
    label: str | None = None
    label_id: int | None = None
    score: float | None = None
    probabilities: dict[str, float] | None = None
    error: str | None = None


class PredictionResult(PredictionItem):
    model_id: str
    task: str
    trained: bool
    preprocess: str | None = None


class PredictionBatchResult(BaseModel):
    model_id: str
    task: str
    trained: bool
    preprocess: str | None = None
    items: list[PredictionItem]

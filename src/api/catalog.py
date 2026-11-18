"""Checkpoints, tokenizers, and text pipelines the API can serve.

Weights stay on a mounted directory. A checkpoint is listed only when
its files are present. Trainer optimizer state is never listed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

RECOMMENDATION_LABELS = ("recommended", "not_recommended", "no_idea")
RATING_BAND_LABELS = ("low", "medium", "high")


def experiments_dir() -> Path:
    """Directory of transformer checkpoints. Override with ``EXPERIMENTS_DIR``."""
    configured = os.getenv("EXPERIMENTS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "experiments"


def models_dir() -> Path:
    """Directory of the TF-IDF joblib files. Override with ``MODELS_DIR``."""
    configured = os.getenv("MODELS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class ModelSpec:
    """One servable checkpoint."""

    model_id: str
    task: str
    kind: str
    relative_path: str
    root_name: str
    trained: bool
    tokenizer_id: str
    preprocess_id: str
    max_length: int
    description: str

    def path(self) -> Path:
        root = experiments_dir() if self.root_name == "experiments" else models_dir()
        return root / self.relative_path

    def available(self) -> bool:
        path = self.path()
        if self.kind == "tfidf":
            return path.is_file()
        required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
        return all((path / name).is_file() for name in required)


@dataclass(frozen=True)
class TokenizerSpec:
    """One tokenizer the API can run without a classifier."""

    tokenizer_id: str
    max_length: int
    source: str
    description: str
    relative_path: str | None = None
    hub_name: str | None = None

    def path(self) -> Path | None:
        if self.relative_path is None:
            return None
        return experiments_dir() / self.relative_path

    def available(self) -> bool:
        if self.hub_name:
            return True
        path = self.path()
        return path is not None and (path / "tokenizer.json").is_file()


@dataclass(frozen=True)
class PipelineSpec:
    """One text pipeline."""

    pipeline_id: str
    description: str


PIPELINES: tuple[PipelineSpec, ...] = (
    PipelineSpec(
        "shared",
        "NFKC, Persian letters and digits, one half-space, URLs removed, "
        "character runs capped, then Hazm stemming. The TF-IDF models were fit on this text.",
    ),
    PipelineSpec(
        "text-v2",
        "Letters, then digits, then emoji. Does not stem. "
        "This is the extra stage the TookaBERT runs apply to already stemmed text.",
    ),
    PipelineSpec(
        "full",
        "The shared pipeline, then text-v2. "
        "Raw review text for every TookaBERT checkpoint uses this pipeline.",
    ),
)


def _transformer(
    model_id: str,
    task: str,
    relative_path: str,
    trained: bool,
    description: str,
    preprocess_id: str = "full",
    max_length: int = 128,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        task=task,
        kind="transformer",
        relative_path=relative_path,
        root_name="experiments",
        trained=trained,
        tokenizer_id="tookabert",
        preprocess_id=preprocess_id,
        max_length=max_length,
        description=description,
    )


MODEL_SPECS: tuple[ModelSpec, ...] = (
    _transformer(
        "tookabert-v2-epoch-1",
        "recommendation",
        "epochs/epoch-1",
        True,
        "TookaBERT preprocessing v2, recommendation head, epoch 1.",
    ),
    _transformer(
        "tookabert-v2-epoch-2",
        "recommendation",
        "epochs/epoch-2",
        True,
        "TookaBERT preprocessing v2, recommendation head, epoch 2.",
    ),
    _transformer(
        "tookabert-v2-epoch-3",
        "recommendation",
        "epochs/epoch-3",
        True,
        "TookaBERT preprocessing v2, recommendation head, epoch 3.",
    ),
    _transformer(
        "tookabert-v2-epoch-4",
        "recommendation",
        "epochs/epoch-4",
        True,
        "TookaBERT preprocessing v2, recommendation head, epoch 4.",
    ),
    _transformer(
        "tookabert-v2-epoch-5",
        "recommendation",
        "epochs/epoch-5",
        True,
        "TookaBERT preprocessing v2, recommendation head, epoch 5.",
    ),
    _transformer(
        "tookabert-noidea-epoch-1",
        "recommendation",
        "tookabert-noidea-penalty-2026-11-13/epoch-1",
        True,
        "TookaBERT continued from epoch 2 with a penalty on true no_idea, epoch 1.",
    ),
    _transformer(
        "tookabert-noidea-epoch-2",
        "recommendation",
        "tookabert-noidea-penalty-2026-11-13/epoch-2",
        True,
        "TookaBERT no_idea penalty, epoch 2.",
    ),
    _transformer(
        "tookabert-noidea-epoch-3",
        "recommendation",
        "tookabert-noidea-penalty-2026-11-13/epoch-3",
        True,
        "TookaBERT no_idea penalty, epoch 3.",
    ),
    _transformer(
        "tookabert-rating-band",
        "rating_band",
        "PartAI--TookaBERT-Base--rating-band",
        True,
        "TookaBERT rating band, one trained epoch. Labels are low, medium, and high.",
    ),
    _transformer(
        "tookabert-rating-band-untrained",
        "rating_band",
        "baselines/tookabert-epoch2-rating-band",
        False,
        "Epoch-2 encoder with a new untrained 3-class rating-band head.",
    ),
    _transformer(
        "tookabert-rate-untrained",
        "rate",
        "baselines/tookabert-epoch2-regression",
        False,
        "Epoch-2 encoder with a new untrained one-output rate layer. It has no trained score.",
    ),
    ModelSpec(
        model_id="tfidf-recommendation",
        task="recommendation",
        kind="tfidf",
        relative_path="recommendation-tfidf-logreg.joblib",
        root_name="models",
        trained=True,
        tokenizer_id="parsbert",
        preprocess_id="shared",
        max_length=512,
        description="TF-IDF logistic regression on ParsBERT token ids. Predicts recommendation status.",
    ),
    ModelSpec(
        model_id="tfidf-rating-band",
        task="rating_band",
        kind="tfidf",
        relative_path="rating-band-tfidf-logreg.joblib",
        root_name="models",
        trained=True,
        tokenizer_id="parsbert",
        preprocess_id="shared",
        max_length=512,
        description="TF-IDF logistic regression on ParsBERT token ids. Predicts low, medium, and high.",
    ),
)


TOKENIZERS: tuple[TokenizerSpec, ...] = (
    TokenizerSpec(
        tokenizer_id="tookabert",
        max_length=128,
        source="PartAI/TookaBERT-Base",
        description="TookaBERT BPE, the tokenizer saved with the epoch-2 checkpoint. Training length is 128.",
        relative_path="epochs/epoch-2",
    ),
    TokenizerSpec(
        tokenizer_id="parsbert",
        max_length=512,
        source="HooshvareLab/bert-fa-base-uncased",
        description="ParsBERT WordPiece. The TF-IDF baselines and the tokenized parquet use this tokenizer at length 512.",
        hub_name="HooshvareLab/bert-fa-base-uncased",
    ),
)


def task_from_labels(labels: tuple[str, ...]) -> str | None:
    """Map a checkpoint's label names onto a task, when the names are known."""
    names = set(labels)
    if names == set(RECOMMENDATION_LABELS):
        return "recommendation"
    if names == set(RATING_BAND_LABELS):
        return "rating_band"
    if names == {"rate"}:
        return "rate"
    return None


def read_transformer_labels(path: Path) -> tuple[str, ...]:
    """Read ``id2label`` from a checkpoint config, in id order."""
    raw = json.loads((path / "config.json").read_text())
    id2label = raw.get("id2label") or {}
    ordered: list[str] = []
    for index in range(len(id2label)):
        value = id2label.get(str(index), id2label.get(index))
        if value is None:
            raise ValueError(f"{path} id2label is missing id {index}")
        ordered.append(str(value))
    if not ordered:
        raise ValueError(f"{path} has no id2label")
    return tuple(ordered)


def available_models() -> list[ModelSpec]:
    """Specs whose weight files exist and whose labels match the declared task."""
    found: list[ModelSpec] = []
    for spec in MODEL_SPECS:
        if not spec.available():
            logger.info("skipping %s; files are not at %s", spec.model_id, spec.path())
            continue
        if spec.kind == "transformer":
            labels = read_transformer_labels(spec.path())
            task = task_from_labels(labels)
            if task != spec.task:
                logger.error(
                    "skipping %s; config labels %s do not match task %s",
                    spec.model_id,
                    labels,
                    spec.task,
                )
                continue
        found.append(spec)
    return found


def model_by_id(model_id: str) -> ModelSpec | None:
    for spec in available_models():
        if spec.model_id == model_id:
            return spec
    return None


def pipeline_by_id(pipeline_id: str) -> PipelineSpec | None:
    for spec in PIPELINES:
        if spec.pipeline_id == pipeline_id:
            return spec
    return None


def tokenizer_by_id(tokenizer_id: str) -> TokenizerSpec | None:
    for spec in TOKENIZERS:
        if spec.tokenizer_id == tokenizer_id and spec.available():
            return spec
    return None


def task_slug(task: str) -> str:
    """URL piece for a task name."""
    if task == "rating_band":
        return "rating-band"
    return task

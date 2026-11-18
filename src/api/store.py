"""Load checkpoints on demand. The files themselves stay outside the image."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src.api.catalog import ModelSpec, TokenizerSpec
from src.api.metrics import LOAD_LATENCY, MODEL_ERRORS, MODELS_LOADED, TOKENIZERS_LOADED


class ModelNotLoadedError(Exception):
    """A predict or embed call arrived before the checkpoint was loaded."""


@dataclass
class LoadedModel:
    """A checkpoint resident in this process."""

    spec: ModelSpec
    labels: tuple[str, ...]
    device: str
    transformer: object | None = None
    tokenizer: object | None = None
    vectorizer: object | None = None
    classifier: object | None = None


class ModelStore:
    """Holds loaded classifiers and tokenizers. Callers take the lock per id."""

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._tokenizers: dict[str, object] = {}
        self._lock = threading.Lock()

    def loaded_model_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._models

    def tokenizer_loaded(self, tokenizer_id: str) -> bool:
        with self._lock:
            return tokenizer_id in self._tokenizers

    def require(self, spec: ModelSpec) -> LoadedModel:
        with self._lock:
            loaded = self._models.get(spec.model_id)
        if loaded is None:
            raise ModelNotLoadedError(
                f"Model {spec.model_id} is not loaded. "
                f"POST /models/{spec.model_id}/load first."
            )
        return loaded

    def load(self, spec: ModelSpec) -> tuple[LoadedModel, float]:
        """Load ``spec`` if it is not already in memory. Returns the resident model and the seconds spent."""
        with self._lock:
            existing = self._models.get(spec.model_id)
        if existing is not None:
            return existing, 0.0
        started = time.perf_counter()
        try:
            if spec.kind == "transformer":
                loaded = self._load_transformer(spec)
            elif spec.kind == "tfidf":
                loaded = self._load_tfidf(spec)
            else:
                raise ValueError(f"unknown model kind: {spec.kind}")
        except Exception:
            MODEL_ERRORS.labels(spec.model_id, "load").inc()
            raise
        seconds = time.perf_counter() - started
        with self._lock:
            current = self._models.get(spec.model_id)
            if current is None:
                self._models[spec.model_id] = loaded
                current = loaded
            MODELS_LOADED.set(len(self._models))
        LOAD_LATENCY.labels(spec.model_id).observe(seconds)
        return current, seconds

    def unload(self, model_id: str) -> bool:
        with self._lock:
            loaded = self._models.pop(model_id, None)
            MODELS_LOADED.set(len(self._models))
        if loaded is None:
            return False
        del loaded
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return True

    def get_tokenizer(self, spec: TokenizerSpec):
        with self._lock:
            cached = self._tokenizers.get(spec.tokenizer_id)
        if cached is not None:
            return cached
        tokenizer = _load_tokenizer(spec)
        with self._lock:
            self._tokenizers.setdefault(spec.tokenizer_id, tokenizer)
            TOKENIZERS_LOADED.set(len(self._tokenizers))
            return self._tokenizers[spec.tokenizer_id]

    def _load_transformer(self, spec: ModelSpec) -> LoadedModel:
        import torch
        from transformers import AutoModelForSequenceClassification

        path = spec.path()
        config = _load_config(path)
        label_count = len(config.id2label)
        config.num_labels = label_count
        model = AutoModelForSequenceClassification.from_pretrained(path, config=config)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        tokenizer = _load_tokenizer_from_dir(path)
        labels = _labels_from_config(config)
        return LoadedModel(
            spec=spec,
            labels=labels,
            device=device_name,
            transformer=model,
            tokenizer=tokenizer,
        )

    def _load_tfidf(self, spec: ModelSpec) -> LoadedModel:
        import joblib

        payload = joblib.load(spec.path())
        labels = payload.get("labels") or payload.get("bands")
        if not labels:
            raise ValueError(f"{spec.path()} has no labels")
        return LoadedModel(
            spec=spec,
            labels=tuple(str(label) for label in labels),
            device="cpu",
            vectorizer=payload["vectorizer"],
            classifier=payload["classifier"],
        )


def _labels_from_config(config) -> tuple[str, ...]:
    id2label = config.id2label
    ordered: list[str] = []
    for index in range(len(id2label)):
        value = id2label.get(index, id2label.get(str(index)))
        if value is None:
            raise ValueError(f"id2label is missing id {index}")
        ordered.append(str(value))
    return tuple(ordered)


def _load_config(path: Path):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(path)
    except Exception:
        raw = json.loads((path / "config.json").read_text())
        raw.pop("dtype", None)
        raw.pop("transformers_version", None)
        model_type = raw.pop("model_type", "bert")
        return AutoConfig.for_model(model_type, **raw)


def _load_tokenizer(spec: TokenizerSpec):
    if spec.hub_name:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(spec.hub_name)
    path = spec.path()
    if path is None:
        raise FileNotFoundError(f"tokenizer {spec.tokenizer_id} has no files")
    return _load_tokenizer_from_dir(path)


def _load_tokenizer_from_dir(path: Path):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)

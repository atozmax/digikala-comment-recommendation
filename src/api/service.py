"""Preprocess, tokenize, embed, and classify one text or a batch."""

from __future__ import annotations

import time

from src.api import metrics
from src.api.catalog import ModelSpec, PipelineSpec, TokenizerSpec
from src.api.schemas import (
    EmbeddingItem,
    EmbeddingResult,
    PredictionItem,
    PredictionResult,
    PreprocessResult,
    TokenizeItem,
    TokenizeResult,
    Weight,
)
from src.api.store import LoadedModel, ModelStore
from src.experiments.text_v2 import apply_digits, apply_emoji, apply_letters
from src.preprocess import preprocess_text


class EmptyTextError(Exception):
    """The body was blank, or the pipeline removed every character."""


def run_pipeline(pipeline: PipelineSpec, text: str) -> str:
    """Apply one named pipeline. ``full`` is the shared pass followed by text-v2."""
    started = time.perf_counter()
    try:
        if pipeline.pipeline_id == "shared":
            return preprocess_text(text)
        if pipeline.pipeline_id == "text-v2":
            return apply_emoji(apply_digits(apply_letters(text)))
        if pipeline.pipeline_id == "full":
            return apply_emoji(apply_digits(apply_letters(preprocess_text(text))))
        raise ValueError(f"unknown pipeline: {pipeline.pipeline_id}")
    finally:
        metrics.PREPROCESS_LATENCY.labels(pipeline.pipeline_id).observe(time.perf_counter() - started)


def preprocess_one(pipeline: PipelineSpec, text: str) -> PreprocessResult:
    rewritten = run_pipeline(pipeline, text)
    empty = not rewritten.strip()
    if empty:
        metrics.EMPTY_TEXT.labels("preprocess").inc()
    return PreprocessResult(pipeline=pipeline.pipeline_id, text=rewritten, empty=empty)


def preprocess_many(pipeline: PipelineSpec, texts: list[str]) -> list[PreprocessResult]:
    metrics.BATCH_ITEMS.labels("preprocess").observe(len(texts))
    return [preprocess_one(pipeline, text) for text in texts]


def tokenize_one(store: ModelStore, spec: TokenizerSpec, text: str, max_length: int | None) -> TokenizeResult:
    limit = max_length or spec.max_length
    started = time.perf_counter()
    try:
        item = _encode(store.get_tokenizer(spec), text, limit)
    finally:
        metrics.TOKENIZE_LATENCY.labels(spec.tokenizer_id).observe(time.perf_counter() - started)
    return TokenizeResult(
        tokenizer_id=spec.tokenizer_id,
        max_length=limit,
        input_ids=item.input_ids,
        tokens=item.tokens,
        truncated=item.truncated,
    )


def tokenize_many(
    store: ModelStore,
    spec: TokenizerSpec,
    texts: list[str],
    max_length: int | None,
) -> tuple[int, list[TokenizeItem]]:
    metrics.BATCH_ITEMS.labels("tokenize").observe(len(texts))
    limit = max_length or spec.max_length
    started = time.perf_counter()
    try:
        tokenizer = store.get_tokenizer(spec)
        items = [_encode(tokenizer, text, limit) for text in texts]
    finally:
        metrics.TOKENIZE_LATENCY.labels(spec.tokenizer_id).observe(time.perf_counter() - started)
    return limit, items


def embed_one(store: ModelStore, spec: ModelSpec, text: str) -> EmbeddingResult:
    started = time.perf_counter()
    try:
        item = _embed_texts(store, spec, [text])[0]
    finally:
        metrics.EMBED_LATENCY.labels(spec.model_id).observe(time.perf_counter() - started)
    return EmbeddingResult(model_id=spec.model_id, kind=_embed_kind(spec), **item.model_dump())


def embed_many(store: ModelStore, spec: ModelSpec, texts: list[str]) -> list[EmbeddingItem]:
    metrics.BATCH_ITEMS.labels("embed").observe(len(texts))
    started = time.perf_counter()
    try:
        return _embed_texts(store, spec, texts)
    finally:
        metrics.EMBED_LATENCY.labels(spec.model_id).observe(time.perf_counter() - started)


def predict_one(store: ModelStore, spec: ModelSpec, text: str, *, preprocess: str | None) -> PredictionResult:
    prepared = _prepare(spec, text, preprocess)
    started = time.perf_counter()
    try:
        item = _predict_texts(store, spec, [prepared], mode=_mode(preprocess))[0]
    finally:
        metrics.INFERENCE_LATENCY.labels(spec.model_id, spec.task).observe(time.perf_counter() - started)
    return PredictionResult(
        model_id=spec.model_id,
        task=spec.task,
        trained=spec.trained,
        preprocess=preprocess,
        **item.model_dump(),
    )


def predict_many(
    store: ModelStore,
    spec: ModelSpec,
    texts: list[str],
    *,
    preprocess: str | None,
) -> list[PredictionItem]:
    metrics.BATCH_ITEMS.labels("predict").observe(len(texts))
    prepared = [_prepare_or_blank(spec, text, preprocess) for text in texts]
    started = time.perf_counter()
    try:
        return _predict_texts(store, spec, prepared, mode=_mode(preprocess))
    finally:
        metrics.INFERENCE_LATENCY.labels(spec.model_id, spec.task).observe(time.perf_counter() - started)


def _mode(preprocess: str | None) -> str:
    return "text" if preprocess else "predict"


def _prepare(spec: ModelSpec, text: str, preprocess: str | None) -> str:
    if preprocess is None:
        if not text.strip():
            metrics.EMPTY_TEXT.labels("predict").inc()
            raise EmptyTextError("text is blank")
        return text
    from src.api.catalog import pipeline_by_id

    pipeline = pipeline_by_id(preprocess)
    if pipeline is None:
        raise ValueError(f"unknown pipeline: {preprocess}")
    rewritten = run_pipeline(pipeline, text)
    if not rewritten.strip():
        metrics.EMPTY_TEXT.labels("predict").inc()
        raise EmptyTextError(f"text is empty after the {preprocess} pipeline")
    return rewritten


def _prepare_or_blank(spec: ModelSpec, text: str, preprocess: str | None) -> str:
    try:
        return _prepare(spec, text, preprocess)
    except EmptyTextError:
        return ""


def _encode(tokenizer, text: str, max_length: int) -> TokenizeItem:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(input_ids)]
    untruncated = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    truncated = len(untruncated["input_ids"]) > max_length
    return TokenizeItem(input_ids=input_ids, tokens=tokens, truncated=truncated)


def _embed_kind(spec: ModelSpec) -> str:
    if spec.kind == "tfidf":
        return "tfidf"
    return "hidden_mean"


def _embed_texts(store: ModelStore, spec: ModelSpec, texts: list[str]) -> list[EmbeddingItem]:
    loaded = store.require(spec)
    if spec.kind == "tfidf":
        return _embed_tfidf(store, loaded, texts)
    return _embed_transformer(loaded, texts)


def _embed_transformer(loaded: LoadedModel, texts: list[str]) -> list[EmbeddingItem]:
    import torch

    usable = [index for index, text in enumerate(texts) if text.strip()]
    items = [
        EmbeddingItem(text=text, dimensions=0, error="text is blank")
        if not text.strip()
        else None
        for text in texts
    ]
    if not usable:
        for item in items:
            if item is not None and item.error:
                metrics.EMPTY_TEXT.labels("embed").inc()
        return [item if item is not None else EmbeddingItem(text="", dimensions=0, error="text is blank") for item in items]
    tokenizer = loaded.tokenizer
    model = loaded.transformer
    encoded = tokenizer(
        [texts[index] for index in usable],
        add_special_tokens=True,
        truncation=True,
        max_length=loaded.spec.max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
    encoder = model.bert if hasattr(model, "bert") else model.base_model
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    vectors = pooled.detach().cpu().tolist()
    for slot, vector in zip(usable, vectors):
        items[slot] = EmbeddingItem(
            text=texts[slot],
            dimensions=len(vector),
            vector=[round(float(value), 6) for value in vector],
        )
    resolved: list[EmbeddingItem] = []
    for item in items:
        if item is None:
            resolved.append(EmbeddingItem(text="", dimensions=0, error="text is blank"))
        else:
            if item.error:
                metrics.EMPTY_TEXT.labels("embed").inc()
            resolved.append(item)
    return resolved


def _embed_tfidf(store: ModelStore, loaded: LoadedModel, texts: list[str]) -> list[EmbeddingItem]:
    from src.api.catalog import tokenizer_by_id

    spec = tokenizer_by_id(loaded.spec.tokenizer_id)
    if spec is None:
        raise FileNotFoundError(f"tokenizer {loaded.spec.tokenizer_id} is not available")
    tokenizer = store.get_tokenizer(spec)
    items: list[EmbeddingItem] = []
    documents: list[str] = []
    slots: list[int] = []
    for index, text in enumerate(texts):
        if not text.strip():
            metrics.EMPTY_TEXT.labels("embed").inc()
            items.append(EmbeddingItem(text=text, dimensions=0, error="text is blank"))
            continue
        items.append(EmbeddingItem(text=text, dimensions=0))
        documents.append(_id_document(tokenizer, text, loaded.spec.max_length))
        slots.append(index)
    if documents:
        matrix = loaded.vectorizer.transform(documents)
        names = loaded.vectorizer.get_feature_names_out()
        for row, slot in enumerate(slots):
            column = matrix.getrow(row)
            weights = [
                Weight(token=str(names[int(key)]), value=round(float(column.data[pos]), 6))
                for pos, key in enumerate(column.indices)
            ]
            items[slot] = EmbeddingItem(
                text=texts[slot],
                dimensions=int(matrix.shape[1]),
                weights=weights,
            )
    return items


def _predict_texts(store: ModelStore, spec: ModelSpec, texts: list[str], *, mode: str) -> list[PredictionItem]:
    loaded = store.require(spec)
    if spec.kind == "tfidf":
        return _predict_tfidf(store, loaded, texts, mode)
    return _predict_transformer(loaded, texts, mode)


def _predict_transformer(loaded: LoadedModel, texts: list[str], mode: str) -> list[PredictionItem]:
    import torch

    items: list[PredictionItem | None] = [None] * len(texts)
    usable = [index for index, text in enumerate(texts) if text.strip()]
    for index, text in enumerate(texts):
        if not text.strip():
            items[index] = PredictionItem(text=text, error="text is empty")
    if not usable:
        return [item for item in items if item is not None]
    tokenizer = loaded.tokenizer
    model = loaded.transformer
    encoded = tokenizer(
        [texts[index] for index in usable],
        add_special_tokens=True,
        truncation=True,
        max_length=loaded.spec.max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    if loaded.spec.task == "rate":
        values = logits.detach().cpu().reshape(-1).tolist()
        for slot, value in zip(usable, values):
            score = round(float(value), 6)
            items[slot] = PredictionItem(text=texts[slot], label="rate", label_id=0, score=score)
            metrics.PREDICTIONS.labels(loaded.spec.model_id, loaded.spec.task, "rate", mode).inc()
        return [item for item in items if item is not None]
    probabilities = torch.softmax(logits, dim=-1).detach().cpu()
    chosen = probabilities.argmax(dim=-1).tolist()
    for row, slot in enumerate(usable):
        label_id = int(chosen[row])
        label = loaded.labels[label_id]
        mapping = {
            loaded.labels[index]: round(float(probabilities[row, index]), 6)
            for index in range(len(loaded.labels))
        }
        items[slot] = PredictionItem(
            text=texts[slot],
            label=label,
            label_id=label_id,
            score=mapping[label],
            probabilities=mapping,
        )
        metrics.PREDICTIONS.labels(loaded.spec.model_id, loaded.spec.task, label, mode).inc()
    return [item for item in items if item is not None]


def _predict_tfidf(store: ModelStore, loaded: LoadedModel, texts: list[str], mode: str) -> list[PredictionItem]:
    from src.api.catalog import tokenizer_by_id

    spec = tokenizer_by_id(loaded.spec.tokenizer_id)
    if spec is None:
        raise FileNotFoundError(f"tokenizer {loaded.spec.tokenizer_id} is not available")
    tokenizer = store.get_tokenizer(spec)
    items: list[PredictionItem | None] = [None] * len(texts)
    documents: list[str] = []
    slots: list[int] = []
    for index, text in enumerate(texts):
        if not text.strip():
            items[index] = PredictionItem(text=text, error="text is empty")
            continue
        documents.append(_id_document(tokenizer, text, loaded.spec.max_length))
        slots.append(index)
    if documents:
        matrix = loaded.vectorizer.transform(documents)
        predicted = loaded.classifier.predict(matrix)
        probabilities = loaded.classifier.predict_proba(matrix)
        classes = [int(value) for value in loaded.classifier.classes_]
        for row, slot in enumerate(slots):
            label_id = int(predicted[row])
            label = loaded.labels[label_id]
            mapping = {
                loaded.labels[class_id]: round(float(probabilities[row, column]), 6)
                for column, class_id in enumerate(classes)
            }
            items[slot] = PredictionItem(
                text=texts[slot],
                label=label,
                label_id=label_id,
                score=mapping[label],
                probabilities=mapping,
            )
            metrics.PREDICTIONS.labels(loaded.spec.model_id, loaded.spec.task, label, mode).inc()
    return [item for item in items if item is not None]


def _id_document(tokenizer, text: str, max_length: int) -> str:
    """Space-separated token ids with the classifier, separator, and pad ids removed.

    That is the document string the TF-IDF vectorizer was fit on.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    special = {
        token_id
        for token_id in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    kept = [str(int(token_id)) for token_id in encoded["input_ids"] if int(token_id) not in special]
    return " ".join(kept)

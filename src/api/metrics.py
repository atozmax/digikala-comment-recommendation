"""Prometheus metrics for the review API.

The process and Python collectors are registered by prometheus_client.
Grafana reads this registry from ``GET /metrics``.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "review_http_requests_total",
    "HTTP requests handled by the review API.",
    ("method", "path", "status"),
)
HTTP_LATENCY = Histogram(
    "review_http_request_seconds",
    "HTTP request latency in seconds.",
    ("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
PREPROCESS_LATENCY = Histogram(
    "review_preprocess_seconds",
    "Preprocessing latency in seconds.",
    ("pipeline",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
TOKENIZE_LATENCY = Histogram(
    "review_tokenize_seconds",
    "Tokenizer latency in seconds.",
    ("tokenizer",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
EMBED_LATENCY = Histogram(
    "review_embed_seconds",
    "Embedding latency in seconds.",
    ("model",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
INFERENCE_LATENCY = Histogram(
    "review_inference_seconds",
    "Classifier latency in seconds, including tokenization.",
    ("model", "task"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
LOAD_LATENCY = Histogram(
    "review_model_load_seconds",
    "Time to load one checkpoint into memory.",
    ("model",),
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
PREDICTIONS = Counter(
    "review_predictions_total",
    "Labels returned by a classifier.",
    ("model", "task", "label", "mode"),
)
BATCH_ITEMS = Histogram(
    "review_batch_items",
    "Number of texts in one batch call.",
    ("operation",),
    buckets=(1, 2, 4, 8, 16, 32),
)
EMPTY_TEXT = Counter(
    "review_empty_text_total",
    "Texts that were blank, or blank after preprocessing.",
    ("operation",),
)
MODEL_ERRORS = Counter(
    "review_model_errors_total",
    "Failed model loads and failed inference calls.",
    ("model", "reason"),
)
MODELS_LOADED = Gauge(
    "review_models_loaded",
    "Checkpoints currently held in memory.",
)
MODELS_CATALOG = Gauge(
    "review_models_catalog",
    "Checkpoints found on disk when the process started.",
)
TOKENIZERS_LOADED = Gauge(
    "review_tokenizers_loaded",
    "Tokenizers currently held in memory.",
)

"""FastAPI application for the review checkpoints.

Routes are registered from the checkpoints present at startup. Mount the
weight directories before the process starts. A later file drop is picked
up on the next start.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import metrics, service
from src.api.catalog import (
    PIPELINES,
    available_models,
    model_by_id,
    pipeline_by_id,
    task_slug,
    tokenizer_by_id,
)
from src.api.catalog import TOKENIZERS
from src.api.schemas import (
    EmbeddingBatchResult,
    EmbeddingResult,
    HealthBody,
    LoadResult,
    ModelInfo,
    PipelineInfo,
    PredictionBatchResult,
    PredictionResult,
    PreprocessBatchResult,
    PreprocessResult,
    TextBody,
    TextsBody,
    TokenizeBatchBody,
    TokenizeBatchResult,
    TokenizeBody,
    TokenizeResult,
    TokenizerInfo,
)
from src.api.service import EmptyTextError
from src.api.store import ModelNotLoadedError, ModelStore


def create_app() -> FastAPI:
    """Build the API, including one route per checkpoint found on disk."""
    store = ModelStore()
    models = available_models()
    metrics.MODELS_CATALOG.set(len(models))
    app = FastAPI(
        title="Digikala review models",
        summary="Recommendation status and rating band from a Persian review body.",
        version="1.0.0",
    )
    app.state.store = store

    @app.middleware("http")
    async def record_http(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = route.path if route is not None else request.url.path
            metrics.HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - started)
            metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()

    @app.get("/health/live", response_model=HealthBody, tags=["health"])
    def live() -> HealthBody:
        """Process is up."""
        return _health("ok", store, models)

    @app.get("/health/ready", response_model=HealthBody, tags=["health"])
    def ready() -> HealthBody:
        """Ready when at least one checkpoint is mounted."""
        if not models:
            raise HTTPException(
                status_code=503,
                detail="No checkpoints are mounted. Check EXPERIMENTS_DIR and MODELS_DIR.",
            )
        return _health("ok", store, models)

    @app.get("/health", response_model=HealthBody, tags=["health"])
    def health() -> HealthBody:
        """Same payload as the live check."""
        return _health("ok", store, models)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/models", response_model=list[ModelInfo], tags=["models"])
    def list_models() -> list[ModelInfo]:
        """Checkpoints this process can load."""
        return [_model_info(spec, store) for spec in models]

    @app.get("/models/{model_id}", response_model=ModelInfo, tags=["models"])
    def get_model(model_id: str) -> ModelInfo:
        spec = model_by_id(model_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return _model_info(spec, store)

    @app.post("/models/{model_id}/load", response_model=LoadResult, tags=["models"])
    async def load_model(model_id: str) -> LoadResult:
        """Read one checkpoint into memory. A second call keeps the resident copy."""
        spec = model_by_id(model_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        try:
            loaded, seconds = await asyncio.to_thread(store.load, spec)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return LoadResult(
            model_id=spec.model_id,
            loaded=True,
            seconds=round(seconds, 3),
            task=spec.task,
            trained=spec.trained,
            labels=list(loaded.labels),
            device=loaded.device,
        )

    @app.post("/models/{model_id}/unload", response_model=LoadResult, tags=["models"])
    async def unload_model(model_id: str) -> LoadResult:
        """Drop one checkpoint from memory."""
        spec = model_by_id(model_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        await asyncio.to_thread(store.unload, model_id)
        return LoadResult(
            model_id=spec.model_id,
            loaded=False,
            seconds=0.0,
            task=spec.task,
            trained=spec.trained,
            labels=[],
            device="",
        )

    @app.get("/pipelines", response_model=list[PipelineInfo], tags=["preprocess"])
    def list_pipelines() -> list[PipelineInfo]:
        return [PipelineInfo(pipeline_id=item.pipeline_id, description=item.description) for item in PIPELINES]

    @app.get("/tokenizers", response_model=list[TokenizerInfo], tags=["tokenize"])
    def list_tokenizers() -> list[TokenizerInfo]:
        listed = []
        for spec in TOKENIZERS:
            if not spec.available():
                continue
            listed.append(
                TokenizerInfo(
                    tokenizer_id=spec.tokenizer_id,
                    max_length=spec.max_length,
                    loaded=store.tokenizer_loaded(spec.tokenizer_id),
                    source=spec.source,
                    description=spec.description,
                )
            )
        return listed

    for pipeline in PIPELINES:
        _add_preprocess_routes(app, pipeline.pipeline_id)

    for spec in TOKENIZERS:
        if spec.available():
            _add_tokenizer_routes(app, store, spec.tokenizer_id)

    for spec in models:
        _add_model_routes(app, store, spec.model_id)

    return app


def _health(status: str, store: ModelStore, models: list) -> HealthBody:
    return HealthBody(
        status=status,
        models_on_disk=len(models),
        models_loaded=len(store.loaded_model_ids()),
        tokenizers_loaded=sum(1 for spec in TOKENIZERS if store.tokenizer_loaded(spec.tokenizer_id)),
    )


def _model_info(spec, store: ModelStore) -> ModelInfo:
    return ModelInfo(
        model_id=spec.model_id,
        task=spec.task,
        kind=spec.kind,
        trained=spec.trained,
        loaded=store.is_loaded(spec.model_id),
        labels=_labels_for(spec),
        tokenizer_id=spec.tokenizer_id,
        preprocess_id=spec.preprocess_id,
        max_length=spec.max_length,
        path=str(spec.path()),
        description=spec.description,
    )


def _labels_for(spec) -> list[str]:
    if spec.kind == "transformer":
        from src.api.catalog import read_transformer_labels

        return list(read_transformer_labels(spec.path()))
    if spec.task == "recommendation":
        return ["recommended", "not_recommended", "no_idea"]
    if spec.task == "rating_band":
        return ["low", "medium", "high"]
    return ["rate"]


async def _call(model_id: str | None, func):
    try:
        return await asyncio.to_thread(func)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EmptyTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if model_id is not None:
            metrics.MODEL_ERRORS.labels(model_id, "inference").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _add_preprocess_routes(app: FastAPI, pipeline_id: str) -> None:
    async def one(body: TextBody) -> PreprocessResult:
        pipeline = pipeline_by_id(pipeline_id)
        return await _call(None, lambda: service.preprocess_one(pipeline, body.text))

    async def batch(body: TextsBody) -> PreprocessBatchResult:
        pipeline = pipeline_by_id(pipeline_id)
        items = await _call(None, lambda: service.preprocess_many(pipeline, body.texts))
        return PreprocessBatchResult(pipeline=pipeline_id, items=items)

    one.__name__ = f"preprocess_{pipeline_id}"
    batch.__name__ = f"preprocess_{pipeline_id}_batch"
    app.add_api_route(
        f"/preprocess/{pipeline_id}",
        one,
        methods=["POST"],
        response_model=PreprocessResult,
        tags=["preprocess"],
        summary=f"Run the {pipeline_id} pipeline on one text",
        operation_id=f"preprocess_{pipeline_id}".replace("-", "_"),
    )
    app.add_api_route(
        f"/preprocess/{pipeline_id}/batch",
        batch,
        methods=["POST"],
        response_model=PreprocessBatchResult,
        tags=["preprocess"],
        summary=f"Run the {pipeline_id} pipeline on a batch",
        operation_id=f"preprocess_{pipeline_id}_batch".replace("-", "_"),
    )


def _add_tokenizer_routes(app: FastAPI, store: ModelStore, tokenizer_id: str) -> None:
    async def one(body: TokenizeBody) -> TokenizeResult:
        spec = tokenizer_by_id(tokenizer_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown tokenizer: {tokenizer_id}")
        return await _call(None, lambda: service.tokenize_one(store, spec, body.text, body.max_length))

    async def batch(body: TokenizeBatchBody) -> TokenizeBatchResult:
        spec = tokenizer_by_id(tokenizer_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown tokenizer: {tokenizer_id}")

        def run():
            limit, items = service.tokenize_many(store, spec, body.texts, body.max_length)
            return limit, items

        limit, items = await _call(None, run)
        return TokenizeBatchResult(tokenizer_id=tokenizer_id, max_length=limit, items=items)

    one.__name__ = f"tokenize_{tokenizer_id}"
    batch.__name__ = f"tokenize_{tokenizer_id}_batch"
    app.add_api_route(
        f"/tokenize/{tokenizer_id}",
        one,
        methods=["POST"],
        response_model=TokenizeResult,
        tags=["tokenize"],
        summary=f"Tokenize one text with {tokenizer_id}",
        operation_id=f"tokenize_{tokenizer_id}".replace("-", "_"),
    )
    app.add_api_route(
        f"/tokenize/{tokenizer_id}/batch",
        batch,
        methods=["POST"],
        response_model=TokenizeBatchResult,
        tags=["tokenize"],
        summary=f"Tokenize a batch with {tokenizer_id}",
        operation_id=f"tokenize_{tokenizer_id}_batch".replace("-", "_"),
    )


def _add_model_routes(app: FastAPI, store: ModelStore, model_id: str) -> None:
    spec = model_by_id(model_id)
    if spec is None:
        return
    slug = task_slug(spec.task)
    _add_embed_routes(app, store, spec)
    _add_predict_routes(app, store, spec, slug, preprocess=None, prefix=f"/predict/{slug}/{spec.model_id}")
    _add_predict_routes(
        app,
        store,
        spec,
        slug,
        preprocess=spec.preprocess_id,
        prefix=f"/{slug}/{spec.model_id}",
    )


def _add_embed_routes(app: FastAPI, store: ModelStore, spec) -> None:
    model_id = spec.model_id

    async def one(body: TextBody) -> EmbeddingResult:
        result = await _call(model_id, lambda: service.embed_one(store, spec, body.text))
        if result.error:
            raise HTTPException(status_code=422, detail=result.error)
        return result

    async def batch(body: TextsBody) -> EmbeddingBatchResult:
        items = await _call(model_id, lambda: service.embed_many(store, spec, body.texts))
        kind = "tfidf" if spec.kind == "tfidf" else "hidden_mean"
        return EmbeddingBatchResult(model_id=model_id, kind=kind, items=items)

    one.__name__ = f"embed_{model_id}"
    batch.__name__ = f"embed_{model_id}_batch"
    app.add_api_route(
        f"/embed/{model_id}",
        one,
        methods=["POST"],
        response_model=EmbeddingResult,
        tags=["embed"],
        summary=f"Embed one text with {model_id}",
        description=_embed_description(spec),
        operation_id=f"embed_{model_id}".replace("-", "_"),
    )
    app.add_api_route(
        f"/embed/{model_id}/batch",
        batch,
        methods=["POST"],
        response_model=EmbeddingBatchResult,
        tags=["embed"],
        summary=f"Embed a batch with {model_id}",
        description=_embed_description(spec),
        operation_id=f"embed_{model_id}_batch".replace("-", "_"),
    )


def _embed_description(spec) -> str:
    if spec.kind == "tfidf":
        return (
            f"Tokenizes with {spec.tokenizer_id}, drops special tokens, and returns the TF-IDF weights. "
            "Does not run a text pipeline. Load the model first."
        )
    return (
        "Tokenizes the text and returns the attention-mask mean of the last hidden state. "
        "Does not run a text pipeline. Load the model first."
    )


def _add_predict_routes(app: FastAPI, store: ModelStore, spec, slug: str, preprocess: str | None, prefix: str) -> None:
    model_id = spec.model_id
    raw = preprocess is not None
    tag = "review" if raw else "predict"
    action = "Score raw text" if raw else "Score text already in model form"

    async def one(body: TextBody) -> PredictionResult:
        return await _call(
            model_id,
            lambda: service.predict_one(store, spec, body.text, preprocess=preprocess),
        )

    async def batch(body: TextsBody) -> PredictionBatchResult:
        items = await _call(
            model_id,
            lambda: service.predict_many(store, spec, body.texts, preprocess=preprocess),
        )
        return PredictionBatchResult(
            model_id=model_id,
            task=spec.task,
            trained=spec.trained,
            preprocess=preprocess,
            items=items,
        )

    suffix = "text" if raw else "predict"
    one.__name__ = f"{suffix}_{slug}_{model_id}"
    batch.__name__ = f"{suffix}_{slug}_{model_id}_batch"
    description = spec.description
    if raw:
        description += f" This call runs the {preprocess} pipeline, then the tokenizer and the classifier."
    else:
        description += " This call does not preprocess. Send the surface form that checkpoint was trained on."
    description += " Load the model first."
    app.add_api_route(
        prefix,
        one,
        methods=["POST"],
        response_model=PredictionResult,
        tags=[tag],
        summary=f"{action} with {model_id}",
        description=description,
        operation_id=f"{suffix}_{slug}_{model_id}".replace("-", "_"),
    )
    app.add_api_route(
        f"{prefix}/batch",
        batch,
        methods=["POST"],
        response_model=PredictionBatchResult,
        tags=[tag],
        summary=f"{action} in a batch with {model_id}",
        description=description,
        operation_id=f"{suffix}_{slug}_{model_id}_batch".replace("-", "_"),
    )


app = create_app()

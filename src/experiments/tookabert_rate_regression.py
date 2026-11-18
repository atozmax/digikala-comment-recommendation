"""Phase 2: predict the numeric Digikala rating with TookaBERT.

The text stages are the same three functions as the recommendation
classifier: letters, then digits, then emoji, from
``src.experiments.text_v2``. This file does not import or edit that
classifier. Artifacts go under ``experiments/tookabert-rate-regression-2026-11-17``.

The target is ``rate`` on the closed interval [0, 5]. The loss is mean
squared error. The head is a new one-number layer, so the saved
three-class classifier is not loaded into it. The encoder starts from
the same PartAI/TookaBERT-Base weights. Blocks 8–11 train at 2e-5 and
the new head at 1e-4. The bottom eight blocks stay frozen.

Bands used only for the 20% sample and the score tables match the
report: low is [0, 2.5), medium is [2.5, 4), high is [4, 5].

Run from the repository root:

    python -m src.experiments.tookabert_rate_regression run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.experiments.text_v2 import apply_digits, apply_emoji, apply_letters

EXPERIMENT = "tookabert-rate-regression-2026-11-17"
EXPERIMENT_DATE = "2026-11-17"
PHASE = 2
SEED = 2026
TRAIN_FRACTION = 0.20
MAX_LENGTH = 128
BLOCK_LR = 2e-5
HEAD_LR = 1e-4
TOP_BLOCKS = 4
EFFECTIVE_BATCH = 32
PER_DEVICE_BATCH = 8
LAST_EPOCH = 1
RATE_MIN = 0.0
RATE_MAX = 5.0
MEDIUM_FROM = 2.5
HIGH_FROM = 4.0
BANDS = ("low", "medium", "high")
MODEL_NAME = "PartAI/TookaBERT-Base"
CSV_FILE_ID = "1vCkaGbSrXtlKUsdOrguUtuEYobT5vFxr"
WEIGHTS_FOLDER_ID = "1eyfGAVKe-9nVXnJaQQWWVM2_Zb9fLa4u"
INFERENCE_BATCH = 64
CLASSIFICATION_ROOT = Path("experiments") / "tookabert-preproc-v2-2026-11-12"
MASTER_PORT = "29531"

PREPROCESS_FLAGS = {
    "stem": False,
    "remove_english": False,
    "fold_digits_to_persian": True,
    "drop_pure_digit_tokens": True,
    "drop_emoji": True,
    "keep_interior_zwnj": True,
    "remove_url_email": True,
    "cap_char_run": 2,
    "source": "same letter, digit, and emoji stages as the recommendation run",
}


def default_root() -> Path:
    return Path("experiments") / EXPERIMENT


def rating_band(rate: float) -> str:
    """Map one rating onto low, medium, or high.

    ``low`` is below 2.5, ``medium`` is from 2.5 up to but not including
    4, and ``high`` is from 4 through 5.
    """
    if rate < MEDIUM_FROM:
        return "low"
    if rate < HIGH_FROM:
        return "medium"
    return "high"


def prepare_body(text: str) -> str:
    """Run the three text stages in the same order as the classifier."""
    return apply_emoji(apply_digits(apply_letters(text or "")))


def empty_score() -> dict:
    return {
        "rows": 0,
        "abs_error": 0.0,
        "sq_error": 0.0,
        "clipped_abs_error": 0.0,
        "clipped_sq_error": 0.0,
        "y_sum": 0.0,
        "y_sq": 0.0,
        "pred_sum": 0.0,
        "bands": {name: {"rows": 0, "abs_error": 0.0, "sq_error": 0.0} for name in BANDS},
    }


def add_score(left: dict, right: dict) -> dict:
    merged = empty_score()
    for key in ("rows", "abs_error", "sq_error", "clipped_abs_error", "clipped_sq_error", "y_sum", "y_sq", "pred_sum"):
        merged[key] = left[key] + right[key]
    for name in BANDS:
        for key in ("rows", "abs_error", "sq_error"):
            merged["bands"][name][key] = left["bands"][name][key] + right["bands"][name][key]
    return merged


def accumulate_score(totals: dict, truth: float, prediction: float) -> None:
    """Add one review. Clipped error uses the prediction forced into [0, 5]."""
    error = prediction - truth
    clipped = min(max(prediction, RATE_MIN), RATE_MAX)
    clipped_error = clipped - truth
    totals["rows"] += 1
    totals["abs_error"] += abs(error)
    totals["sq_error"] += error * error
    totals["clipped_abs_error"] += abs(clipped_error)
    totals["clipped_sq_error"] += clipped_error * clipped_error
    totals["y_sum"] += truth
    totals["y_sq"] += truth * truth
    totals["pred_sum"] += prediction
    band = totals["bands"][rating_band(truth)]
    band["rows"] += 1
    band["abs_error"] += abs(error)
    band["sq_error"] += error * error


def regression_metrics(totals: dict) -> dict[str, float | int | dict]:
    """MAE, RMSE, and R² from summed errors. R² is 0 when every rating is equal."""
    rows = int(totals["rows"])
    if rows == 0:
        raise ValueError("regression metrics need at least one row")
    mse = totals["sq_error"] / rows
    clipped_mse = totals["clipped_sq_error"] / rows
    ss_tot = totals["y_sq"] - (totals["y_sum"] ** 2) / rows
    r2 = 0.0 if ss_tot == 0 else 1.0 - (totals["sq_error"] / ss_tot)
    report: dict[str, float | int | dict] = {
        "rows": rows,
        "mae": totals["abs_error"] / rows,
        "rmse": mse ** 0.5,
        "mse": mse,
        "r2": r2,
        "clipped_mae": totals["clipped_abs_error"] / rows,
        "clipped_rmse": clipped_mse ** 0.5,
        "mean_rate": totals["y_sum"] / rows,
        "mean_prediction": totals["pred_sum"] / rows,
    }
    for name in BANDS:
        band = totals["bands"][name]
        count = int(band["rows"])
        report[f"{name}_rows"] = count
        report[f"{name}_mae"] = (band["abs_error"] / count) if count else 0.0
        report[f"{name}_rmse"] = ((band["sq_error"] / count) ** 0.5) if count else 0.0
    return report


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(root: Path, message: str) -> None:
    line = f"{_now()} {message}"
    print(line, flush=True)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _status(root: Path, stage: str, state: str, message: str, **progress) -> None:
    payload = {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "task": "rate_regression",
        "stage": stage,
        "state": state,
        "message": message,
        "updated_at": _now(),
        "progress": progress,
    }
    _write_json(root / "status.json", payload)
    _log(root, f"{stage} {state}: {message}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _done(stage_dir: Path) -> bool:
    return (stage_dir / "DONE").is_file()


def _mark_done(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "DONE").write_text(_now() + "\n", encoding="utf-8")


def _stage_config(stage: str, **extra) -> dict:
    payload = {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "date": EXPERIMENT_DATE,
        "task": "rate_regression",
        "target": "rate",
        "rate_min": RATE_MIN,
        "rate_max": RATE_MAX,
        "bands": {
            "low": f"[{RATE_MIN}, {MEDIUM_FROM})",
            "medium": f"[{MEDIUM_FROM}, {HIGH_FROM})",
            "high": f"[{HIGH_FROM}, {RATE_MAX}]",
        },
        "loss": "mean_squared_error",
        "seed": SEED,
        "stage": stage,
        "preprocessing": PREPROCESS_FLAGS,
        "model_name": MODEL_NAME,
        "top_blocks": TOP_BLOCKS,
        "block_lr": BLOCK_LR,
        "head_lr": HEAD_LR,
        "max_length": MAX_LENGTH,
        "effective_batch_size": EFFECTIVE_BATCH,
    }
    payload.update(extra)
    return payload


def _map_bodies(frame: pl.DataFrame, function, root: Path, stage: str) -> pl.DataFrame:
    bodies = frame.get_column("body").to_list()
    total = len(bodies)
    rewritten: list[str] = []
    started = time.perf_counter()
    for index, body in enumerate(bodies, start=1):
        rewritten.append(function(body or ""))
        if index == total or index % 200_000 == 0:
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0
            _status(root, stage, "running", f"row {index} of {total} ({rate:.0f} rows/s)", done=index, total=total)
    return frame.with_columns(pl.Series("body", rewritten))


def _resolve_weights(directory: Path) -> Path | None:
    if (directory / "model.safetensors").is_file():
        return directory
    nested = sorted(directory.glob("*/model.safetensors"))
    if len(nested) == 1:
        return nested[0].parent
    return None


def _read_csv(csv_path: Path) -> pl.DataFrame:
    try:
        frame = pl.read_csv(csv_path, encoding="utf8-lossy", infer_schema_length=10_000)
    except TypeError:
        frame = pl.read_csv(csv_path, infer_schema_length=10_000)
    if frame.columns and frame.columns[0].startswith("\ufeff"):
        frame = frame.rename({frame.columns[0]: frame.columns[0].lstrip("\ufeff")})
    needed = {"id", "body", "rate"}
    missing = needed.difference(frame.columns)
    if missing:
        raise RuntimeError(f"CSV is missing columns: {sorted(missing)}")
    columns = [
        pl.col("id").cast(pl.Int64),
        pl.col("body").cast(pl.String).fill_null(""),
        pl.col("rate").cast(pl.Float64, strict=False),
    ]
    if "product_id" in frame.columns:
        columns.append(pl.col("product_id").cast(pl.Int64, strict=False))
    if "recommendation_status" in frame.columns:
        columns.append(pl.col("recommendation_status").cast(pl.String))
    return frame.select(columns).filter(
        pl.col("rate").is_not_null()
        & (pl.col("rate") >= RATE_MIN)
        & (pl.col("rate") <= RATE_MAX)
        & (pl.col("body") != "")
    )


def _rated_parquet(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = {"id", "body", "rate"}.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    return frame.filter(
        pl.col("rate").is_not_null()
        & (pl.col("rate") >= RATE_MIN)
        & (pl.col("rate") <= RATE_MAX)
        & (pl.col("body") != "")
    )


def _find_csv() -> Path | None:
    candidates = [
        CLASSIFICATION_ROOT / "download" / "digikala-comments-preprocessed.csv",
        Path("data/processed/digikala-comments-preprocessed.csv"),
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return None


def _find_prepared_reviews() -> Path | None:
    path = CLASSIFICATION_ROOT / "preprocess-emoji" / "reviews.parquet"
    if path.is_file():
        return path
    return None


def download_csv(root: Path) -> Path:
    stage_dir = root / "download"
    csv_path = stage_dir / "digikala-comments-preprocessed.csv"
    if csv_path.is_file() and csv_path.stat().st_size > 1_000_000:
        _log(root, f"skip download ({csv_path})")
        return csv_path
    existing = _find_csv()
    if existing is not None:
        _log(root, f"using the comments file already on disk ({existing})")
        _write_json(stage_dir / "config.json", _stage_config("download", source=str(existing), copied=False))
        _mark_done(stage_dir)
        return existing
    stage_dir.mkdir(parents=True, exist_ok=True)
    archive = stage_dir / "digikala-comments-preprocessed.download"
    if not (archive.is_file() and zipfile.is_zipfile(archive)):
        _status(root, "download", "running", f"downloading Drive file {CSV_FILE_ID}")
        import gdown

        gdown.download(id=CSV_FILE_ID, output=str(archive), quiet=False, resume=True)
    with zipfile.ZipFile(archive) as bundle:
        names = [
            name
            for name in bundle.namelist()
            if name.endswith(".csv") and not name.startswith("__MACOSX/") and not Path(name).name.startswith("._")
        ]
        if len(names) != 1:
            raise RuntimeError(f"expected one CSV in the zip, found {names}")
        with bundle.open(names[0]) as source, csv_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    _write_json(stage_dir / "config.json", _stage_config("download", drive_file_id=CSV_FILE_ID, bytes=csv_path.stat().st_size))
    _mark_done(stage_dir)
    return csv_path


def preprocess_stage(root: Path, csv_path: Path) -> Path:
    """Write the three text stages under this experiment. Return the emoji parquet."""
    emoji_dir = root / "preprocess-emoji"
    emoji_path = emoji_dir / "reviews.parquet"
    if _done(emoji_dir) and emoji_path.is_file():
        _log(root, f"skip preprocessing ({emoji_path})")
        return emoji_path
    letters_dir = root / "preprocess-letters"
    digits_dir = root / "preprocess-digits"
    if _done(letters_dir):
        current = pl.read_parquet(letters_dir / "reviews.parquet")
    else:
        _status(root, "preprocess-letters", "running", f"reading {csv_path}")
        current = _read_csv(csv_path)
        current = _map_bodies(current, apply_letters, root, "preprocess-letters")
        letters_dir.mkdir(parents=True, exist_ok=True)
        current.write_parquet(letters_dir / "reviews.parquet")
        _write_json(letters_dir / "config.json", _stage_config("preprocess-letters", rows=current.height))
        _mark_done(letters_dir)
    if _done(digits_dir):
        current = pl.read_parquet(digits_dir / "reviews.parquet")
    else:
        current = _map_bodies(current, apply_digits, root, "preprocess-digits")
        digits_dir.mkdir(parents=True, exist_ok=True)
        current.write_parquet(digits_dir / "reviews.parquet")
        _write_json(digits_dir / "config.json", _stage_config("preprocess-digits", rows=current.height))
        _mark_done(digits_dir)
    current = _map_bodies(current, apply_emoji, root, "preprocess-emoji")
    before_drop = current.height
    current = current.filter(pl.col("body") != "")
    emoji_dir.mkdir(parents=True, exist_ok=True)
    current.write_parquet(emoji_path)
    _write_json(
        emoji_dir / "config.json",
        _stage_config(
            "preprocess-emoji",
            rows_before_empty_drop=before_drop,
            empty_bodies_deleted=before_drop - current.height,
            rows=current.height,
        ),
    )
    _mark_done(emoji_dir)
    _status(root, "preprocess-emoji", "done", f"rows {current.height}")
    return emoji_path


def reviews_stage(root: Path) -> Path:
    """Return the parquet this run scores. Reuse the classifier text when it is already written."""
    own = root / "preprocess-emoji" / "reviews.parquet"
    if own.is_file() and _done(root / "preprocess-emoji"):
        return own
    prepared = _find_prepared_reviews()
    if prepared is not None:
        _log(root, f"reusing prepared reviews ({prepared})")
        _write_json(
            root / "preprocess-emoji" / "config.json",
            _stage_config("preprocess-emoji", source=str(prepared), copied=False),
        )
        _mark_done(root / "preprocess-emoji")
        return prepared
    return preprocess_stage(root, download_csv(root))


def _band_counts(frame: pl.DataFrame) -> dict[str, int]:
    counts = {name: 0 for name in BANDS}
    grouped = frame.group_by("band").len()
    for name, count in zip(grouped.get_column("band").to_list(), grouped.get_column("len").to_list()):
        counts[str(name)] = int(count)
    return counts


def _with_band(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("rate") < MEDIUM_FROM)
        .then(pl.lit("low"))
        .when(pl.col("rate") < HIGH_FROM)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
        .alias("band")
    )


def sample_stage(root: Path, reviews_path: Path) -> Path:
    stage_dir = root / "sample"
    ids_path = stage_dir / "train_ids.parquet"
    if _done(stage_dir) and ids_path.is_file():
        _log(root, f"skip sample ({ids_path})")
        return ids_path
    _status(root, "sample", "running", "drawing 20% of each rating band")
    frame = _with_band(_rated_parquet(reviews_path)).sort("id")
    before = _band_counts(frame)
    parts: list[pl.DataFrame] = []
    after: dict[str, int] = {}
    for name in BANDS:
        group = frame.filter(pl.col("band") == name)
        n_keep = round(TRAIN_FRACTION * group.height)
        after[name] = n_keep
        if n_keep:
            parts.append(group.sample(n=n_keep, shuffle=True, seed=SEED))
    train = pl.concat(parts).select("id").sort("id") if parts else pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)})
    stage_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(ids_path)
    _write_json(
        stage_dir / "config.json",
        _stage_config(
            "sample",
            reviews=str(reviews_path),
            train_fraction=TRAIN_FRACTION,
            band_counts_before=before,
            band_counts_train=after,
            train_rows=train.height,
            heldout_rows=frame.height - train.height,
            rated_rows=frame.height,
        ),
    )
    _mark_done(stage_dir)
    _status(root, "sample", "done", f"train {train.height} held-out {frame.height - train.height} bands {after}")
    return ids_path


def model_stage(root: Path) -> Path:
    stage_dir = root / "model"
    found = _resolve_weights(stage_dir / "start")
    if found is not None:
        _log(root, f"skip model ({found})")
        return found
    sibling = _resolve_weights(CLASSIFICATION_ROOT / "model" / "start")
    if sibling is None:
        sibling = _resolve_weights(CLASSIFICATION_ROOT / "model")
    if sibling is not None:
        _log(root, f"using the encoder already on disk ({sibling})")
        _write_json(stage_dir / "config.json", _stage_config("model", source=str(sibling), copied=False))
        _mark_done(stage_dir)
        return sibling
    stage_dir.mkdir(parents=True, exist_ok=True)
    weights = stage_dir / "start"
    _status(root, "model", "running", f"downloading weights folder {WEIGHTS_FOLDER_ID}")
    import gdown

    gdown.download_folder(id=WEIGHTS_FOLDER_ID, output=str(weights), quiet=False, resume=True)
    found = _resolve_weights(weights)
    if found is None:
        raise RuntimeError(f"model.safetensors not found under {weights}")
    _write_json(stage_dir / "config.json", _stage_config("model", weights_folder_id=WEIGHTS_FOLDER_ID))
    _mark_done(stage_dir)
    return found


def _batch_plan(gpu_count: int) -> tuple[int, int]:
    cards = max(gpu_count, 1)
    accumulation = max(1, EFFECTIVE_BATCH // (PER_DEVICE_BATCH * cards))
    return PER_DEVICE_BATCH, accumulation


def tokenize_stage(root: Path, reviews_path: Path, ids_path: Path, weights: Path) -> Path:
    stage_dir = root / "sample" / "tokenized-train"
    if (stage_dir / "dataset_info.json").is_file() or (stage_dir / "state.json").is_file():
        _log(root, f"skip tokenize ({stage_dir})")
        return stage_dir
    _status(root, "sample", "running", "tokenizing the 20% training rows")
    from datasets import Dataset
    from transformers import AutoTokenizer

    train_ids = pl.read_parquet(ids_path).select("id")
    frame = _rated_parquet(reviews_path).join(train_ids, on="id", how="inner").select("body", "rate")
    tokenizer = AutoTokenizer.from_pretrained(weights)

    def encode(batch):
        return tokenizer(batch["body"], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_dict(
        {"body": frame.get_column("body").to_list(), "label": frame.get_column("rate").to_list()}
    )
    tokenized = dataset.map(encode, batched=True, batch_size=2048, remove_columns=["body"])
    tokenized.save_to_disk(str(stage_dir))
    _status(root, "sample", "done", f"tokenized train rows {tokenized.num_rows}")
    return stage_dir


def _unfreeze(model) -> int:
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("bert.")
    layers = model.bert.encoder.layer
    if TOP_BLOCKS > len(layers):
        raise ValueError(f"model has {len(layers)} blocks, requested {TOP_BLOCKS}")
    for layer in layers[-TOP_BLOCKS:]:
        for param in layer.parameters():
            param.requires_grad = True
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _parameter_groups(model):
    head, block = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            block.append(param)
        else:
            head.append(param)
    return [{"params": block, "lr": BLOCK_LR}, {"params": head, "lr": HEAD_LR}]


def _load_regressor(weights: Path):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        weights,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )


def train_epoch(root: Path, epoch: int, weights: Path, tokenized_dir: Path) -> None:
    import inspect

    import torch
    from datasets import load_from_disk
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    epoch_dir = root / f"epoch-{epoch}"
    if (epoch_dir / "WEIGHTS_READY").is_file():
        _log(root, f"skip train epoch {epoch}; weights already saved")
        return
    epoch_dir.mkdir(parents=True, exist_ok=True)
    (epoch_dir / "trainer").mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    previous = weights if epoch == 1 else root / f"epoch-{epoch - 1}"
    tokenizer = AutoTokenizer.from_pretrained(previous if epoch > 1 else weights)
    model = _load_regressor(previous)
    trainable = _unfreeze(model)
    if rank == 0:
        _status(root, f"epoch-{epoch}", "running", f"trainable parameters {trainable}; loss mean squared error")

    table = load_from_disk(str(tokenized_dir))

    class Encoded(Dataset):
        def __len__(self):
            return table.num_rows

        def __getitem__(self, index):
            row = table[index]
            return {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
                "labels": float(row["label"]),
            }

    gpu_count = int(os.environ.get("WORLD_SIZE", "1"))
    per_device, accumulation = _batch_plan(gpu_count)
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(rank)[0] >= 8
    trainer_dir = str(epoch_dir / "trainer")
    shared = dict(
        output_dir=trainer_dir,
        learning_rate=BLOCK_LR,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
    )
    try:
        args = TrainingArguments(eval_strategy="no", **shared)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="no", **shared)

    class RegressionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels").float().view(-1)
            outputs = model(**inputs)
            prediction = outputs.logits.view(-1)
            loss = torch.nn.functional.mse_loss(prediction, labels)
            return (loss, outputs) if return_outputs else loss

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if rank != 0 or not logs or "loss" not in logs:
                return
            loss = float(logs["loss"])
            _status(
                root,
                f"epoch-{epoch}",
                "training",
                f"step {state.global_step} of {state.max_steps} loss {loss:.4f}",
                step=state.global_step,
                max_steps=state.max_steps,
                loss=loss,
            )

    token_kwargs = {}
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        token_kwargs["processing_class"] = tokenizer
    else:
        token_kwargs["tokenizer"] = tokenizer
    groups = _parameter_groups(model)
    trainer = RegressionTrainer(
        model=model,
        args=args,
        train_dataset=Encoded(),
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[Progress()],
        **token_kwargs,
    )

    def create_optimizer():
        trainer.optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        return trainer.optimizer

    trainer.create_optimizer = create_optimizer
    last = get_last_checkpoint(trainer_dir)
    trainer.train(resume_from_checkpoint=last)
    if trainer.is_world_process_zero():
        trainer.save_model(str(epoch_dir))
        tokenizer.save_pretrained(epoch_dir)
        _write_json(
            epoch_dir / "run_config.json",
            _stage_config(
                f"epoch-{epoch}",
                trainable_parameters=trainable,
                per_device_batch_size=per_device,
                gradient_accumulation_steps=accumulation,
                world_size=gpu_count,
                bf16=use_bf16,
                started_from=str(previous),
            ),
        )
        (epoch_dir / "WEIGHTS_READY").write_text(_now() + "\n", encoding="utf-8")
        _status(root, f"epoch-{epoch}", "weights-saved", str(epoch_dir))


def _score_classes():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return AutoModelForSequenceClassification, AutoTokenizer


def _score_shard(
    device: str,
    texts: list[str],
    rates: list[float],
    weights: Path,
    root: Path,
    stage: str,
    shard: int,
    shards: int,
    model_cls=None,
    tokenizer_cls=None,
) -> dict:
    import torch

    if model_cls is None or tokenizer_cls is None:
        model_cls, tokenizer_cls = _score_classes()
    totals = empty_score()
    if not texts:
        return totals
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
    tokenizer = tokenizer_cls.from_pretrained(weights)
    model = model_cls.from_pretrained(weights).to(device)
    model.eval()
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    started = time.perf_counter()
    for start in range(0, len(texts), INFERENCE_BATCH):
        batch_texts = texts[start : start + INFERENCE_BATCH]
        batch_rates = rates[start : start + INFERENCE_BATCH]
        encoded = tokenizer(batch_texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(**encoded).logits
            else:
                logits = model(**encoded).logits
        predicted = logits.view(-1).float().cpu().tolist()
        for truth, guess in zip(batch_rates, predicted):
            accumulate_score(totals, float(truth), float(guess))
        done = min(start + INFERENCE_BATCH, len(texts))
        if shard == 0 and (done == len(texts) or start // INFERENCE_BATCH % 50 == 0):
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0
            _status(
                root,
                stage,
                "scoring",
                f"shard {shard + 1}/{shards} row {done} of {len(texts)} ({rate:.0f} rows/s on this shard)",
                done=done,
                total=len(texts),
                shard=shard,
            )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    _log(root, f"{stage} scoring: shard {shard + 1}/{shards} finished {len(texts)} rows")
    return totals


def score_split(root: Path, epoch: int, reviews_path: Path, ids_path: Path, which: str) -> dict:
    import torch

    epoch_dir = root / f"epoch-{epoch}"
    target = epoch_dir / ("heldout_metrics.json" if which == "heldout" else "metrics.json")
    if target.is_file():
        _log(root, f"skip {which} score for epoch {epoch}")
        return _read_json(target)
    train_ids = pl.read_parquet(ids_path).select("id")
    frame = _rated_parquet(reviews_path).select("id", "body", "rate")
    if which == "heldout":
        frame = frame.join(train_ids, on="id", how="anti")
    texts = frame.get_column("body").to_list()
    rates = [float(value) for value in frame.get_column("rate").to_list()]
    if torch.cuda.is_available():
        torch.cuda.init()
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())] or ["cpu"]
    shards = max(len(devices), 1)
    pieces = [(texts[index::shards], rates[index::shards]) for index in range(shards)]
    model_cls, tokenizer_cls = _score_classes()
    _status(root, f"epoch-{epoch}", "scoring", f"{which} rows {len(texts)} on {shards} device(s)")
    results: list[dict | None] = [None] * shards
    errors: list[BaseException] = []

    def run_shard(index: int, device: str) -> None:
        try:
            shard_texts, shard_rates = pieces[index]
            results[index] = _score_shard(
                device,
                shard_texts,
                shard_rates,
                epoch_dir,
                root,
                f"epoch-{epoch}",
                index,
                shards,
                model_cls,
                tokenizer_cls,
            )
        except BaseException as exc:
            errors.append(exc)
            _log(root, f"epoch-{epoch} scoring shard {index + 1}/{shards} failed: {exc}")

    threads = [threading.Thread(target=run_shard, args=(index, device)) for index, device in enumerate(devices)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    totals = empty_score()
    for piece in results:
        if piece is None:
            raise RuntimeError(f"{which} scoring thread failed")
        totals = add_score(totals, piece)
    metrics = regression_metrics(totals)
    metrics["split"] = "heldout_80" if which == "heldout" else "full_rated"
    metrics["epoch"] = epoch
    _write_json(target, metrics)
    _status(
        root,
        f"epoch-{epoch}",
        "scored",
        f"{which} MAE {metrics['mae']:.4f} RMSE {metrics['rmse']:.4f} R2 {metrics['r2']:.4f}",
    )
    return metrics


def _launch_train(root: Path, epoch: int) -> None:
    import torch

    if (root / f"epoch-{epoch}" / "WEIGHTS_READY").is_file():
        _log(root, f"epoch {epoch} weights present")
        return
    gpu_count = torch.cuda.device_count()
    command = [sys.executable, "-m", "src.experiments.tookabert_rate_regression", "train", "--epoch", str(epoch), "--root", str(root)]
    if gpu_count > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpu_count),
            "--master_port",
            MASTER_PORT,
            "-m",
            "src.experiments.tookabert_rate_regression",
            "train",
            "--epoch",
            str(epoch),
            "--root",
            str(root),
        ]
    _log(root, " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _weights_ready(root: Path, epoch: int) -> bool:
    return (root / f"epoch-{epoch}" / "WEIGHTS_READY").is_file()


def _epoch_scored(root: Path, epoch: int) -> bool:
    folder = root / f"epoch-{epoch}"
    return (folder / "heldout_metrics.json").is_file() and (folder / "metrics.json").is_file()


def _score_if_needed(root: Path, epoch: int, reviews_path: Path, ids_path: Path) -> None:
    if _epoch_scored(root, epoch):
        _log(root, f"skip score epoch {epoch}; metrics already saved")
        return
    score_split(root, epoch, reviews_path, ids_path, "heldout")
    score_split(root, epoch, reviews_path, ids_path, "full")
    _mark_done(root / f"epoch-{epoch}")


def _kept_epoch(root: Path, through: int) -> int:
    kept = 0
    best = None
    for epoch in range(1, through + 1):
        path = root / f"epoch-{epoch}" / "heldout_metrics.json"
        if not path.is_file():
            continue
        score = float(_read_json(path)["mae"])
        if best is None or score < best:
            best = score
            kept = epoch
    return kept


def run(root: Path) -> None:
    anchor = root if root.exists() else Path(".")
    free = shutil.disk_usage(anchor).free
    if free < 30 * 1024**3:
        raise SystemExit(f"need at least 30 GB free on {root}, found {free / 1024**3:.1f} GB")
    reviews_path = reviews_stage(root)
    ids_path = sample_stage(root, reviews_path)
    weights = model_stage(root)
    tokenize_stage(root, reviews_path, ids_path, weights)
    for epoch in range(1, LAST_EPOCH + 1):
        if epoch > 1 and not _weights_ready(root, epoch - 1):
            _log(root, f"stopped before epoch {epoch}; epoch {epoch - 1} has no saved weights")
            break
        if _weights_ready(root, epoch):
            _log(root, f"epoch {epoch} weights already saved")
        else:
            _log(root, f"training epoch {epoch}")
            _launch_train(root, epoch)
        _score_if_needed(root, epoch, reviews_path, ids_path)
    kept = _kept_epoch(root, LAST_EPOCH)
    _write_json(
        root / "kept.json",
        {"kept_epoch": kept, "directory": f"epoch-{kept}", "reason": "lowest held-out MAE"},
    )
    _status(root, "run", "done", f"kept epoch {kept}")


def print_status(root: Path) -> None:
    status_path = root / "status.json"
    if not status_path.is_file():
        print(f"no status yet at {status_path}")
        return
    print(status_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.experiments.tookabert_rate_regression")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("command", choices=("run", "train", "status"), nargs="?", default="run")
    parser.add_argument("--epoch", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root
    if args.command == "status":
        print_status(root)
        return
    if args.command == "train":
        weights = model_stage(root)
        if args.epoch > 1:
            weights = root / f"epoch-{args.epoch - 1}"
        train_epoch(root, args.epoch, weights, root / "sample" / "tokenized-train")
        return
    run(root)


if __name__ == "__main__":
    main()

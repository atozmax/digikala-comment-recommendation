"""TookaBERT preprocessing v2 on the local GPU server.

One command downloads the shared comments file, writes each preprocessing
stage, trains the top four blocks, and scores the held-out 80% and the
full labeled set. A finished stage has a ``DONE`` file and is skipped on
the next start.

The recipe matches ``plans/new-experiments-plan.md``: seed 2026, length
128, class-weighted loss, blocks 8–11 at 2e-5, classifier at 1e-4,
effective batch 32. Two GPUs split that batch. Ampere cards train in
bf16. A second epoch runs only when the first epoch's held-out macro F1
is above the saved full-data validation score 0.6722, and its weights
are kept only when its own held-out macro F1 is higher.

Watch the run with ``python -m src.experiments.tookabert_v2 status``
and ``nvidia-smi``.
"""

from __future__ import annotations

import argparse
import hashlib
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

from src.experiments.text_v2 import (
    apply_digits,
    apply_emoji,
    apply_letters,
    confusion_metrics,
)

EXPERIMENT = "tookabert-preproc-v2-2026-11-12"
EXPERIMENT_DATE = "2026-11-12"
SEED = 2026
TRAIN_FRACTION = 0.20
MAX_LENGTH = 128
BLOCK_LR = 2e-5
HEAD_LR = 1e-4
TOP_BLOCKS = 4
EFFECTIVE_BATCH = 32
PER_DEVICE_BATCH = 8
EPOCH1_HELD_OUT_BAR = 0.6722
LAST_EPOCH = 7
LABELS = ("recommended", "not_recommended", "no_idea")
LABEL_CODE = {label: index for index, label in enumerate(LABELS)}
MODEL_NAME = "PartAI/TookaBERT-Base"
CSV_FILE_ID = "1vCkaGbSrXtlKUsdOrguUtuEYobT5vFxr"
WEIGHTS_FOLDER_ID = "1eyfGAVKe-9nVXnJaQQWWVM2_Zb9fLa4u"
INFERENCE_BATCH = 64

PREPROCESS_FLAGS = {
    "stem": False,
    "remove_english": False,
    "fold_digits_to_persian": True,
    "drop_pure_digit_tokens": True,
    "drop_emoji": True,
    "keep_interior_zwnj": True,
    "remove_url_email": True,
    "cap_char_run": 2,
    "source": "shared preprocessed CSV, Hazm already applied upstream, not applied again",
}


def default_root() -> Path:
    return Path("experiments") / EXPERIMENT


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
    (stage_dir / "DONE").write_text(_now() + "\n", encoding="utf-8")


def _stage_config(stage: str, **extra) -> dict:
    payload = {
        "experiment": EXPERIMENT,
        "date": EXPERIMENT_DATE,
        "seed": SEED,
        "label_map": LABEL_CODE,
        "stage": stage,
        "preprocessing": PREPROCESS_FLAGS,
        "model_name": MODEL_NAME,
        "top_blocks": TOP_BLOCKS,
        "block_lr": BLOCK_LR,
        "head_lr": HEAD_LR,
        "max_length": MAX_LENGTH,
        "effective_batch_size": EFFECTIVE_BATCH,
        "epoch2_heldout_macro_f1_bar": EPOCH1_HELD_OUT_BAR,
    }
    payload.update(extra)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            _status(
                root,
                stage,
                "running",
                f"row {index} of {total} ({rate:.0f} rows/s)",
                done=index,
                total=total,
            )
    return frame.with_columns(pl.Series("body", rewritten))


def _class_counts(frame: pl.DataFrame) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    grouped = frame.group_by("recommendation_status").len()
    for status, count in zip(
        grouped.get_column("recommendation_status").to_list(),
        grouped.get_column("len").to_list(),
        strict=True,
    ):
        if status in counts:
            counts[str(status)] = int(count)
    return counts


def download_stage(root: Path) -> Path:
    stage_dir = root / "download"
    csv_path = stage_dir / "digikala-comments-preprocessed.csv"
    if csv_path.is_file() and csv_path.stat().st_size > 1_000_000:
        _log(root, f"skip download ({csv_path})")
        if not _done(stage_dir):
            _write_json(
                stage_dir / "config.json",
                _stage_config("download", drive_file_id=CSV_FILE_ID, bytes=csv_path.stat().st_size, file_kind="local"),
            )
            _mark_done(stage_dir)
        return csv_path
    stage_dir.mkdir(parents=True, exist_ok=True)
    archive = stage_dir / "digikala-comments-preprocessed.download"
    if not (archive.is_file() and zipfile.is_zipfile(archive)):
        _status(root, "download", "running", f"downloading Drive file {CSV_FILE_ID}")
        import gdown

        gdown.download(id=CSV_FILE_ID, output=str(archive), quiet=False, resume=True)
    else:
        _log(root, f"using the zip already on disk ({archive})")
    digest = _sha256(archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            names = [
                name
                for name in bundle.namelist()
                if name.endswith(".csv")
                and not name.startswith("__MACOSX/")
                and not Path(name).name.startswith("._")
            ]
            if len(names) != 1:
                raise RuntimeError(f"expected one CSV in the zip, found {names}")
            with bundle.open(names[0]) as source, csv_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        kind = "zip"
    else:
        archive.replace(csv_path)
        kind = "csv"
    config = _stage_config(
        "download",
        drive_file_id=CSV_FILE_ID,
        sha256=digest,
        file_kind=kind,
        bytes=csv_path.stat().st_size,
    )
    _write_json(stage_dir / "config.json", config)
    _mark_done(stage_dir)
    _status(root, "download", "done", f"sha256 {digest} bytes {csv_path.stat().st_size}")
    return csv_path


def _labeled_frame(csv_path: Path) -> pl.DataFrame:
    try:
        frame = pl.read_csv(csv_path, encoding="utf8-lossy", infer_schema_length=10_000)
    except TypeError:
        frame = pl.read_csv(csv_path, infer_schema_length=10_000)
    if frame.columns and frame.columns[0].startswith("\ufeff"):
        frame = frame.rename({frame.columns[0]: frame.columns[0].lstrip("\ufeff")})
    needed = {"id", "body", "product_id", "recommendation_status", "rate"}
    missing = needed.difference(frame.columns)
    if missing:
        raise RuntimeError(f"CSV is missing columns: {sorted(missing)}")
    return frame.select(
        pl.col("id").cast(pl.Int64),
        pl.col("body").cast(pl.String).fill_null(""),
        pl.col("product_id").cast(pl.Int64, strict=False),
        pl.col("recommendation_status").cast(pl.String),
        pl.col("rate").cast(pl.Float64, strict=False),
    ).filter(pl.col("recommendation_status").is_in(list(LABELS)) & (pl.col("body") != ""))


def preprocess_stage(root: Path, csv_path: Path) -> Path:
    """Write letters, digits, and emoji parquet files. Return the emoji file."""
    letters_dir = root / "preprocess-letters"
    digits_dir = root / "preprocess-digits"
    emoji_dir = root / "preprocess-emoji"
    emoji_path = emoji_dir / "reviews.parquet"
    if _done(emoji_dir) and emoji_path.is_file():
        _log(root, f"skip preprocessing ({emoji_path})")
        return emoji_path

    if _done(letters_dir):
        current = pl.read_parquet(letters_dir / "reviews.parquet")
    else:
        _status(root, "preprocess-letters", "running", f"reading {csv_path}")
        current = _labeled_frame(csv_path)
        before = _class_counts(current)
        current = _map_bodies(current, apply_letters, root, "preprocess-letters")
        letters_dir.mkdir(parents=True, exist_ok=True)
        current.write_parquet(letters_dir / "reviews.parquet")
        _write_json(
            letters_dir / "config.json",
            _stage_config("preprocess-letters", rows=current.height, class_counts=before),
        )
        _mark_done(letters_dir)

    if not _done(digits_dir):
        current = _map_bodies(current, apply_digits, root, "preprocess-digits")
        digits_dir.mkdir(parents=True, exist_ok=True)
        current.write_parquet(digits_dir / "reviews.parquet")
        _write_json(
            digits_dir / "config.json",
            _stage_config("preprocess-digits", rows=current.height, class_counts=_class_counts(current)),
        )
        _mark_done(digits_dir)
    else:
        current = pl.read_parquet(digits_dir / "reviews.parquet")

    current = _map_bodies(current, apply_emoji, root, "preprocess-emoji")
    before_drop = current.height
    current = current.filter(pl.col("body") != "")
    deleted = before_drop - current.height
    emoji_dir.mkdir(parents=True, exist_ok=True)
    current.write_parquet(emoji_path)
    _write_json(
        emoji_dir / "config.json",
        _stage_config(
            "preprocess-emoji",
            rows_before_empty_drop=before_drop,
            empty_bodies_deleted=deleted,
            rows=current.height,
            class_counts=_class_counts(current),
        ),
    )
    _mark_done(emoji_dir)
    _status(root, "preprocess-emoji", "done", f"rows {current.height}, empty bodies deleted {deleted}")
    return emoji_path


def sample_stage(root: Path, reviews_path: Path) -> Path:
    stage_dir = root / "sample"
    ids_path = stage_dir / "train_ids.parquet"
    if _done(stage_dir) and ids_path.is_file():
        _log(root, f"skip sample ({ids_path})")
        return ids_path
    _status(root, "sample", "running", "drawing 20% of each class")
    frame = pl.read_parquet(reviews_path).sort("id")
    before = _class_counts(frame)
    parts: list[pl.DataFrame] = []
    after: dict[str, int] = {}
    for label in LABELS:
        group = frame.filter(pl.col("recommendation_status") == label)
        n_keep = round(TRAIN_FRACTION * group.height)
        after[label] = n_keep
        if n_keep:
            parts.append(group.sample(n=n_keep, shuffle=True, seed=SEED))
    train = pl.concat(parts).select("id").sort("id") if parts else pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)})
    stage_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(ids_path)
    _write_json(
        stage_dir / "config.json",
        _stage_config(
            "sample",
            train_fraction=TRAIN_FRACTION,
            class_counts_before=before,
            class_counts_train=after,
            train_rows=train.height,
            heldout_rows=frame.height - train.height,
        ),
    )
    _mark_done(stage_dir)
    _status(
        root,
        "sample",
        "done",
        f"train {train.height} held-out {frame.height - train.height} counts {after}",
    )
    return ids_path


def _resolve_weights(directory: Path) -> Path | None:
    if (directory / "model.safetensors").is_file():
        return directory
    nested = sorted(directory.glob("*/model.safetensors"))
    if len(nested) == 1:
        return nested[0].parent
    return None


def model_stage(root: Path) -> Path:
    stage_dir = root / "model"
    weights = stage_dir / "start"
    found = _resolve_weights(weights)
    if found is not None:
        _log(root, f"skip model ({found})")
        if not _done(stage_dir):
            _write_json(stage_dir / "config.json", _stage_config("model", weights_folder_id=WEIGHTS_FOLDER_ID, local=True))
            _mark_done(stage_dir)
        return found
    stage_dir.mkdir(parents=True, exist_ok=True)
    _status(root, "model", "running", f"downloading weights folder {WEIGHTS_FOLDER_ID}")
    import gdown

    gdown.download_folder(
        id=WEIGHTS_FOLDER_ID,
        output=str(weights),
        quiet=False,
        resume=True,
    )
    found = _resolve_weights(weights)
    if found is None:
        raise RuntimeError(f"model.safetensors not found under {weights}")
    weights = found
    _write_json(
        stage_dir / "config.json",
        _stage_config(
            "model",
            weights_folder_id=WEIGHTS_FOLDER_ID,
            trainable_blocks=list(range(12 - TOP_BLOCKS, 12)),
            frozen_blocks=list(range(0, 12 - TOP_BLOCKS)),
        ),
    )
    _mark_done(stage_dir)
    _status(root, "model", "done", f"weights at {weights}")
    return weights


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

    train_ids = pl.read_parquet(ids_path).select("id")
    frame = (
        pl.read_parquet(reviews_path)
        .join(train_ids, on="id", how="inner")
        .select("body", "recommendation_status")
        .with_columns(pl.col("recommendation_status").replace_strict(LABEL_CODE).alias("label"))
        .select("body", "label")
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(weights)

    def encode(batch):
        return tokenizer(batch["body"], truncation=True, max_length=MAX_LENGTH)

    dataset = Dataset.from_dict({"body": frame.get_column("body").to_list(), "label": frame.get_column("label").to_list()})
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


def train_epoch(root: Path, epoch: int, weights: Path, tokenized_dir: Path) -> None:
    import numpy as np
    import torch
    from datasets import load_from_disk
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    epoch_dir = root / f"epoch-{epoch}"
    if (epoch_dir / "WEIGHTS_READY").is_file():
        _log(root, f"skip train epoch {epoch}; weights already saved")
        return
    epoch_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    previous = weights if epoch == 1 else root / f"epoch-{epoch - 1}"
    tokenizer = AutoTokenizer.from_pretrained(previous)
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(previous)
    trainable = _unfreeze(model)
    if rank == 0:
        _status(root, f"epoch-{epoch}", "running", f"trainable parameters {trainable}")

    table = load_from_disk(str(tokenized_dir))
    labels = np.array(table["label"])
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    weights_tensor = torch.tensor(labels.shape[0] / (3.0 * counts), dtype=torch.float)
    if rank == 0:
        _log(root, f"class weights {[round(float(value), 4) for value in weights_tensor]}")

    class Encoded(Dataset):
        def __len__(self):
            return table.num_rows

        def __getitem__(self, index):
            row = table[index]
            return {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"], "labels": int(row["label"])}

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

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            label_ids = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                label_ids,
                weight=weights_tensor.to(outputs.logits.device),
            )
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

    import inspect

    token_kwargs = {}
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        token_kwargs["processing_class"] = tokenizer
    else:
        token_kwargs["tokenizer"] = tokenizer
    groups = _parameter_groups(model)
    trainer = WeightedTrainer(
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
                class_weights=[float(value) for value in weights_tensor],
            ),
        )
        (epoch_dir / "WEIGHTS_READY").write_text(_now() + "\n", encoding="utf-8")
        _status(root, f"epoch-{epoch}", "weights-saved", str(epoch_dir))


def _ensure_hf_config(weights: Path, fallback: Path) -> None:
    """Put a Hugging Face config back if the experiment file replaced it.

    ``save_pretrained`` writes ``config.json`` with ``model_type``. An older
    run of this script then stored the experiment notes under that same
    name, and ``from_pretrained`` refuses the folder.
    """
    config_path = weights / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if "model_type" in payload:
            return
    candidates = list(weights.glob("trainer/checkpoint-*/config.json"))
    if (fallback / "config.json").is_file():
        candidates.append(fallback / "config.json")
    chosen = None
    chosen_step = -1
    for path in candidates:
        step_text = path.parent.name.removeprefix("checkpoint-")
        step = int(step_text) if step_text.isdigit() else chosen_step
        if chosen is None or step >= chosen_step:
            chosen = path
            chosen_step = step
    if chosen is None:
        raise RuntimeError(f"no model config.json with model_type for {weights}")
    shutil.copyfile(chosen, config_path)


def _score_classes():
    """Import Transformers once. Two threads importing it together can fail."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return AutoModelForSequenceClassification, AutoTokenizer


def _score_shard(
    device: str,
    texts: list[str],
    label_ids: list[int],
    weights: Path,
    root: Path,
    stage: str,
    shard: int,
    shards: int,
    model_cls=None,
    tokenizer_cls=None,
) -> list[list[int]]:
    import torch

    if model_cls is None or tokenizer_cls is None:
        model_cls, tokenizer_cls = _score_classes()
    matrix = [[0, 0, 0] for _ in LABELS]
    if not texts:
        return matrix
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
    tokenizer = tokenizer_cls.from_pretrained(weights)
    model = model_cls.from_pretrained(weights).to(device)
    model.eval()
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    started = time.perf_counter()
    for start in range(0, len(texts), INFERENCE_BATCH):
        batch_texts = texts[start : start + INFERENCE_BATCH]
        batch_labels = label_ids[start : start + INFERENCE_BATCH]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(**encoded).logits
            else:
                logits = model(**encoded).logits
        predicted = logits.argmax(dim=-1).tolist()
        for truth, guess in zip(batch_labels, predicted, strict=True):
            matrix[truth][guess] += 1
        done = min(start + INFERENCE_BATCH, len(texts))
        if done == len(texts) or start // INFERENCE_BATCH % 50 == 0:
            if shard == 0:
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
    return matrix


def _add_matrix(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [[left[row][col] + right[row][col] for col in range(3)] for row in range(3)]


def score_split(root: Path, epoch: int, reviews_path: Path, ids_path: Path, which: str) -> dict:
    """Score ``heldout`` or ``full`` with one replica on each GPU. Return metrics."""
    import torch

    epoch_dir = root / f"epoch-{epoch}"
    _ensure_hf_config(epoch_dir, root / "model" / "start")
    target = epoch_dir / ("heldout_metrics.json" if which == "heldout" else "metrics.json")
    if target.is_file():
        _log(root, f"skip {which} score for epoch {epoch}")
        return _read_json(target)
    train_ids = pl.read_parquet(ids_path).select("id")
    frame = pl.read_parquet(reviews_path).select("id", "body", "recommendation_status")
    if which == "heldout":
        frame = frame.join(train_ids, on="id", how="anti")
    texts = frame.get_column("body").to_list()
    label_ids = [LABEL_CODE[value] for value in frame.get_column("recommendation_status").to_list()]
    if torch.cuda.is_available():
        torch.cuda.init()
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())] or ["cpu"]
    shards = max(len(devices), 1)
    pieces = [(texts[index::shards], label_ids[index::shards]) for index in range(shards)]
    model_cls, tokenizer_cls = _score_classes()
    _status(root, f"epoch-{epoch}", "scoring", f"{which} rows {len(texts)} on {shards} device(s)")
    results: list[list[list[int]] | None] = [None] * shards
    errors: list[BaseException] = []

    def run_shard(index: int, device: str) -> None:
        try:
            shard_texts, shard_labels = pieces[index]
            results[index] = _score_shard(
                device,
                shard_texts,
                shard_labels,
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
    matrix = [[0, 0, 0] for _ in LABELS]
    for piece in results:
        if piece is None:
            raise RuntimeError(f"{which} scoring thread failed")
        matrix = _add_matrix(matrix, piece)
    metrics = confusion_metrics(matrix, LABELS)
    metrics["split"] = "heldout_80" if which == "heldout" else "full_labeled"
    metrics["epoch"] = epoch
    _write_json(target, metrics)
    _status(
        root,
        f"epoch-{epoch}",
        "scored",
        f"{which} accuracy {metrics['accuracy']:.4f} macro F1 {metrics['f1_macro']:.4f} micro F1 {metrics['f1_micro']:.4f}",
    )
    return metrics


def _launch_train(root: Path, epoch: int) -> None:
    import torch

    if (root / f"epoch-{epoch}" / "WEIGHTS_READY").is_file():
        _log(root, f"epoch {epoch} weights present")
        return
    gpu_count = torch.cuda.device_count()
    command = [
        sys.executable,
        "-m",
        "src.experiments.tookabert_v2",
        "train",
        "--epoch",
        str(epoch),
        "--root",
        str(root),
    ]
    if gpu_count > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpu_count),
            "--master_port",
            "29511",
            "-m",
            "src.experiments.tookabert_v2",
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


def _latest_ready_epoch(root: Path) -> int:
    latest = 0
    for epoch in range(1, LAST_EPOCH + 1):
        if _weights_ready(root, epoch):
            latest = epoch
    return latest


def _kept_epoch(root: Path, through: int) -> int:
    kept = 0
    best = -1.0
    for epoch in range(1, through + 1):
        path = root / f"epoch-{epoch}" / "heldout_metrics.json"
        if not path.is_file():
            continue
        score = float(_read_json(path)["f1_macro"])
        if score > best:
            best = score
            kept = epoch
    return kept


def _score_if_needed(root: Path, epoch: int, reviews_path: Path, ids_path: Path) -> None:
    if _epoch_scored(root, epoch):
        _log(root, f"skip score epoch {epoch}; metrics already saved")
        return
    score_split(root, epoch, reviews_path, ids_path, "heldout")
    score_split(root, epoch, reviews_path, ids_path, "full")
    _mark_done(root / f"epoch-{epoch}")


def _train_through(root: Path, reviews_path: Path, ids_path: Path, first_epoch: int) -> int:
    """Train and score ``first_epoch`` through epoch 7. Earlier epochs stay on disk."""
    for epoch in range(1, first_epoch):
        if _weights_ready(root, epoch):
            _score_if_needed(root, epoch, reviews_path, ids_path)
    for epoch in range(first_epoch, LAST_EPOCH + 1):
        if epoch > 1 and not _weights_ready(root, epoch - 1):
            _log(root, f"stopped before epoch {epoch}; epoch {epoch - 1} has no saved weights")
            break
        if _weights_ready(root, epoch):
            _log(root, f"epoch {epoch} weights already saved; continuing at the next missing piece")
        else:
            _log(root, f"training epoch {epoch} from epoch {epoch - 1}")
            _launch_train(root, epoch)
        _score_if_needed(root, epoch, reviews_path, ids_path)
    kept = _kept_epoch(root, LAST_EPOCH)
    _write_json(
        root / "kept.json",
        {"kept_epoch": kept, "directory": f"epoch-{kept}", "reason": "highest held-out macro F1"},
    )
    _status(root, "run", "done", f"kept epoch {kept}")
    return kept


def run(root: Path) -> None:
    anchor = root if root.exists() else Path(".")
    free = shutil.disk_usage(anchor).free
    if free < 30 * 1024**3:
        raise SystemExit(f"need at least 30 GB free on {root}, found {free / 1024**3:.1f} GB")
    reviews_path = root / "preprocess-emoji" / "reviews.parquet"
    ids_path = root / "sample" / "train_ids.parquet"
    tokenized = root / "sample" / "tokenized-train"
    tokenized_ready = (tokenized / "dataset_info.json").is_file() or (tokenized / "state.json").is_file()
    latest = _latest_ready_epoch(root)
    if latest >= 1 and reviews_path.is_file() and ids_path.is_file() and tokenized_ready:
        first = latest + 1
        if first > LAST_EPOCH and _epoch_scored(root, LAST_EPOCH):
            _status(root, "run", "done", f"epochs 1 through {LAST_EPOCH} already scored")
            return
        _status(
            root,
            "run",
            "resuming",
            f"last saved epoch is {latest}; continue through epoch {LAST_EPOCH}",
        )
        _train_through(root, reviews_path, ids_path, min(first, LAST_EPOCH))
        return
    _status(root, "run", "starting", "download, preprocess, sample, train, score")
    csv_path = download_stage(root)
    reviews_path = preprocess_stage(root, csv_path)
    ids_path = sample_stage(root, reviews_path)
    weights = model_stage(root)
    tokenize_stage(root, reviews_path, ids_path, weights)
    _launch_train(root, 1)
    heldout = score_split(root, 1, reviews_path, ids_path, "heldout")
    score_split(root, 1, reviews_path, ids_path, "full")
    _mark_done(root / "epoch-1")
    if float(heldout["f1_macro"]) > EPOCH1_HELD_OUT_BAR:
        _log(root, f"epoch 1 held-out macro F1 {heldout['f1_macro']:.4f} is above {EPOCH1_HELD_OUT_BAR}; training epoch 2")
        _launch_train(root, 2)
        score_split(root, 2, reviews_path, ids_path, "heldout")
        score_split(root, 2, reviews_path, ids_path, "full")
        _mark_done(root / "epoch-2")
    else:
        _log(root, f"epoch 2 skipped; held-out macro F1 {heldout['f1_macro']:.4f} is not above {EPOCH1_HELD_OUT_BAR}")
    _train_through(root, reviews_path, ids_path, 3)


def print_status(root: Path) -> None:
    status_path = root / "status.json"
    if not status_path.is_file():
        print(f"no status yet at {status_path}")
        return
    print(status_path.read_text(encoding="utf-8"))
    log_path = root / "run.log"
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        print("--- run.log (last 15) ---")
        print("\n".join(lines[-15:]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.experiments.tookabert_v2")
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

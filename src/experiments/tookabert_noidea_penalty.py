"""Train one more epoch from the epoch 2 TookaBERT weights.

Epoch 2 has the highest held-out macro F1 of the scored server epochs.
This run loads those weights and keeps the same 1,051,627 training
reviews. The loss is the existing class-weighted cross-entropy plus a
penalty on true ``no_idea`` rows: the probability the model places on
``recommended`` or ``not_recommended``.

A finished epoch is left in place. The next start trains the missing
epochs through epoch 3, each from the previous epoch in this directory.

The prepared reviews, training ids, and tokenized sample stay in
``experiments/tookabert-preproc-v2-2026-11-12``. This file writes a new
experiment directory and does not change that run.

    python -m src.experiments.tookabert_noidea_penalty run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENT = "tookabert-noidea-penalty-2026-11-13"
SOURCE_EXPERIMENT = "tookabert-preproc-v2-2026-11-12"
START_EPOCH = 2
LAST_EPOCH = 3
NO_IDEA_INDEX = 2
NO_IDEA_WRONG_PENALTY = 1.0


def no_idea_wrong_mass(probabilities: list[list[float]], labels: list[int]) -> float:
    """Mean probability on the two non-no_idea classes, over true no_idea rows.

    Rows with another true label contribute nothing. An empty selection
    returns 0. ``probabilities[i][0]`` is recommended and ``[1]`` is
    not_recommended.
    """
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    chosen = [row for row, label in zip(probabilities, labels) if label == NO_IDEA_INDEX]
    if not chosen:
        return 0.0
    total = sum(row[0] + row[1] for row in chosen)
    return total / len(chosen)


def source_root() -> Path:
    return Path("experiments") / SOURCE_EXPERIMENT


def default_root() -> Path:
    return Path("experiments") / EXPERIMENT


def _require_source(source: Path) -> tuple[Path, Path, Path, Path]:
    weights = source / f"epoch-{START_EPOCH}"
    reviews = source / "preprocess-emoji" / "reviews.parquet"
    ids = source / "sample" / "train_ids.parquet"
    tokenized = source / "sample" / "tokenized-train"
    tokenized_ready = (tokenized / "dataset_info.json").is_file() or (tokenized / "state.json").is_file()
    absent = [path for path in (weights / "model.safetensors", reviews, ids) if not path.is_file()]
    if not tokenized_ready:
        absent.append(tokenized)
    if absent:
        raise FileNotFoundError("missing epoch 2 inputs: " + ", ".join(str(path) for path in absent))
    return weights, reviews, ids, tokenized


def _previous_weights(root: Path, source: Path, epoch: int) -> Path:
    if epoch == 1:
        return source / f"epoch-{START_EPOCH}"
    return root / f"epoch-{epoch - 1}"


def train_epoch(root: Path, epoch: int, weights: Path, tokenized_dir: Path) -> None:
    """One epoch. The trainer directory is created before either GPU rank lists it."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from datasets import load_from_disk
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from src.experiments.tookabert_v2 import (
        BLOCK_LR,
        SEED,
        _batch_plan,
        _log,
        _parameter_groups,
        _status,
        _unfreeze,
        _write_json,
    )

    stage = f"epoch-{epoch}"
    epoch_dir = root / stage
    if (epoch_dir / "WEIGHTS_READY").is_file():
        _log(root, f"skip train; {stage} weights already saved")
        return
    epoch_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir = epoch_dir / "trainer"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForSequenceClassification.from_pretrained(weights)
    trainable = _unfreeze(model)
    if rank == 0:
        _status(root, stage, "running", f"trainable parameters {trainable}; weights {weights}")

    table = load_from_disk(str(tokenized_dir))
    labels = np.array(table["label"])
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    class_weights = torch.tensor(labels.shape[0] / (3.0 * counts), dtype=torch.float)
    if rank == 0:
        _log(root, f"class weights {[round(float(value), 4) for value in class_weights]}")
        _log(root, f"no_idea wrong-class penalty {NO_IDEA_WRONG_PENALTY}")

    class Encoded(Dataset):
        def __len__(self):
            return table.num_rows

        def __getitem__(self, index):
            row = table[index]
            return {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
                "labels": int(row["label"]),
            }

    gpu_count = int(os.environ.get("WORLD_SIZE", "1"))
    per_device, accumulation = _batch_plan(gpu_count)
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(rank)[0] >= 8
    shared = dict(
        output_dir=str(trainer_dir),
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

    class PenalizedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            label_ids = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weighted = functional.cross_entropy(
                logits,
                label_ids,
                weight=class_weights.to(logits.device),
            )
            probabilities = functional.softmax(logits, dim=-1)
            true_no_idea = label_ids == NO_IDEA_INDEX
            if bool(true_no_idea.any()):
                wrong_mass = probabilities[true_no_idea, 0] + probabilities[true_no_idea, 1]
                penalty = wrong_mass.mean()
            else:
                penalty = logits.new_zeros(())
            loss = weighted + NO_IDEA_WRONG_PENALTY * penalty
            return (loss, outputs) if return_outputs else loss

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if rank != 0 or not logs or "loss" not in logs:
                return
            loss = float(logs["loss"])
            _status(
                root,
                stage,
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
    trainer = PenalizedTrainer(
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
    last = get_last_checkpoint(str(trainer_dir))
    trainer.train(resume_from_checkpoint=last)
    if trainer.is_world_process_zero():
        trainer.save_model(str(epoch_dir))
        tokenizer.save_pretrained(epoch_dir)
        _write_json(
            epoch_dir / "run_config.json",
            {
                "experiment": EXPERIMENT,
                "stage": stage,
                "started_from": str(weights),
                "started_from_experiment": SOURCE_EXPERIMENT if epoch == 1 else EXPERIMENT,
                "no_idea_wrong_penalty": NO_IDEA_WRONG_PENALTY,
                "penalty": "true no_idea: mean probability on recommended or not_recommended",
                "trainable_parameters": trainable,
                "per_device_batch_size": per_device,
                "gradient_accumulation_steps": accumulation,
                "world_size": gpu_count,
                "bf16": use_bf16,
                "class_weights": [float(value) for value in class_weights],
            },
        )
        (epoch_dir / "WEIGHTS_READY").write_text("ready\n", encoding="utf-8")
        _status(root, stage, "weights-saved", str(epoch_dir))


def score_with_matrix(root: Path, epoch: int, reviews_path: Path, ids_path: Path, which: str) -> dict:
    """Score epoch 1 and store the 3 by 3 counts, rows true and columns predicted."""
    import torch

    from src.experiments.text_v2 import confusion_metrics
    from src.experiments.tookabert_v2 import (
        LABELS,
        _add_matrix,
        _log,
        _score_classes,
        _score_shard,
        _status,
        _write_json,
    )

    stage = f"epoch-{epoch}"
    epoch_dir = root / stage
    target = epoch_dir / ("heldout_metrics.json" if which == "heldout" else "metrics.json")
    if target.is_file():
        _log(root, f"skip {which} score for {stage}")
        return {}
    import polars as pl

    from src.experiments.tookabert_v2 import LABEL_CODE, _ensure_hf_config

    _ensure_hf_config(epoch_dir, epoch_dir)
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
    _status(root, stage, "scoring", f"{which} rows {len(texts)} on {shards} device(s)")
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
                stage,
                index,
                shards,
                model_cls,
                tokenizer_cls,
            )
        except BaseException as exc:
            errors.append(exc)

    import threading

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
    metrics["started_from"] = START_EPOCH if epoch == 1 else epoch - 1
    metrics["confusion_matrix"] = matrix
    metrics["confusion_matrix_labels"] = list(LABELS)
    _write_json(target, metrics)
    _status(
        root,
        stage,
        "scored",
        f"{which} accuracy {metrics['accuracy']:.4f} macro F1 {metrics['f1_macro']:.4f} micro F1 {metrics['f1_micro']:.4f}",
    )
    return metrics


def _launch_train(root: Path, source: Path, epoch: int) -> None:
    import torch

    from src.experiments.tookabert_v2 import _log

    if (root / f"epoch-{epoch}" / "WEIGHTS_READY").is_file():
        _log(root, f"epoch-{epoch} weights present")
        return
    command = [
        sys.executable,
        "-m",
        "src.experiments.tookabert_noidea_penalty",
        "train",
        "--epoch",
        str(epoch),
        "--root",
        str(root),
        "--source",
        str(source),
    ]
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpu_count),
            "--master_port",
            "29521",
            "-m",
            "src.experiments.tookabert_noidea_penalty",
            "train",
            "--epoch",
            str(epoch),
            "--root",
            str(root),
            "--source",
            str(source),
        ]
    _log(root, " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run(root: Path, source: Path) -> None:
    from src.experiments.tookabert_v2 import _log, _mark_done, _status

    _require_source(source)
    root.mkdir(parents=True, exist_ok=True)
    _status(root, "run", "running", f"continue through epoch {LAST_EPOCH}")
    reviews = source / "preprocess-emoji" / "reviews.parquet"
    ids = source / "sample" / "train_ids.parquet"
    tokenized = source / "sample" / "tokenized-train"
    for epoch in range(1, LAST_EPOCH + 1):
        if not (root / f"epoch-{epoch}" / "WEIGHTS_READY").is_file():
            previous = _previous_weights(root, source, epoch)
            if epoch > 1 and not (previous / "model.safetensors").is_file():
                raise FileNotFoundError(f"missing weights for epoch {epoch}: {previous}")
            _log(root, f"training epoch {epoch} from {previous}")
            _launch_train(root, source, epoch)
        heldout = root / f"epoch-{epoch}" / "heldout_metrics.json"
        full = root / f"epoch-{epoch}" / "metrics.json"
        if heldout.is_file() and full.is_file():
            _log(root, f"skip score epoch {epoch}; metrics already saved")
        else:
            if not heldout.is_file():
                score_with_matrix(root, epoch, reviews, ids, "heldout")
            if not full.is_file():
                score_with_matrix(root, epoch, reviews, ids, "full")
        _mark_done(root / f"epoch-{epoch}")
    _log(root, f"run done: epochs 1 through {LAST_EPOCH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.experiments.tookabert_noidea_penalty")
    parser.add_argument("command", choices=("run", "train"))
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root or default_root()
    source = args.source or source_root()
    if args.command == "train":
        _require_source(source)
        weights = _previous_weights(root, source, args.epoch)
        tokenized = source / "sample" / "tokenized-train"
        train_epoch(root, args.epoch, weights, tokenized)
        return
    run(root, source)


if __name__ == "__main__":
    main()

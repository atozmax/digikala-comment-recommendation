"""Save an untrained rating-band classifier from the epoch-2 encoder.

The encoder stays the TookaBERT preprocessing-v2 epoch 2 weights. The
recommendation head is replaced by a new 3-class layer for the ordered
bands ``low``, ``medium``, and ``high``. The new layer is drawn from
Normal(0, 0.02) with seed 2026 and a zero bias. It has not been trained
on ``rating_band``.

Run from the repository root:

    python -m src.experiments.tookabert_rating_band_baseline
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

SOURCE = Path("experiments/epochs/epoch-2")
DEST = Path("experiments/baselines/tookabert-epoch2-rating-band")
SEED = 2026
INITIALIZER_STD = 0.02
HIDDEN_SIZE = 768
LABELS = ("low", "medium", "high")


def build(source: Path = SOURCE, dest: Path = DEST) -> Path:
    """Write the rating-band checkpoint and return its directory."""
    from safetensors.numpy import load_file, save_file

    weights = source / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"epoch-2 weights not found at {weights}")

    tensors = load_file(weights)
    if tuple(tensors["classifier.weight"].shape) != (3, HIDDEN_SIZE):
        raise ValueError(f"expected a 3-class head, found {tensors['classifier.weight'].shape}")

    rng = np.random.default_rng(SEED)
    tensors["classifier.weight"] = rng.normal(
        0.0, INITIALIZER_STD, size=(len(LABELS), HIDDEN_SIZE)
    ).astype(np.float32)
    tensors["classifier.bias"] = np.zeros((len(LABELS),), dtype=np.float32)

    dest.mkdir(parents=True, exist_ok=True)
    save_file(tensors, dest / "model.safetensors")

    config = json.loads((source / "config.json").read_text())
    config["num_labels"] = len(LABELS)
    config["problem_type"] = "single_label_classification"
    config["id2label"] = {str(index): name for index, name in enumerate(LABELS)}
    config["label2id"] = {name: index for index, name in enumerate(LABELS)}
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(source / name, dest / name)

    note = {
        "source": str(source),
        "target": "rating_band",
        "labels": list(LABELS),
        "label_ids": {name: index for index, name in enumerate(LABELS)},
        "bands": {
            "low": "0 <= rate < 2.5",
            "medium": "2.5 <= rate < 4",
            "high": "4 <= rate <= 5",
        },
        "problem_type": "single_label_classification",
        "num_labels": len(LABELS),
        "head": "classifier.weight [3, 768], classifier.bias [3], recommendation head discarded",
        "init": f"normal(0, {INITIALIZER_STD}), bias 0, numpy seed {SEED}",
        "trained": False,
    }
    (dest / "baseline.json").write_text(json.dumps(note, indent=2) + "\n")
    return dest


def main() -> None:
    dest = build()
    print(f"saved untrained rating-band classifier at {dest}")


if __name__ == "__main__":
    main()

"""Save an untrained regression baseline from the epoch-2 classifier.

The encoder stays the TookaBERT preprocessing-v2 epoch 2 weights. The
3-class ``classifier`` layer is replaced by a new linear layer with one
output. That layer is drawn from Normal(0, 0.02) with seed 2026 and a
zero bias. It has not been trained on ``rate``.

Run from the repository root:

    python -m src.experiments.tookabert_regression_baseline
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

SOURCE = Path("experiments/epochs/epoch-2")
DEST = Path("experiments/baselines/tookabert-epoch2-regression")
SEED = 2026
INITIALIZER_STD = 0.02
HIDDEN_SIZE = 768


def build(source: Path = SOURCE, dest: Path = DEST) -> Path:
    """Write the regression checkpoint and return its directory."""
    from safetensors.numpy import load_file, save_file

    weights = source / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"epoch-2 weights not found at {weights}")

    tensors = load_file(weights)
    if tuple(tensors["classifier.weight"].shape) != (3, HIDDEN_SIZE):
        raise ValueError(f"expected a 3-class head, found {tensors['classifier.weight'].shape}")

    rng = np.random.default_rng(SEED)
    tensors["classifier.weight"] = rng.normal(0.0, INITIALIZER_STD, size=(1, HIDDEN_SIZE)).astype(np.float32)
    tensors["classifier.bias"] = np.zeros((1,), dtype=np.float32)

    dest.mkdir(parents=True, exist_ok=True)
    save_file(tensors, dest / "model.safetensors")

    config = json.loads((source / "config.json").read_text())
    config["num_labels"] = 1
    config["problem_type"] = "regression"
    config["id2label"] = {"0": "rate"}
    config["label2id"] = {"rate": 0}
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(source / name, dest / name)

    note = {
        "source": str(source),
        "target": "rate",
        "problem_type": "regression",
        "num_labels": 1,
        "head": "classifier.weight [1, 768], classifier.bias [1]",
        "init": f"normal(0, {INITIALIZER_STD}), bias 0, numpy seed {SEED}",
        "trained": False,
    }
    (dest / "baseline.json").write_text(json.dumps(note, indent=2) + "\n")
    return dest


def main() -> None:
    dest = build()
    print(f"saved untrained regression baseline at {dest}")


if __name__ == "__main__":
    main()

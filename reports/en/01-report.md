**(1405-08-20)** · Wednesday, 11 November 2026

# Where the next training run starts

The next training run starts from the saved weights on Google Drive. It does not start from the original Hugging Face checkpoints, and it does not resume optimizer step 115,027. Those Drive folders contain `model.safetensors` plus the tokenizer. They do not contain optimizer state, so the next epoch is a new epoch that begins at these weights.

The small JSON files from the same folders are already under `artifacts/colab/`. The weight files stayed on Drive. Section 12 of `reports/en/00-report.md` records the scores. This note records only the files the next run loads.

## Starting weights

Each finished model has three `model.safetensors` files: `runs/...--head`, `runs/...--last_block`, and `best/...`. The `best/` copy is the last-block run, and that is the folder training loads.

| Model | Drive folder | Starting directory |
| --- | --- | --- |
| ParsBERT, `HooshvareLab/bert-fa-base-uncased` | https://drive.google.com/drive/folders/1GKqsgeXT8uFFLYddJDnA9509bIiDHosd?usp=sharing | `best/HooshvareLab--bert-fa-base-uncased` |
| XLM-RoBERTa, `FacebookAI/xlm-roberta-base` | https://drive.google.com/drive/folders/1AEtn221vAuEW3yAzgQ6-JSeJd0-keHpm?usp=sharing | `best/FacebookAI--xlm-roberta-base` |
| TookaBERT, `PartAI/TookaBERT-Base` | https://drive.google.com/drive/folders/1mcsqBfSqyEtD8iu5G0l8poeYR7oaitLl?usp=sharing | `best/PartAI--TookaBERT-Base` |

Each of those directories holds `config.json`, `metrics.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`. The matching `runs/<name>--last_block` directory is the same run before it was copied into `best/`. The `runs/<name>--head` directory is the earlier frozen-encoder run. A continuation loads `best/`, not `head/`.

`notebooks/colab/04-shiraz-finetune.ipynb` has no weight file. `lifeweb-ai/shiraz` is not a starting point until a head epoch is saved.

## What is already trained

The loaded weights already include a trained classifier and the top encoder block, `layers[-1]`. The blocks below that are still the original pretrained encoder. The next run keeps those saved weights and unfreezes the top four blocks, `layers[-4:]`, with learning rate `2e-5` on the blocks and `1e-4` on the classifier. The bottom eight blocks stay frozen.

Validation of the weights being loaded:

| Model | Accuracy | Macro F1 | recommended F1 | not_recommended F1 | no_idea F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ParsBERT | 0.8004 | 0.6687 | 0.8947 | 0.6902 | 0.4211 |
| XLM-RoBERTa | 0.7989 | 0.6630 | 0.8943 | 0.6824 | 0.4123 |
| TookaBERT | 0.8053 | 0.6722 | 0.8983 | 0.6969 | 0.4214 |

TookaBERT is the first continuation. It has the highest validation accuracy and the highest validation macro F1 of the three saved models. ParsBERT and XLM-RoBERTa stay available at the paths above for the same kind of run. TookaBERT's `best/` folder still has no last-block `test_metrics.json`; its test file is the earlier head score.

## The 10% runs do not replace these weights

On 21 Aban 1405 (12 November 2026) the top-four notebooks in `notebooks/colab3/` finished on a random 10% slice. Their Drive folders are separate from the table above. Section 13 of `reports/en/00-report.md` records the scores. A later full-data run still loads the `best/` directories in the table above.

| Notebook | Drive folder | Kept directory |
| --- | --- | --- |
| `notebooks/colab3/01-tookabert-top4.ipynb` | https://drive.google.com/drive/folders/1eSvV-_vO_Pgygj34poKD8ZmJfiO5w6tI?usp=sharing | `best/PartAI--TookaBERT-Base--top4_sample10` |
| `notebooks/colab3/02-parsbert-top4.ipynb` | https://drive.google.com/drive/folders/1uewUGcYTCkjs0UC4JH9IB2GOZvzIjdd2?usp=sharing | `best/HooshvareLab--bert-fa-base-uncased--top4_sample10` |

The same TookaBERT Drive folder now also holds the published-model score from `notebooks/colab3/05-hooshvare-digikala-sentiment.ipynb`, at `runs/HooshvareLab--bert-fa-base-uncased-sentiment-digikala/test_metrics.json`. That file is a score of every labeled review. It is not a training checkpoint. Section 14 of `reports/en/00-report.md` records it. The folder link is https://drive.google.com/drive/folders/1mcsqBfSqyEtD8iu5G0l8poeYR7oaitLl?usp=sharing.

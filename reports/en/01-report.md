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

## TookaBERT preprocessing v2, epoch 1

On 21 Aban 1405 (12 November 2026) `src/experiments/tookabert_v2.py` finished epoch 1 on the GPU server. The run is `experiments/tookabert-preproc-v2-2026-11-12`. It trained the top four blocks of `PartAI/TookaBERT-Base` on a stratified 20% of the labeled rows, seed 2026. The held-out 80% is the validation set that decides a second epoch. The score files have accuracy, macro F1, micro F1, and per-class precision, recall, and F1. They do not record a loss.

Every epoch trains on the same 1,051,627 reviews. The counts are from `sample/config.json`.

| Class | Labeled rows | Training rows |
| --- | ---: | ---: |
| recommended | 4,160,050 | 832,010 |
| not_recommended | 504,452 | 100,890 |
| no_idea | 593,633 | 118,727 |
| Total | 5,258,135 | 1,051,627 |

The held-out 80% is 4,206,508 rows.

Held-out 80%, 4,206,508 rows, from `epoch-1/heldout_metrics.json`:

| | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: |
| Held-out 80% | 0.8305 | 0.6965 | 0.8305 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| recommended | 0.9599 | 0.8748 | 0.9154 | 3,328,040 |
| not_recommended | 0.6691 | 0.7843 | 0.7221 | 403,562 |
| no_idea | 0.3793 | 0.5595 | 0.4521 | 474,906 |

The full labeled set, 5,258,135 rows, is in `epoch-1/metrics.json`. Accuracy 0.8314, macro F1 0.6983, micro F1 0.8314. Per-class F1 is 0.9158 for recommended (support 4,160,050), 0.7243 for not_recommended (support 504,452), and 0.4549 for no_idea (support 593,633).

Held-out macro F1 0.6965 is above the saved full-data validation 0.6722, so epoch 2 trained and was scored. The run finished at 21:47 UTC and wrote `kept.json` with epoch 2, because its held-out macro F1 is the higher of the two. The score files still have no loss. The last logged training loss, averaged over steps 32,801–32,850, is 1.0810 for epoch 1 and 1.0954 for epoch 2. The 20% training rows were not scored as their own classification set.

## TookaBERT preprocessing v2, epoch 2

Epoch 2 used the same 20% training rows and the same held-out 80%. Files: `epoch-2/heldout_metrics.json` and `epoch-2/metrics.json`.

| Split | Rows | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Epoch 1 held-out 80% | 4,206,508 | 0.8305 | 0.6965 | 0.8305 |
| Epoch 2 held-out 80% | 4,206,508 | 0.8312 | 0.6972 | 0.8312 |
| Epoch 1 full labeled set | 5,258,135 | 0.8314 | 0.6983 | 0.8314 |
| Epoch 2 full labeled set | 5,258,135 | 0.8330 | 0.7010 | 0.8330 |

Held-out per-class F1:

| Class | Epoch 1 | Epoch 2 | Support |
| --- | ---: | ---: | ---: |
| recommended | 0.9154 | 0.9160 | 3,328,040 |
| not_recommended | 0.7221 | 0.7217 | 403,562 |
| no_idea | 0.4521 | 0.4538 | 474,906 |

Full-set per-class F1: recommended 0.9158 then 0.9167 (support 4,160,050), not_recommended 0.7243 then 0.7269 (support 504,452), no_idea 0.4549 then 0.4593 (support 593,633). Epoch 2 held-out precision and recall are 0.9600 / 0.8759 for recommended, 0.6732 / 0.7779 for not_recommended, and 0.3800 / 0.5632 for no_idea. The held-out macro F1 gain from epoch 1 to epoch 2 is 0.0006. `no_idea` remains the weak class.

## TookaBERT preprocessing v2, epoch 3

Epoch 3 resumed from the saved epoch 2 weights on the same 20% training rows. Weights were saved at 23:20 UTC on 12 November 2026. The held-out score was written at 23:37 UTC and the full-set score at 23:59 UTC. Files: `epoch-3/heldout_metrics.json` and `epoch-3/metrics.json`. Epoch 4 training had started when these numbers were copied.

| Split | Rows | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Epoch 3 held-out 80% | 4,206,508 | 0.8289 | 0.6927 | 0.8289 |
| Epoch 3 full labeled set | 5,258,135 | 0.8315 | 0.6981 | 0.8315 |

Held-out per-class scores, from `epoch-3/heldout_metrics.json`:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| recommended | 0.9594 | 0.8750 | 0.9153 | 3,328,040 |
| not_recommended | 0.6690 | 0.7640 | 0.7133 | 403,562 |
| no_idea | 0.3750 | 0.5609 | 0.4495 | 474,906 |

Full-set per-class F1: recommended 0.9162 (support 4,160,050), not_recommended 0.7209 (support 504,452), no_idea 0.4572 (support 593,633). Full-set precision and recall are 0.9602 / 0.8759 for recommended, 0.6761 / 0.7721 for not_recommended, and 0.3815 / 0.5702 for no_idea. Held-out macro F1 is 0.0045 below epoch 2. The last logged training loss, averaged over steps 32,801–32,850, is 1.1615. The score files have no loss. Among epochs 1, 2, and 3, epoch 2 still has the highest held-out macro F1.

## TookaBERT preprocessing v2, epoch 4

Epoch 4 resumed from the saved epoch 3 weights on the same 20% training rows. The held-out score was written at 01:14 UTC on 13 November 2026 (22 Aban 1405) and the full-set score at 01:35 UTC. Files: `epoch-4/heldout_metrics.json` and `epoch-4/metrics.json`. Epoch 5 then stopped before the first training step, so it has no score file.

| Split | Rows | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Epoch 4 held-out 80% | 4,206,508 | 0.8270 | 0.6881 | 0.8270 |
| Epoch 4 full labeled set | 5,258,135 | 0.8301 | 0.6944 | 0.8301 |

Held-out per-class scores, from `epoch-4/heldout_metrics.json`:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| recommended | 0.9583 | 0.8749 | 0.9147 | 3,328,040 |
| not_recommended | 0.6622 | 0.7531 | 0.7048 | 403,562 |
| no_idea | 0.3713 | 0.5544 | 0.4448 | 474,906 |

Full-set per-class F1: recommended 0.9158 (support 4,160,050), not_recommended 0.7134 (support 504,452), no_idea 0.4539 (support 593,633). Full-set precision and recall are 0.9593 / 0.8761 for recommended, 0.6705 / 0.7623 for not_recommended, and 0.3791 / 0.5655 for no_idea. Held-out macro F1 is 0.0091 below epoch 2 and 0.0046 below epoch 3. The last logged training loss, averaged over steps 32,801–32,850, is 1.2423. The score files have no loss. Among epochs 1 through 4, epoch 2 still has the highest held-out macro F1.

## TookaBERT preprocessing v2, epoch 5

Epoch 5 resumed from the saved epoch 4 weights on the same 20% training rows. The held-out score was written at 14:24 UTC on 13 November 2026 (22 Aban 1405) and the full-set score at 14:54 UTC. Files: `epoch-5/heldout_metrics.json` and `epoch-5/metrics.json`. Epoch 6 training started after that score and is not scored.

| Split | Rows | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Epoch 5 held-out 80% | 4,206,508 | 0.8260 | 0.6840 | 0.8260 |
| Epoch 5 full labeled set | 5,258,135 | 0.8293 | 0.6909 | 0.8293 |

Held-out per-class scores, from `epoch-5/heldout_metrics.json`:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| recommended | 0.9566 | 0.8760 | 0.9145 | 3,328,040 |
| not_recommended | 0.6549 | 0.7448 | 0.6969 | 403,562 |
| no_idea | 0.3696 | 0.5450 | 0.4405 | 474,906 |

Full-set per-class F1: recommended 0.9157 (support 4,160,050), not_recommended 0.7064 (support 504,452), no_idea 0.4506 (support 593,633). Full-set precision and recall are 0.9578 / 0.8772 for recommended, 0.6638 / 0.7548 for not_recommended, and 0.3783 / 0.5573 for no_idea. Held-out macro F1 is 0.0132 below epoch 2 and 0.0041 below epoch 4. The last logged training loss, averaged over steps 32,801–32,850, is 1.3420. The score files have no loss. Among epochs 1 through 5, epoch 2 still has the highest held-out macro F1.

## TookaBERT epoch 2 with a no_idea penalty

On 22 Aban 1405 (13 November 2026) `src/experiments/tookabert_noidea_penalty.py` loads the epoch 2 weights and trains three epochs on the same 1,051,627 reviews. The output directory is `experiments/tookabert-noidea-penalty-2026-11-13`. The earlier experiment directory stays as it was. All three epochs are scored.

Epoch 2 held-out recall for `no_idea` is 0.5632 on 474,906 rows, so 267,484 predictions are correct and 207,422 are misses. Those misses were predicted `recommended` or `not_recommended`. `heldout_metrics.json` does not store which of the two.

The custom term is `no_idea_wrong_mass` in that file. For one batch it keeps only the rows whose true label is `no_idea`, reads the softmax probabilities, and averages `p(recommended) + p(not_recommended)`. That quantity is the probability left after `p(no_idea)`, so it grows when a true `no_idea` review is given to either of the other two classes. Other true labels add nothing to this average. If the batch has no `no_idea` row, the term is 0. The trainer adds it once:

class-weighted cross-entropy + 1.0 × that mean.

The class weights are unchanged from the earlier run. The new score files include `confusion_matrix`, with the true label on the rows and the prediction on the columns.

Epoch 1 finished at 13:42 UTC on 13 November 2026. Files: `experiments/tookabert-noidea-penalty-2026-11-13/epoch-1/heldout_metrics.json` and `metrics.json`.

| Split | Rows | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Held-out 80% | 4,206,508 | 0.7946 | 0.6776 | 0.7946 |
| Full labeled set | 5,258,135 | 0.7972 | 0.6828 | 0.7972 |

Held-out per-class scores:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| recommended | 0.9676 | 0.8247 | 0.8904 | 3,328,040 |
| not_recommended | 0.7250 | 0.6748 | 0.6990 | 403,562 |
| no_idea | 0.3276 | 0.6859 | 0.4434 | 474,906 |

Full-set per-class F1: recommended 0.8913 (support 4,160,050), not_recommended 0.7074 (support 504,452), no_idea 0.4496 (support 593,633). Full-set precision and recall are 0.9684 / 0.8256 for recommended, 0.7335 / 0.6832 for not_recommended, and 0.3323 / 0.6951 for no_idea. The last logged training loss, averaged over steps 32,801–32,850, is 1.6898.

Held-out confusion counts, from `confusion_matrix`:

| True label | Predicted recommended | Predicted not_recommended | Predicted no_idea |
| --- | ---: | ---: | ---: |
| recommended | 2,744,577 | 34,476 | 548,987 |
| not_recommended | 11,548 | 272,333 | 119,681 |
| no_idea | 80,319 | 68,836 | 325,751 |

The two `no_idea` mistakes are 80,319 predicted `recommended` and 68,836 predicted `not_recommended`, 149,155 together. On the source epoch 2 those two mistakes added up to 207,422, with the split unknown. True `recommended` predicted `no_idea` is 548,987. Held-out macro F1 0.6776 is 0.0196 below the source epoch 2.

## TookaBERT no_idea penalty, epoch 2

Epoch 2 continued from this run's epoch 1 weights on the same 1,051,627 reviews. The weights were saved at 15:27 UTC on 13 November 2026, the held-out score at 15:45 UTC, and the full-set score at 16:32 UTC. Files: `experiments/tookabert-noidea-penalty-2026-11-13/epoch-2/heldout_metrics.json` and `metrics.json`.

| Split | Rows | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Held-out 80% | 4,206,508 | 0.7961 | 0.6749 | 0.7961 |
| Full labeled set | 5,258,135 | 0.7991 | 0.6810 | 0.7991 |

Held-out per-class scores, from `epoch-2/heldout_metrics.json`:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| recommended | 0.9660 | 0.8288 | 0.8922 | 3,328,040 |
| not_recommended | 0.7057 | 0.6815 | 0.6934 | 403,562 |
| no_idea | 0.3279 | 0.6638 | 0.4390 | 474,906 |

Full-set per-class F1: recommended 0.8932 (support 4,160,050), not_recommended 0.7031 (support 504,452), no_idea 0.4467 (support 593,633). Full-set precision and recall are 0.9670 / 0.8299 for recommended, 0.7155 / 0.6912 for not_recommended, and 0.3337 / 0.6750 for no_idea. The last logged training loss, averaged over steps 32,801–32,850, is 1.7856.

Held-out confusion counts, from `confusion_matrix`:

| True label | Predicted recommended | Predicted not_recommended | Predicted no_idea |
| --- | ---: | ---: | ---: |
| recommended | 2,758,396 | 39,819 | 529,825 |
| not_recommended | 12,269 | 275,034 | 116,259 |
| no_idea | 84,791 | 74,857 | 315,258 |

The two `no_idea` mistakes are 84,791 predicted `recommended` and 74,857 predicted `not_recommended`, 159,648 together. That is 10,493 more than this run's epoch 1 and 47,774 fewer than the 207,422 misses on the source epoch 2. True `recommended` predicted `no_idea` is 529,825, down from 548,987 on this run's epoch 1. Held-out macro F1 0.6749 is 0.0027 below this run's epoch 1 and 0.0223 below the source epoch 2.

## TookaBERT no_idea penalty, epoch 3

Epoch 3 continued from this run's epoch 2 weights on the same 1,051,627 reviews. The weights were saved at 17:31 UTC on 13 November 2026, the held-out score at 17:50 UTC, and the full-set score at 18:14 UTC. The run finished at 18:14 UTC. Files: `experiments/tookabert-noidea-penalty-2026-11-13/epoch-3/heldout_metrics.json` and `metrics.json`.

| Split | Rows | Accuracy | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Held-out 80% | 4,206,508 | 0.7986 | 0.6733 | 0.7986 |
| Full labeled set | 5,258,135 | 0.8019 | 0.6801 | 0.8019 |

Held-out per-class scores, from `epoch-3/heldout_metrics.json`:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| recommended | 0.9641 | 0.8342 | 0.8944 | 3,328,040 |
| not_recommended | 0.6890 | 0.6925 | 0.6907 | 403,562 |
| no_idea | 0.3295 | 0.6392 | 0.4349 | 474,906 |

Full-set per-class F1: recommended 0.8956 (support 4,160,050), not_recommended 0.7010 (support 504,452), no_idea 0.4436 (support 593,633). Full-set precision and recall are 0.9652 / 0.8353 for recommended, 0.6991 / 0.7030 for not_recommended, and 0.3362 / 0.6517 for no_idea. The last logged training loss, averaged over steps 32,801–32,850, is 1.9460.

Held-out confusion counts, from `confusion_matrix`:

| True label | Predicted recommended | Predicted not_recommended | Predicted no_idea |
| --- | ---: | ---: | ---: |
| recommended | 2,776,250 | 44,942 | 506,848 |
| not_recommended | 13,314 | 279,459 | 110,789 |
| no_idea | 90,132 | 81,218 | 303,556 |

The two `no_idea` mistakes are 90,132 predicted `recommended` and 81,218 predicted `not_recommended`, 171,350 together. That is 11,702 more than this run's epoch 2 and 36,072 fewer than the 207,422 misses on the source epoch 2. True `recommended` predicted `no_idea` is 506,848, down from 529,825 on this run's epoch 2. Held-out macro F1 0.6733 is 0.0016 below this run's epoch 2, 0.0043 below this run's epoch 1, and 0.0239 below the source epoch 2. Among the three epochs of this run, epoch 1 has the highest held-out macro F1.

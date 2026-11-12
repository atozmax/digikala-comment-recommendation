# Plan: next sentiment experiments

12 November 2026. This file is the plan only. It does not add a notebook.

The task stays the three-way recommendation label on `body`: `recommended`, `not_recommended`, `no_idea`. `rate` stays out of the features. Scores already in `reports/en/00-report.md` sections 12–14 are the evidence below. A 10% score is not a replacement for a full-set score.

## 1. What the past runs showed

### Models

| Run | What it was | Result |
| --- | --- | --- |
| ParsBERT, `HooshvareLab/bert-fa-base-uncased`, `notebooks/colab/01-parsbert-finetune.ipynb` | Head, then last encoder block, full data | Last-block test accuracy 0.8000, macro F1 0.6682. Finished. |
| XLM-RoBERTa, `FacebookAI/xlm-roberta-base`, `notebooks/colab/02-xlmr-finetune.ipynb` | Same two stages | Last-block test accuracy 0.7993, macro F1 0.6637. Finished, slowest, lowest of the three. |
| TookaBERT, `PartAI/TookaBERT-Base`, `notebooks/colab/03-tookabert-finetune.ipynb` | Same two stages | Last-block validation accuracy 0.8053, macro F1 0.6722. Best full-data validation. No last-block test file; `best/test_metrics.json` is still the head test (macro F1 0.5707). |
| Shiraz, `lifeweb-ai/shiraz`, `notebooks/colab/04-shiraz-finetune.ipynb` | MobileBERT head | No epoch finished. `TypeError` on `tokenizer`, then `NameError` for `DRIVE_DIR`. No weights. |
| TookaBERT top 4, `notebooks/colab3/01-tookabert-top4.ipynb` | Blocks 8–11 from the saved last-block weights, random 10% | Kept epoch 2. Slice validation macro F1 0.6825, slice test 0.6867. Epochs after that were worse. |
| ParsBERT top 4, `notebooks/colab3/02-parsbert-top4.ipynb` | Same recipe | Kept epoch 1. Slice validation macro F1 0.6778, slice test 0.6816. Later epochs were worse. |
| Published sentiment ParsBERT, `HooshvareLab/bert-fa-base-uncased-sentiment-digikala`, `notebooks/colab3/05-hooshvare-digikala-sentiment.ipynb` | No training. Score every labeled row | Accuracy 0.7403, macro F1 0.5918 on 5,258,342 rows. Card F1 0.8172 is the 62,321-comment set, not this file. |

`notebooks/colab2/` holds the full-data continuations (TookaBERT top 4, ParsBERT top 4, TookaBERT all blocks, TookaBERT with a doubled `no_idea` loss). Those notebooks have no finished scores in the reports. The all-blocks and `no_idea`-weight runs were written for separate accounts and are not evidence yet.

Per-class F1 is the same story in every finished run. `recommended` sits near 0.90 after fine-tuning (0.86 on the published 62k model). `not_recommended` is near 0.69–0.71 on the fine-tunes and 0.61 on the published model. `no_idea` stays near 0.42 on the fine-tunes and 0.31 on the published model. More epochs on the 10% slice did not fix `no_idea`. About 79% of labeled rows are `recommended`, so accuracy near 0.80 is a small step past the majority class.

### Tokenizers and embeddings

Each fine-tune used that checkpoint's own tokenizer and its encoder embeddings. There was no separate embedding model.

- ParsBERT and TookaBERT are WordPiece BERT models, hidden size 768, 12 layers. The notebooks keep Persian digits for them. Trainable head is 2,307 parameters. Last block is about 7.09 million. Top four blocks on the 10% runs were 28,353,795 parameters.
- XLM-RoBERTa uses the SentencePiece tokenizer, vocabulary 250,002. The notebook folds digits to ASCII before that tokenizer. It was slower and scored below both Persian BERT models.
- The published sentiment checkpoint is the same ParsBERT vocabulary, but its label ids are `no_idea=0`, `not_recommended=1`, `recommended=2`. This project uses `recommended=0`, `not_recommended=1`, `no_idea=2`. Those integers must not be mixed.
- Shiraz is MobileBERT. It never produced a checkpoint, so its tokenizer is untested here.

### Preprocessing the past runs actually trained on

`src/preprocess.py` writes `data/processed/digikala-comments-preprocessed.csv`, which is the file the Colab notebooks download. That pass already does NFKC, maps Arabic letters onto Persian letters, folds ASCII and Arabic-Indic digits onto Persian digits, strips diacritics and tatweel, keeps one interior half-space, drops repeated or edge half-spaces, replaces URLs and emails, caps a character run at two, and then Hazm-stems every token. Emoji and Latin letters stay. Section 10 of the report measured why: Latin letters are in 0.65% of reviews and are mostly brands and model codes; 172 reviews are emoji only; Arabic kaf/ye is 0.82%.

What worked is the character fold that makes `ك` and `ک` share a subword, plus training the top of a Persian BERT. What did not move the weak class is extra epochs, the 62k sentiment checkpoint, and XLM-RoBERTa. Hazm stemming was applied before those runs. The new runs should stop stemming. ParsBERT and TookaBERT were pretrained on surface words, and a stem is a different string than the wordpiece vocabulary expects.

`notebooks/colab2/03-tookabert-all-blocks.ipynb` and `notebooks/colab2/04-tookabert-no-idea-weight.ipynb` are still the right unread ideas: all 12 blocks at `1e-5`, and a doubled `no_idea` loss on top-4. They are not results until a notebook prints scores.

## 2. Preprocessing for the next run

Do these stages in order, and save the text after each stage. The experiment config records which stages were on.

1. Keep the current cleaning: drop empty bodies, collapse whitespace, one row per id. Do not fill a missing `recommendation_status`. Do not drop repeated bodies. 31,409 texts carry more than one recommendation label.
2. NFKC, then the Arabic-to-Persian letter map already in `src/preprocess.py` (`ك` to `ک`, `ي` and `ى` to `ی`, and the alef and heh forms in that map). Strip Arabic diacritics, tatweel, and invisible marks other than ZWNJ.
3. Keep one half-space inside a word. Collapse a doubled half-space. Delete a half-space at an edge or beside a space. Do not delete every half-space. Section 10 says an interior half-space is part of the spelling, and the Persian BERT vocabularies were trained with it.
4. Fold ASCII and Arabic-Indic digits onto Persian digits, matching the ParsBERT and TookaBERT path. Do not also fold them to ASCII in the same run.
5. Numbers: after that fold, delete a token that is only digits. Keep a token that mixes digits and letters (`A54`, `128GB`). Pure numbers are sizes, years, and prices. They are a small share of reviews (Persian digits 3.17%, ASCII digits 1.56%). Model codes are not pure numbers, and those should stay.
6. Emoji: delete emoji characters, then drop a body that is empty after that. Section 10 found 172 emoji-only reviews. Losing those 172 labels is acceptable. Keeping emoji did not help `no_idea`, and the published model was not trained as an emoji classifier. This is a change from the current "keep emoji" rule, and the config must say so.
7. Do not Hazm-stem. Do not remove Persian stopwords. Do not run the Hazm lemmatizer.

### English text

Do not remove English in the main run.

Latin letters are in 0.65% of reviews. Of those, 36,765 also contain Arabic-script letters and 3,307 do not. The report's reading is that this slice is brand names and model codes, not a second language corpus. Deleting it removes `Samsung` and `iPhone` from a Persian sentence and deletes the 3,307 reviews that have no Persian letters left. That is a real label loss for a small, informative slice.

The check is a second experiment, not the default. Same sample, same model, with Latin letters deleted and empty bodies dropped. If held-out macro F1 moves by less than 0.005, English was not worth a separate pipeline. If it drops, the main run was right to keep it.

### One extra step worth adding

Replace a URL or email with a space, and cap a run of the same character at two. `src/preprocess.py` already does both. URLs are not recommendation text. `عاااالی` and `عالی` should share a subword. Neither step deletes a class of reviews the way English removal would.

## 3. Models and tokenizers to run

Run one model first.

**TookaBERT, `PartAI/TookaBERT-Base`, with `AutoTokenizer` from that same checkpoint.** It has the best full-data validation (macro F1 0.6722) and the best 10% slice score. Start from `best/PartAI--TookaBERT-Base` on the original Drive folder, not from the raw Hugging Face weights and not from the published sentiment checkpoint. Unfreeze blocks 8–11 at `2e-5` and the classifier at `1e-4`. The bottom eight blocks stay frozen. That is the recipe that gained on the 10% slice, and it uses 28.4 million trainable parameters instead of a full 12-block backward.

Label ids stay `recommended=0`, `not_recommended=1`, `no_idea=2`. The loaded head already uses that order.

**Second, only if the TookaBERT run finishes: ParsBERT, `HooshvareLab/bert-fa-base-uncased`, same tokenizer, same top-4 recipe, from `best/HooshvareLab--bert-fa-base-uncased`.** It is the next finished model (full-data test macro F1 0.6682). It is the control that the gain is not specific to TookaBERT.

Do not start from `HooshvareLab/bert-fa-base-uncased-sentiment-digikala`. On this file it scores macro F1 0.5918, and `no_idea` F1 is 0.3144, below the fine-tunes. Its label order is also different.

Leave XLM-RoBERTa and Shiraz out of this round. XLM-RoBERTa was slower and worse. Shiraz has no saved weights.

Class-weighted cross-entropy stays. The 10% ParsBERT run was best at epoch 1, and TookaBERT was best at epoch 2 with later epochs worse. Train one epoch, then a second only when the held-out macro F1 is higher. Do not plan a third.

## 4. Notebook shape

One Colab notebook, GPU runtime, under `notebooks/colab4/` when it is built later. It does not call MLflow.

1. Mount the account's Drive. Results go to `MyDrive/digikala-experiments/checkpoints/<experiment-name>/`.
2. Download the shared preprocessed CSV (Drive file `1vCkaGbSrXtlKUsdOrguUtuEYobT5vFxr`) if that stage is not already on Drive. The new preprocessing runs on top of that file, and the experiment config says stemming is not applied again.
3. Run the preprocessing stages in section 2. Write the stage text to Drive before the next stage.
4. Draw 20% of the labeled rows, stratified by `recommendation_status`, seed 2026. Within each class, `n_keep = round(0.20 * count)`. Print the three class counts before and after. This 20% is the only training text.
5. Load TookaBERT from the saved `best/` folder (shared folder `1mcsqBfSqyEtD8iu5G0l8poeYR7oaitLl`, directory `best/PartAI--TookaBERT-Base`). Reuse the top-4 unfreeze, batch 16, gradient accumulation 2, maximum length 128, fp16, seed 2026.
6. After each epoch, score **two** sets with the same metric function:
   - the other 80%, which never entered training
   - the full labeled set, which is what this plan is asked to publish

The full-set number includes the 20% training rows, so it will sit above the 80% number. The report prints both and says which is which. The 80% number is the one that decides a second epoch.

The first notebook is TookaBERT only. ParsBERT is a copy of the same notebook with the checkpoint name changed, after TookaBERT has a saved score.

## 5. Checkpoints

Every artifact name starts with the experiment name and the date: `tookabert-preproc-v2-2026-11-12`. A ParsBERT copy uses `parsbert-preproc-v2-2026-11-12`. The date is the day the experiment is created, not the day a later epoch happens to finish.

Stages, in order, each with its own Drive directory:

| Stage | What is saved |
| --- | --- |
| `download` | CSV size, Drive file id, sha256 of the zip |
| `preprocess-letters` | text after letter, diacritic, and half-space rules, plus the preprocessing config |
| `preprocess-digits` | text after the Persian digit fold and the pure-number deletion |
| `preprocess-emoji` | text after emoji removal, and the count of bodies deleted because they became empty |
| `sample` | row ids of the 20% train set, the seed, and the three class counts |
| `model` | tokenizer files, label map, which blocks are trainable, learning rates |
| `epoch-1`, `epoch-2` | `model.safetensors`, tokenizer, `metrics.json`, `heldout_metrics.json`, `config.json` |

`config.json` for every stage repeats the experiment name, date, seed, label map, preprocessing flags, and the stage name. `metrics.json` is the full labeled set. `heldout_metrics.json` is the 80%.

The notebook starts by listing `MyDrive/digikala-experiments/checkpoints/<experiment-name>/`. If a stage directory already contains its marker file (`DONE`), that stage is skipped and its files are read back. Training loads the latest `epoch-*` directory that contains `model.safetensors` and `trainer_state.json` when the epoch did not finish, and starts the next epoch when `DONE` is present. Optimizer state is saved inside the epoch directory so a killed Colab session does not repeat a finished epoch. Weights are copied to Drive before the full-set scoring pass, because scoring 5.26 million rows is the long part.

Local Trainer files stay on `/content`. Drive gets the stage directory. `save_total_limit` stays 1 inside a stage so the disk does not fill.

## 6. Metrics

On both the full labeled set and the held-out 80%, record:

- accuracy
- F1 macro and F1 micro
- per-class precision, recall, and F1 for `recommended`, `not_recommended`, and `no_idea`, with support

Macro F1 is the number that decides whether epoch 2 is kept. Micro F1 will sit close to accuracy because of the large `recommended` class. It is recorded so the two are not confused later. The same function writes both files, so a class cannot be dropped from one and not the other.

## 7. Next steps

1. Add `notebooks/colab4/01-tookabert-preproc-v2.ipynb` that follows sections 4 and 5.
2. Run it on a GPU account. Publish the full-set and the 80% metrics before starting ParsBERT.
3. If held-out macro F1 is above the section 12 TookaBERT validation 0.6722, run the Latin-deletion ablation on the same 20% ids.
4. Only then copy the notebook for ParsBERT.
5. Leave all-blocks, Shiraz, and XLM-RoBERTa until those two scores exist.

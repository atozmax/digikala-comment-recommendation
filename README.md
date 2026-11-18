# Digikala product review models

**۳۰ مهر ۱۴۰۵** (1405-07-30)

Phase 1 predicts recommendation status and an ordered rating band from the Persian review body. The descriptive audit is in `reports/en/00-report.md` and `reports/fa/00-report.md`.

## Data on Google Drive

On 30 Mehr 1405, `notebooks/colab/00-upload-to-google-drive-from-colab.ipynb` was run in Colab. The first cell mounts Drive. The notebook then downloaded the public release [RadeAI/Digikala_comments_products](https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/main) and saved two files:

| File | Drive path | Bytes |
| --- | --- | ---: |
| `digikala-comments.csv` | `MyDrive/digikala-products-recommender-systems/data/raw/digikala-comments.csv` | 1,278,526,959 |
| `digikala-products.csv` | `MyDrive/digikala-products-recommender-systems/data/raw/digikala-products.csv` | 258,733,765 |

The comments file matches the local audit copy at `data/raw/digikala-comments.csv` (6,156,289 rows, 15 columns). Its Drive header begins with a UTF-8 BOM, so read it with `encoding="utf-8-sig"`. The products file was stored and its header was checked. Its rows are not audited yet.

`digikala.rar` on the same page was not copied. It is the packed form of this release. A later run of the notebook skips either CSV when the Drive copy already has the size above.

The Colab machine does not keep these files after the runtime stops. The next session mounts Drive and reads the paths above.

## Commands

From the repository root, with `pandas` and `python-dotenv` installed. `python -m src` only lists commands. Name the script you want. Later scripts are added the same way.

Clean the comments:

```bash
python -m src cleaning
```

That reads `data/raw/digikala-comments.csv` and writes `data/cleaned/digikala-comments-cleaned.csv`. The same two paths are in `.env.example`. Copy that file to `.env` to keep them there. `.env` is not committed. The command loads `.env` first, then uses those paths when you do not pass flags.

Pass either path on the command line to override `.env`:

```bash
python -m src cleaning \
  --input data/raw/digikala-comments.csv \
  --output data/cleaned/digikala-comments-cleaned.csv
```

A flag wins over `.env`, and `.env` wins over the hardcoded defaults. A variable already set in the shell is left as it is. The command prints the loaded row count and how many rows each cleaning step drops, each with its percent of the loaded rows.

Preprocess the cleaned bodies, then tokenize that file:

```bash
python -m src preprocess
python -m src tokenize
```

`preprocess` reads `data/cleaned/digikala-comments-cleaned.csv` and writes `data/processed/digikala-comments-preprocessed.csv`. It maps Arabic letters onto Persian letters, folds ASCII and Arabic-Indic digits onto Persian digits, keeps one half-space inside a word, and keeps emoji and Latin letters. `tokenize` reads that CSV and writes `data/processed/digikala-comments-tokenized.parquet` with the `HooshvareLab/bert-fa-base-uncased` tokenizer, truncated at 512 tokens. The paths and the checkpoint name are in `.env.example`.

## API and Docker

The image contains the API and its libraries. Checkpoints stay on the host and are mounted read-only. `experiments/` holds the TookaBERT checkpoints. `models/` holds the two TF-IDF joblib files. Optimizer files under `trainer/` are not served.

```bash
docker compose up --build
```

| Service | URL |
| --- | --- |
| API docs | http://localhost:8000/docs |
| Metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana signs in as `admin` / `admin`. The Review API dashboard is in the Digikala folder. Anonymous view is enabled. Prometheus scrapes `GET /metrics` every 5 seconds. The series are defined in `src/api/metrics.py`: request rate and latency, predictions by model and label, preprocess / tokenize / embed / inference latency, batch size, model load time, empty-text counts, load errors, and how many checkpoints are loaded.

Routes are built from the files present when the process starts. Restart the API after adding a checkpoint.

`POST /models/{model_id}/load` reads one checkpoint into memory. `POST /models/{model_id}/unload` drops it. Predict and embed return 409 until that model is loaded. A second load keeps the copy already in memory.

Text pipelines, each with a `/batch` route:

| Route | What it does |
| --- | --- |
| `POST /preprocess/shared` | The stemmed text the TF-IDF models were fit on |
| `POST /preprocess/text-v2` | Letters, then digits, then emoji. No stemming |
| `POST /preprocess/full` | `shared`, then `text-v2`. Raw text for every TookaBERT checkpoint |

Tokenizers, each with a `/batch` route: `POST /tokenize/tookabert` (length 128) and `POST /tokenize/parsbert` (length 512). ParsBERT is downloaded on first use into the Compose Hugging Face cache.

Every checkpoint has four score routes, plus embed routes. `{id}` is the model id from `GET /models`.

| Route | Input |
| --- | --- |
| `POST /predict/recommendation/{id}` | Text already in the model's training surface form |
| `POST /predict/rating-band/{id}` | Same, for a rating-band checkpoint |
| `POST /predict/rate/{id}` | Same, for the untrained rate layer |
| `POST /recommendation/{id}` | Raw review text. Runs that model's pipeline, then scores |
| `POST /rating-band/{id}` | Raw review text for a rating-band checkpoint |
| `POST /rate/{id}` | Raw review text for the rate layer |
| `POST /embed/{id}` | Text as given. No character pipeline |

Each of those has a `/batch` route. A batch accepts up to 32 texts. The two untrained heads are `tookabert-rating-band-untrained` and `tookabert-rate-untrained`. Their responses set `"trained": false`.

Without Compose, from the repository root:

```bash
python -m src api
```

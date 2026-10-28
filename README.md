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

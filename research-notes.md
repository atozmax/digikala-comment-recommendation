# Research notes: tokenizers, Persian preprocessing, and DVC

# Tokenizers for mostly Persian reviews

For these Digikala reviews, fine-tune HooshvareLab/bert-fa-base-uncased and use the tokenizer that ships with that checkpoint. Keep FacebookAI/xlm-roberta-base as the backup.

The text is mostly Persian, with English mixed in as brand names and specs (Samsung, 128GB, battery). The tokenizer has to cover both scripts.

- An English-only vocabulary turns most Persian words into [UNK].
- A Persian vocabulary that never saw Latin characters does the same to odd model codes and brand spellings.
- A vocabulary shared across about 100 languages splits Persian words into more pieces, so a long review hits the 512-token cap sooner and the ending gets cut off.
- The English slice should stay as one or a few known pieces that already have pretrained vectors.

HooshvareLab/bert-fa-base-uncased is a Persian WordPiece model with a 100,000-token vocabulary (Apache-2.0, about 162M parameters). Its vocabulary already contains review English such as samsung, iphone, battery, android, wifi, usb, ram, gb, good, and quality. Pretraining includes Digikala magazine text, which is not these review labels. Do not start from HooshvareLab/bert-fa-base-uncased-sentiment-digikala. That checkpoint was already fine-tuned on Digikala user comments, and this project only allows pretrained weights from outside this dataset.

FacebookAI/xlm-roberta-base is the backup (SentencePiece, 250,002 tokens, Persian and English together, about 278M parameters). Odd Latin strings become English subwords instead of [UNK]. Persian text splits into more tokens, so the same max_length keeps less of the review. Switch to it only if a tokenization check of body shows ParsBERT’s unknown-token rate on the Latin slice is high.

Other encoders that exist and can be fine-tuned later:

- PartAI/TookaBERT-Base is a stronger recent Persian encoder (BPE, 48,000 tokens) if a later run is about Persian quality rather than the English slice.
- sbunlp/fabert and ViraIntelligentDataMining/AriaBERT are Persian BERT/RoBERTa-base alternatives.
- google-bert/bert-base-multilingual-cased includes Persian, but thinly. It is a weak baseline.
- lifeweb-ai/shiraz is a small MobileBERT (about 46.6M parameters) if GPU memory is the limit.

Skip an English-only BERT tokenizer, a whitespace split, and translating the reviews to English first. Colloquial Persian, half-spaces, and mixed brand names are where a translation changes the rating signal.

Measure token length with the chosen tokenizer before locking max_length. Every encoder above has a 512-token cap.

| Model | Tokenizer | Vocab | Languages | Fine-tune? |
| --- | --- | --- | --- | --- |
| HooshvareLab/bert-fa-base-uncased | WordPiece | 100,000 | Persian. Pretraining includes Digikala magazine, not these review labels. Apache-2.0. | Yes. BERT-base, about 162M parameters. |
| FacebookAI/xlm-roberta-base | SentencePiece BPE | 250,002 | 100 languages, including Persian and English. MIT. | Yes. About 278M parameters. |
| google-bert/bert-base-multilingual-cased | WordPiece | 119,547 | 104 languages, Persian included thinly. | Yes, weak Persian baseline. |
| PartAI/TookaBERT-Base | BPE, vocab 48,000 | 48,000 | Persian. Apache-2.0. | Yes. PartAI/TookaBERT-Large is the heavier sibling. |
| sbunlp/fabert | WordPiece | 50,000 | Persian blogs. License not stated on the card. | Yes. 124M parameters. |
| ViraIntelligentDataMining/AriaBERT | BPE | 60,000 | Persian, including user opinions. Apache-2.0. | Yes. RoBERTa-base depth. |
| lifeweb-ai/shiraz | WordPiece-style, [zwnj] is a real token | 50,000 | Persian. Apache-2.0. | Yes, small MobileBERT, 46.6M parameters. |

On Persian sentiment, the TookaBERT card’s own table puts TookaBERT-base above ParsBERT, XLM-RoBERTa-base, FaBERT, mBERT, and AriaBERT on DeepSentiPers. FaBERT’s paper, on a different setup, reports FaBERT ahead of ParsBERT. Treat those as separate scoreboards, not one ranking.

# Preprocessing user-written Persian text

After the whitespace cleanup already in src/cleaning.py, normalize characters so the same word becomes the same subwords. Leave the surface words the tokenizer was pretrained on.

normalize_bodies already turns line breaks and tabs into a space, collapses whitespace, and strips. It does not change letters, digits, emoji, half-spaces, or punctuation. Run the steps below after that, then collapse whitespace again and drop a body that became empty.

Libraries disagree, so do not call a normalizer with its defaults. Hazm maps ASCII digits to Persian, deletes tatweel and core diacritics, and inserts ZWNJ for می / ها / تر (that insertion is buggy: بندر ها becomes بندر‌ها, hazm issue 367). Parsivar maps Persian digits to ASCII and can rewrite dates and Pinglish. DadmaTools unify_chars also maps Persian digits to ASCII; full_cleaning=True deletes stopwords, punctuation, and numbers. ParsBERT’s pretraining only cites hazm as strip junk, then standardize Persian characters.

Do these by default:

1. NFKC. This folds Arabic presentation forms to base letters (يﺎ to ي, ﯼ to ی, ﻛ to ك) and fullwidth Latin and digits to ASCII. It does not turn ي/ك into ی/ک, and it does not merge Persian, Arabic-Indic, and ASCII digits.
2. Arabic letters to Persian letters, after NFKC. Map ك to ک, ي and ى to ی, أ/إ/ٱ to ا, ة/ۀ/ھ to ه, and ؤ to و. Keep آ separate from ا. These are different code points, so the tokenizer will not share pieces until they are one character.
3. Delete diacritics and tatweel (U+064B–U+065F, U+0670, U+06D6–U+06ED, and U+0640). Do this after NFKC so alef plus madda has already become آ.
4. Keep one interior half-space (U+200C). Collapse repeated ZWNJ to one, and delete a ZWNJ at the ends or next to a space. Also delete zero-width space, ZWJ, BOM, and bidi marks. می‌شود and میشود segment differently, and Persian pretraining kept the half-space inside the word. Do not insert missing half-spaces.
5. Keep digits, in one script. Fold Arabic-Indic ٠-٩ onto the script that dominates the tokenizer. For HooshvareLab/bert-fa-base-uncased, use Persian ۰-۹. For xlm-roberta-base, use ASCII 0-9.
6. Delete URL and email spans and replace them with a space. A URL becomes a long run of low-value pieces and crowds max_length. Do not substitute a made-up <URL> token unless that string is already in the vocabulary.
7. Cap a run of 3 or more identical characters at 2. عاااالی becomes عاالی, and !!!! becomes !!. A real double letter stays. Hazm’s published example collapses سلامممم all the way to سلام, which is stronger than this.

Leave these as written:

- English and brand names. Case belongs to the tokenizer’s own normalizer.
- Emoji. Reviews use them as the sentiment signal, including when the body is only an emoji. Stripping is an experiment if the tokenizer maps every emoji to one unknown piece.
- Phone numbers. A phone pattern also eats prices, order ids, and product codes.
- Punctuation such as ؟, !, and ،.
- Stopwords. Lists contain negation (نه, نیست, بدون), and removing them flips sentiment. The project already treats stopword removal as an experiment.
- Stemming and lemmatization. Hazm lemmas look like رفت#رو, which is not the surface form in the vocabulary. Rewriting the word also erases polarity (نمی‌خرم versus خریدم).

Do not run DadmaTools full_cleaning, Parsivar date normalizing, or Parsivar Pinglish conversion.

Finish with the same whitespace collapse as normalize_bodies, then drop a body that became empty.

# DVC for this dataset

DVC is not set up in this repo yet. Only .gitignore already keeps the CSVs, the cache, and weight files out of git. The copies on Google Drive are ordinary files, not a DVC remote. There is no .dvc/ directory, no dvc.yaml, and no *.dvc files.

What it is. DVC versions large files without putting the bytes in git. dvc add hashes the CSV, stores the bytes in a local cache, and writes a small pointer such as digikala-comments.csv.dvc. You commit the pointer. A teammate clones the repo and runs dvc pull to get the same file. The comments file is about 1.28 GB (1,278,526,959 bytes). The products file is 258,733,765 bytes. Committing them to git would make every clone heavy forever.

Why it fits this dataset. A commit can point at the raw comments file, the products file, and the cleaned CSV together. Later model weights (.pt, .bin, .safetensors) can be tracked the same way. You do not need a dvc.yaml pipeline just to version these files.

How to add it. The current ignore rules hide data/raw/* and data/cleaned/*, which would also hide the pointer files. Un-ignore *.dvc in those folders first. Show this gitignore fragment:

```
data/raw/*
!data/raw/.gitkeep
!data/raw/*.dvc
!data/raw/.gitignore
data/cleaned/*
!data/cleaned/.gitkeep
!data/cleaned/*.dvc
!data/cleaned/.gitignore
```

Then from the repo root:

```bash
pip install dvc
dvc init
dvc add data/raw/digikala-comments.csv data/raw/digikala-products.csv
dvc add data/cleaned/digikala-comments-cleaned.csv
git add .gitignore .dvc/config .dvc/.gitignore data/raw/*.dvc data/raw/.gitignore data/cleaned/*.dvc data/cleaned/.gitignore
```

Do not commit the CSVs, .dvc/cache, or .dvc/config.local. Secrets belong in the environment or in .dvc/config.local. .dvc/config is shared and may contain the remote URL. After a remote exists: dvc push. A teammate clones and runs dvc pull.

Remotes:

S3 or MinIO is the standard remote.

```bash
pip install "dvc[s3]"
dvc remote add -d storage s3://bucket/path
dvc remote modify storage endpointurl https://minio.example:9000
dvc push
```

The endpointurl line is only for MinIO or another S3-compatible server. Keys stay in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or in .dvc/config.local, not in git. The bucket identity needs list, get, put, and delete on that prefix.

Google Drive is not built in. The maintained plugin is dvc[gdrive] (dvc-gdrive, PyPI 3.0.1, January 2024; current DVC pins dvc-gdrive>=3,<4). It uses PyDrive2. dvc-gs is Google Cloud Storage (gs://), not Google Drive.

```bash
pip install "dvc[gdrive]"
dvc remote add -d storage gdrive://<folder-id>/dvcstore
dvc push
```

The folder id is the last segment of https://drive.google.com/drive/folders/<folder-id>, not the MyDrive/... path. Use a dedicated folder. gdrive://root is a bad shared remote because each Google account would read and write a different Drive. Google’s default DVC app is blocked ("This app is blocked"), so you need your own Google Cloud project, the Drive API enabled, and an OAuth desktop client. Each person signs in on the first push or pull. For CI, use a service account JSON key stored with dvc remote modify --local, and share the Drive folder with that service account. dvc push stores content-addressed objects in that folder. It will not reuse the CSVs already sitting in MyDrive/digikala-products-recommender-systems/data/raw/. Those existing copies stay a manual backup.

A database is not a DVC remote. DVC will not version the review rows inside Postgres. If you already query a database, dvc import-db can export a CSV snapshot, and you still push that file to S3 or Drive. The database password belongs in .dvc/config.local.

Recommendation: use S3 or MinIO if you have a bucket. Otherwise use a dedicated Drive folder with dvc[gdrive], and leave the existing Drive copies as a manual backup. Track three files separately: data/raw/digikala-comments.csv, data/raw/digikala-products.csv, and data/cleaned/digikala-comments-cleaned.csv. Add model weights the same way when they exist, after a gitignore exception so the pointer is committed and the weight file is not.

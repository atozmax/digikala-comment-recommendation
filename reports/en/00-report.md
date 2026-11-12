**(1405-07-30)** · Thursday, 22 October 2026

# Descriptive analysis: schema and missing values

This note records the first full-file audit in `notebooks/01-data-cleaning.ipynb`. Every count in sections 1–8 comes from that notebook: one pandas parse of `data/raw/digikala-comments.csv`, then `DataFrame.info()` and `DataFrame.isnull().sum()`. No row was sampled. Section 9 records the second pass, `notebooks/02-descriptive-analysis.ipynb`, which measures the items section 6 left open. Section 10 records the character pass in that same notebook: digits, letters, emoji, the half-space, and whitespace inside the raw `body`. Section 11 records the TF-IDF logistic-regression baseline in `notebooks/04-baseline.ipynb`. Section 12 records the recommendation fine-tunes in the four Colab notebooks. Section 13 records the 10% top-four runs in `notebooks/colab3/`. Section 14 records the published ParsBERT sentiment model scored on every labeled review.

## 1. File and parse

| Item | Measured value |
| --- | ---: |
| Path | `data/raw/digikala-comments.csv` |
| Size on disk | 1.28 GB (1,278,526,959 bytes) |
| Parsed rows | 6,156,289 |
| Columns | 15 |
| Index | `0 … 6,156,288` |
| Pandas memory estimate | 1.6 GB |

Rows are records returned by the CSV parser, not physical line breaks. A review body can contain newlines, so a line count would overstate the table.

The project brief describes this public Digikala set as about 7 million records. The file loaded here parses to **6,156,289 rows and 15 columns**. Later counts in this project should use that parsed total.

Empty and null-like tokens were coerced to missing before the null counts:

```python
df = pd.read_csv(
    "../data/raw/digikala-comments.csv",
    na_values=["", "nan", "NaN", "NAN", "null", "NULL", "None", "none", None, "-"],
)
```

A bare hyphen is therefore missing, not a category. `created_at` is still a string and has not been parsed. `rate` is `float64`. Its range is checked later in this same notebook; section 8 records that check.

## 2. Schema

Dtypes are those reported by `df.info()` after the load above.

| # | Column | Dtype | Role in this project |
| ---: | --- | --- | --- |
| 0 | `id` | int64 | Row identity |
| 1 | `title` | str | Not a model input |
| 2 | `body` | str | The only model input |
| 3 | `created_at` | str | Timestamp metadata; not parsed yet |
| 4 | `rate` | float64 | Source of `rating_band` only; not a feature |
| 5 | `recommendation_status` | str | Target of the recommendation task |
| 6 | `is_buyer` | bool | Metadata; not a feature |
| 7 | `product_id` | int64 | Grouped splitting only |
| 8 | `advantages` | str | Not a feature |
| 9 | `disadvantages` | str | Not a feature |
| 10 | `likes` | int64 | Not a feature |
| 11 | `dislikes` | int64 | Not a feature |
| 12 | `seller_title` | str | Not a feature |
| 13 | `seller_code` | str | Not a feature |
| 14 | `true_to_size_rate` | str | Not a feature |

The required model reads `body` and nothing else. `recommendation_status` is the recommendation label. `rate` is used only to derive the ordered bands `low` / `medium` / `high`. `product_id` is reserved for a product-grouped split.

## 3. Missing values

Denominator for every percentage: **6,156,289** parsed rows. “Present” is `rows − missing`.

| Column | Missing | Missing % | Present | Present % |
| --- | ---: | ---: | ---: | ---: |
| `id` | 0 | 0.00 | 6,156,289 | 100.00 |
| `title` | 2,865,303 | 46.54 | 3,290,986 | 53.46 |
| `body` | 637 | 0.010 | 6,155,652 | 99.990 |
| `created_at` | 0 | 0.00 | 6,156,289 | 100.00 |
| `rate` | 0 | 0.00 | 6,156,289 | 100.00 |
| `recommendation_status` | 894,426 | 14.53 | 5,261,863 | 85.47 |
| `is_buyer` | 0 | 0.00 | 6,156,289 | 100.00 |
| `product_id` | 0 | 0.00 | 6,156,289 | 100.00 |
| `advantages` | 5,448,084 | 88.50 | 708,205 | 11.50 |
| `disadvantages` | 5,744,017 | 93.30 | 412,272 | 6.70 |
| `likes` | 0 | 0.00 | 6,156,289 | 100.00 |
| `dislikes` | 0 | 0.00 | 6,156,289 | 100.00 |
| `seller_title` | 302,181 | 4.91 | 5,854,108 | 95.09 |
| `seller_code` | 302,181 | 4.91 | 5,854,108 | 95.09 |
| `true_to_size_rate` | 6,065,516 | 98.53 | 90,773 | 1.47 |

Seven columns are complete under this null policy: `id`, `created_at`, `rate`, `is_buyer`, `product_id`, `likes`, and `dislikes`.

## 4. What the missingness implies

**The text field is usable.** `body` is empty on 637 rows (0.010%). Those rows have no model input and cannot enter training or inference. The other 6,155,652 rows have a body. `title` is empty on 46.54% of rows; the input policy already excludes it, so that gap is not repaired by copying the title into the body.

**The recommendation task has a large labeled set and a natural unlabeled pool.** `recommendation_status` is missing on 894,426 rows (14.53%). Those rows are not supervised examples. The labeled pool is 5,261,863 rows (85.47%). Section 9 counts the classes inside that pool. The missing labels should stay missing: they are not a class to impute.

**Rating bands are not blocked by null rates.** `rate` has no missing values, so every row can be tested against the band edges (`low` below 2.5, `medium` from 2.5 up to but not including 4, `high` from 4 through 5). A complete column is not the same as a valid one. Section 8 records the range on the four-column frame: minimum 0.0, maximum 2500.0 before the rate filter, and 0.0–5.0 after it.

**Pros, cons, and size-fit are sparse side text.** `advantages` is filled on 11.50% of rows and `disadvantages` on 6.70%. Shoppers who use these fields write a pro more often than a con. `true_to_size_rate` is filled on 1.47% of rows, which fits a size-fit score that applies only to some products. All three columns are outside the model inputs, so their sparsity does not change the feature set.

**Seller fields fail together in the counts.** `seller_title` and `seller_code` each have 302,181 missing values (4.91%). Equal counts are consistent with one seller block being absent on the same rows. Section 9 checks that row by row: those 302,181 rows are exactly the non-buyers, and every buyer has a seller code.

**The frame is already heavy.** The CSV is 1.28 GB on disk and about 1.6 GB once loaded with these dtypes. Schema and null counts fit in one pass. Later full-file work (label vocabulary, text length, reviews per product) should stay on the full parsed file when the column is still present, using chunks or a columnar engine if a second in-memory copy becomes tight. The four-column cleaning in section 8 removes `recommendation_status`, so that label is not on the cleaned frame.

## 5. Decisions this pass supports

1. Quote **6,156,289 × 15** as the audited size, and keep the 1.28 GB file size next to it.
2. Drop the **637** rows with an empty `body` from every text model. Do not backfill them from `title`, `advantages`, or `disadvantages`.
3. Train the recommendation head only on the **5,261,863** rows with a `recommendation_status`. Hold the **894,426** unlabeled rows out of supervised loss. Do not fill that label.
4. Derive `rating_band` from `rate` only after the section 8 filter. That pass keeps rates in [0, 5], including 0 and fractional values, and removes the single rate of 2500. Null rate is not the failure mode.
5. Keep `product_id` for grouped splitting. It is complete, which removes null-product rows as a split problem. After the empty-body drop and before the id drop, the notebook counts 331,593 distinct product ids. Section 9 measures reviews per product on the full raw file, where the distinct count is 331,599.
6. Leave `title`, seller fields, likes, dislikes, advantages, disadvantages, and `true_to_size_rate` out of the feature vector, matching the input policy. Their missingness is documented so it is not silently “fixed.”

## 6. Not measured in this notebook

Section 8 records the rate range, the one out-of-range rate, duplicate ids, exact duplicate rows, repeated bodies, and 331,593 distinct `product_id` values before the id drop. The following Part A items are still open. They are absent from `notebooks/01-data-cleaning.ipynb`, so this report does not estimate them:

- unique sellers, and the distribution of reviews per product (the distinct product count after the id drop is not printed)
- identical review text with conflicting `recommendation_status` or `rate` (the recommendation column was removed before the duplicate pass)
- the vocabulary of `recommendation_status`, and counts for `recommended`, `not_recommended`, and `no_idea`
- counts of the three rating bands (the notebook prints raw `rate` value counts; it does not assign the bands)
- character and word length of `body`, including very short and very long reviews
- whether missing `recommendation_status` depends on `is_buyer`, `rate`, or product frequency

Those measurements are in section 9. They come from `notebooks/02-descriptive-analysis.ipynb` on the same raw file. This section does not estimate them from notebook 01.

## 7. Copy on Google Drive

On 30 Mehr 1405 the Hugging Face release was downloaded in Colab and written to Google Drive. The notebook is `notebooks/colab/00-upload-to-google-drive-from-colab.ipynb`. Its first cell mounts Drive. A file already on Drive with the expected size is left in place.

Source: [RadeAI/Digikala_comments_products](https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/main).

| File | Drive path | Bytes | Header written by the run |
| --- | --- | ---: | --- |
| `digikala-comments.csv` | `MyDrive/digikala-products-recommender-systems/data/raw/digikala-comments.csv` | 1,278,526,959 | `id`, `title`, `body`, `created_at`, `rate`, `recommendation_status`, `is_buyer`, `product_id`, `advantages`, `disadvantages`, `likes`, `dislikes`, `seller_title`, `seller_code`, `true_to_size_rate` |
| `digikala-products.csv` | `MyDrive/digikala-products-recommender-systems/data/raw/digikala-products.csv` | 258,733,765 | `id`, `title_fa`, `Rate`, `Rate_cnt`, `Category1`, `Category2`, `Brand`, `Price`, `Seller`, `Is_Fake`, `min_price_last_month`, `sub_category` |

The comments file is the same 1,278,526,959-byte table audited above from `data/raw/digikala-comments.csv`. The Colab header print starts with a UTF-8 BOM before `id`, so a later read of the Drive copy should use `encoding="utf-8-sig"` if the first column must be exactly `id`. The products file was saved and its header was checked. Its rows have not been audited yet.

`digikala.rar` on the same page was not copied. It is the packed form of this release.

These two files stay on Drive after the Colab runtime stops. The next session mounts Drive again and reads those paths. Nothing from that run is kept in Colab memory.

## 8. Cleaning after the column cut

The same notebook then kept four columns and cleaned those rows. Every count in this section is printed in `notebooks/01-data-cleaning.ipynb`.

### Columns and missing bodies

The columns kept are `id`, `body`, `product_id`, and `rate`. The other eleven columns were removed, including `recommendation_status`. On that four-column frame, before any row drop, `df.shape` was `(6,156,289, 4)`. Missing values were `body` 637 and `id`, `product_id`, and `rate` 0. Rows with any missing value: 637, the same empty bodies.

`df.dropna(subset=['body'])` dropped those 637 rows. The shape was `(6,155,652, 4)` and every missing count was 0.

That frame was written with `df.to_csv('../data/cleaned/digikala-comments-cleaned.csv', index=False)` and then read back. The rate and duplicate measurements below are on that read-back, still before the rate-range filter and the id drop. A later cell writes the same path again after the cleaning cell. The first write does not contain those later drops. The second write is the post-clean frame. This report does not re-read the file.

The four-column frame has no recommendation label. Class counts for `recommendation_status` still have to come from the 15-column file.

### Rate

On the read-back, `rate` ran from 0.0 to 2500.0. Exactly one row sat outside [0, 5]: `id` 38631814, body «کوچک و با کیفیت», `product_id` 5095301, rate 2500.0. Rows with rate 0: 528,998. Rate 0 is a real score and was kept. Grouping by `rate` shows 178 distinct values, including fractional rates inside 0–5 (for example 0.05 and 4.88) and the single 2500. Fractional rates inside the interval were kept.

The cell `df[(df['rate'] > 5) | (df['rate'] < 0)].dropna(subset=['body'], inplace=True)` does not remove that row from `df`. `dropna(..., inplace=True)` runs on the filtered result, not on `df`. The unique-product count and the duplicate counts that follow are still on this pre-clean frame. `len(df['product_id'].unique())` printed 331,593. That count is after the empty-body drop and before the cleaning cell. It was not printed again after the id drop.

### Two duplicate counts, and the id drop

`df[df.duplicated()]` uses `keep='first'` on all columns. It showed 2,957 extra copies of rows that match on `id`, `body`, `product_id`, and `rate`.

`duplicates = df[df.duplicated(keys, keep=False)]`, with those four columns as keys, then `len(duplicates)`, was 5,911. `keep=False` counts every row in an exact-duplicate group, including the copy that is kept. 5,911 − 2,957 = 2,954 rows that have at least one exact copy. A few ids appear three times, which is why the extra copies are 2,957 rather than 2,954.

A note at the end of the notebook records the finding from that analysis. The cleaning drop uses `df['id'].duplicated().sum()`, which the note gives as 3,229, and `drop_duplicates(subset=['id'], keep='first')` removes those 3,229 surplus rows. One row per id is kept, so the drop is 3,229, not 5,911. Dropping 5,911 would delete the copy that should stay. 3,229 is larger than 2,957 because 272 rows share an id but do not match on `body`, `product_id`, or `rate`. They are still removed, because the rule is one row per id. After the id drop, `drop_duplicates()` on the full row removes 0 rows, because the exact copies already shared an id.

### The cleaning cell

The cleaning cell, in order, keeps rates in [0, 5], normalizes `body` by replacing `\r`, `\n`, and `\t` with a space, collapsing whitespace, and stripping, drops bodies that are empty after that, drops duplicate ids (`keep='first'`), drops any remaining full-row duplicates (`keep='first'`), and counts repeated bodies without dropping them.

Its stored stdout is:

```
dropped out-of-range rates: 0
dropped empty bodies after cleanup: 0
duplicate ids: 0
dropped duplicate ids: 0
dropped duplicate rows: 0
rows with a repeated body: 2117933
(6152422, 4)
rate range: 0.0 5.0
```

`df.info()` after the cell shows 6,152,422 entries, index `0 … 6,152,421`, and the same four columns. The rate range on that frame is 0.0–5.0.

Repeated bodies were counted and kept. `df['body'].duplicated().sum()` after the id drop is 2,117,933. That is the number of extra copies under `keep='first'`, not every row that shares a body. Short texts such as «عالی» are real reviews of different products.

The stored drop counts do not add up to the change in shape. The reloaded frame has 6,155,652 rows. The frame after the cleaning cell has 6,152,422. The difference is 3,230, and no cell between the reload and the final `info()` removes rows except this one. The note's 1 out-of-range rate and 3,229 duplicate ids, with 0 empty bodies and 0 full-row duplicates, sum to 3,230. This report takes the size of those drops from the note and from the two shapes. It does not replace the note's 3,229 with the 0 printed in the stored stream. The stream's 2,117,933 repeated bodies, the final shape `(6,152,422, 4)`, and the final rate range 0.0–5.0 are the figures that stream and the following `info()` agree on. Empty bodies after cleanup are 0 in both the stream and the note.

## 9. Second descriptive pass

Every count in this section is printed in `notebooks/02-descriptive-analysis.ipynb`. The notebook scans `data/raw/digikala-comments.csv` with the same missing-value tokens as section 1 and keeps `id`, `body`, `rate`, `recommendation_status`, `is_buyer`, `product_id`, and `seller_code`. No row is sampled. The denominator is the full parsed file, 6,156,289 rows, not the cleaned four-column frame.

Whitespace in `body` is collapsed before length and repeated-text counts. Those counts leave out the 637 bodies that are empty after that collapse, so their denominator is 6,155,652. Rating bands are `low` for 0 ≤ rate < 2.5, `medium` for 2.5 ≤ rate < 4, `high` for 4 ≤ rate ≤ 5, and `invalid` outside that interval. Word length is a split on spaces. It is not an encoder token count.

### Products and sellers

| Item | Value |
| --- | ---: |
| Parsed rows | 6,156,289 |
| Distinct products | 331,599 |
| Distinct seller codes | 29,213 |
| Rows with no seller code | 302,181 |
| Empty bodies after whitespace collapse | 637 |
| File size | 1,278,526,959 bytes |

The 331,599 products are the raw file. Section 8's 331,593 is the count after the 637 empty bodies were dropped. The six-product gap is that difference in frame, not a second catalog.

`is_buyer` and a missing seller code are the same split. A group count hides empty pairings, so the notebook writes the two zeros explicitly:

| is_buyer | seller missing | rows |
| --- | --- | ---: |
| false | false | 0 |
| false | true | 302,181 |
| true | false | 5,854,108 |
| true | true | 0 |

Every review with no seller is a non-buyer. Every review with a seller is a buyer. The file has no buyer without a seller and no non-buyer with one.

Reviews per product are skewed. The median product has 4 reviews. The mean is 18.57. The 75th percentile is 13, the 90th is 46, the 99th is 200, and the maximum is 400. 91,944 products have one review. 18,047 have at least 100.

| Product size | Products | Reviews | Share of reviews |
| --- | ---: | ---: | ---: |
| 1 | 91,944 | 91,944 | 1.49% |
| 2–5 | 106,514 | 322,841 | 5.24% |
| 6–20 | 73,204 | 788,532 | 12.81% |
| 21–100 | 42,033 | 1,854,574 | 30.12% |
| 101 or more | 17,904 | 3,098,398 | 50.33% |

The busiest 1% of products is 3,315 products and holds 667,725 reviews (10.85%). The busiest 10% is 33,159 products and holds 4,132,077 reviews (67.12%). A row-wise split would put many products on both sides of the cut. The split has to be grouped by `product_id`.

### Missing recommendation labels

The 894,426 missing labels are not spread evenly.

| is_buyer | Label missing | Rows | Share of that buyer flag |
| --- | --- | ---: | ---: |
| false | yes | 302,181 | 100% |
| true | no | 5,261,863 | 89.88% |
| true | yes | 592,245 | 10.12% |

Every non-buyer is unlabeled. Among buyers, 10.12% are unlabeled. The unlabeled pool is the entire non-buyer block plus that buyer slice. 302,181 + 592,245 = 894,426.

By rating band, the missing share is 17.04% of `high` (682,505 of 4,004,905), 10.50% of `medium` (111,207 of 1,059,054), and 9.22% of `low` (100,714 of 1,092,329). The single out-of-range rate is labeled.

By product size, the missing share falls as the product gets more reviews: 20.30% of one-review products, 18.98% of products with 2–5 reviews, 17.20% with 6–20, 15.07% with 21–100, and 12.89% with 101 or more. Missingness is tied to buyer status, score, and product frequency. It is not a class to fill in.

### Recommendation classes and rating bands

| recommendation_status | Rows | Share of all rows | Share of labeled rows |
| --- | ---: | ---: | ---: |
| `recommended` | 4,162,839 | 67.62% | 79.11% |
| missing | 894,426 | 14.53% | — |
| `no_idea` | 594,121 | 9.65% | 11.29% |
| `not_recommended` | 504,903 | 8.20% | 9.60% |

Among labeled rows, `recommended` is about eight times `not_recommended`. Accuracy would reward the large class. The recommendation head needs class weights and macro F1.

On the raw file, `rate` runs from 0.0 to 2500.0. The mean is 3.6483, the median is 4.0, and the standard deviation is 1.8619. The 25th percentile is 3.0. There are 178 distinct values. 228,128 rows (3.71%) are not whole numbers. 529,098 rows are exactly 0. One row is outside [0, 5].

| rating_band | Rows | Share of all rows |
| --- | ---: | ---: |
| `low` | 1,092,329 | 17.74% |
| `medium` | 1,059,054 | 17.20% |
| `high` | 4,004,905 | 65.05% |
| `invalid` | 1 | 0.00% |

`high` is the majority band. The one invalid rate is the 2500 from section 8, and it sits on a `recommended` row.

The numeric rate tracks the label. Median rate is 4.0 for `recommended`, 4.0 for a missing label, 3.0 for `no_idea`, and 1.0 for `not_recommended`. Means are 3.934, 4.041, 2.923, and 1.448. Inside each labeled class:

| recommendation_status | low | medium | high |
| --- | ---: | ---: | ---: |
| `recommended` | 11.12% | 11.82% | 77.06% |
| `no_idea` | 17.54% | 65.32% | 17.14% |
| `not_recommended` | 84.06% | 13.44% | 2.50% |

The raw value still should not be a feature. It has 178 levels, including fractions, and a score of 5 is not five times a score of 1. The ordered band is the target.

### Body length

On the 6,155,652 non-empty bodies, character length has minimum 1, median 29, 75th percentile 57, 90th percentile 106, and 95th percentile 152. 827,860 bodies (13.45%) have at most 10 characters. Whitespace words have median 6, 95th percentile 32, 99th percentile 62, and maximum 908. 453,793 bodies (7.37%) are a single word. 1,675,526 (27.22%) have at most three words. 7,527 have at least 128 words, 803 (0.0130%) have at least 256, and 69 have at least 512.

| Words | Bodies | Share of non-empty bodies |
| --- | ---: | ---: |
| 1 | 453,793 | 7.37% |
| 2–3 | 1,221,733 | 19.85% |
| 4–5 | 1,174,681 | 19.08% |
| 6–10 | 1,521,668 | 24.72% |
| 11–20 | 1,083,274 | 17.60% |
| 21–40 | 521,716 | 8.48% |
| 41–80 | 149,257 | 2.42% |
| 81–160 | 25,853 | 0.42% |
| 161 or more | 3,677 | 0.06% |

Negative reviews are longer. Median words are 10 for `not_recommended`, 8 for `no_idea`, 6 for `recommended`, and 5 for a missing label. The 95th percentile is 45, 38, 28, and 32 words. By band, median words are 8 for `low`, 7 for `medium`, and 5 for `high`.

A 512-token cap cuts a thin tail. The 99th percentile is 62 whitespace words, and 803 bodies reach 256 words. One Persian word can be several encoder tokens, so the token length is longer than this word count. The cap is still far above the body of the distribution.

### Repeated ids and repeated text

On the raw file, 3,226 ids occur more than once. Those ids cover 6,455 rows, so the extra copies are 6,455 − 3,226 = 3,229. That is the same surplus the cleaning note in section 8 records. Of the repeated ids, 43 carry two recommendation values, 79 carry two rates, and 155 carry two bodies.

After whitespace collapse there are 4,034,601 distinct bodies. 193,882 of those strings repeat, and the reviews that use them number 2,314,933. 31,409 repeated strings carry more than one recommendation value (1,609,737 rows). 117,355 carry more than one rate (2,105,785 rows).

The texts that repeat most, and disagree on the label, are short and appear on many products. «عالی» is 134,693 reviews, all three recommendation values, 33 distinct rates, and 60,174 products. «خوب بود», «خوبه», and «خوب» are the next three. A shared body is not one training label. Those rows stay separate.

### Decisions this pass adds

1. Score the recommendation head with macro F1 and class weights. `recommended` is 79.11% of labeled rows and `not_recommended` is 9.60%.
2. Split by `product_id`. The busiest tenth of products holds 67.12% of reviews.
3. Keep the 894,426 unlabeled rows out of supervised loss. All 302,181 non-buyers are in that pool, so the gap is structural.
4. Train `rating_band`, not the raw rate. Keep the single rate of 2500 out of the bands.
5. A 512-token cap truncates the far tail. The 99th percentile is 62 whitespace words.
6. Do not merge reviews that share a body. 31,409 repeated texts carry more than one recommendation value.

## 10. Characters inside the raw body

The last cell of `notebooks/02-descriptive-analysis.ipynb` reads the original `body`, before spaces and line breaks are collapsed. The denominator is every parsed review, 6,156,289, including empty bodies. A review that contains two kinds of character is counted in both. The saved output of that cell is the printed reading below. It does not include a separate match total for each pattern.

172 reviews are only an emoji once spaces are removed. That is 0.00% of the file at two decimal places. An emoji-only body is still a review.

| Digit script | Share of reviews |
| --- | ---: |
| ASCII `0-9` | 1.56% |
| Persian `۰-۹` | 3.17% |
| Arabic-Indic `٠-٩` | 0.02% |

2,796 reviews use both ASCII and Persian digits. 13 use both ASCII and Arabic-Indic. 166 use both Persian and Arabic-Indic. The three scripts are different code points, so `5`, `۵`, and `٥` do not share a subword until they are folded.

| Letter | Share of reviews |
| --- | ---: |
| Persian kaf or ye, `ک` or `ی` | 90.13% |
| Arabic kaf or ye, `ك`, `ي`, or `ى` | 0.82% |
| Latin letter | 0.65% |

Of the reviews with a Latin letter, 36,765 also contain an Arabic-script letter and 3,307 do not. The Latin slice is brand names and model codes inside Persian text, not a second corpus. Other scripts are almost absent: Cyrillic 1 review, Hebrew 1, Greek 2, Armenian 1, Han 2, and Hangul 5.

A half-space (ZWNJ, U+200C) appears in 3.15% of reviews. 4,201 reviews have two half-spaces in a row. 33,976 have a half-space at an end or next to a space. One half-space inside a word is part of the Persian spelling. The repeated and edge copies are the ones a cleaner can drop.

Line breaks appear in 9.21% of reviews. A run of two or more ordinary spaces appears in 3.42%. The tab share rounds to 0.00%. A no-break space appears in 234 reviews, a zero-width space in 7, a zero-width joiner in 911, and a byte-order mark in 2. The line breaks, tabs, and repeated spaces are what the whitespace collapse in cleaning removes.

### What this pass changes

1. Fold Arabic-Indic and ASCII digits onto one digit script before tokenization. Persian digits are the largest of the three, and a few thousand reviews mix two scripts.
2. Map Arabic `ك`, `ي`, and `ى` onto Persian `ک` and `ی`. The Arabic forms are in 0.82% of reviews, enough to split a spelling from the Persian form.
3. Leave Latin letters as written. They are the brand and model-code slice, and 3,307 reviews have Latin letters with no Arabic-script letter.
4. Keep emoji. 172 reviews are emoji only once spaces are removed, so deleting emoji deletes those reviews' only text.
5. Keep one half-space inside a word. Collapse a repeated half-space, and delete a half-space at an end or beside a space. That is 4,201 and 33,976 reviews, not the whole 3.15%.
6. Keep collapsing line breaks and repeated spaces. Ignore Cyrillic, Hebrew, Greek, Armenian, Han, and Hangul as a preprocessing rule. Together they are 12 reviews.

## 11. TF-IDF baseline for rating bands

`notebooks/04-baseline.ipynb` loads `data/tokenized/digikala-comments-tokenized.parquet` with Polars. The file has 6,152,207 rows. `rate` is mapped to `rating_band` and is not a model feature. Every row in this file falls inside a band: high 4,002,165 (65.05%), low 1,091,653 (17.74%), medium 1,058,389 (17.20%).

The split is grouped by `product_id`, with seed 2026. The shares are 70% train, 15% validation, and 15% test. A product's reviews stay in one split. The notebook printed 0 products in more than one split.

| Split | High | Low | Medium | Rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,801,508 | 764,155 | 740,859 | 4,306,522 |
| Validation | 600,325 | 163,751 | 158,765 | 922,841 |
| Test | 600,332 | 163,747 | 158,765 | 922,844 |

The features are the ParsBERT tokenizer ids of `body`, with the classifier, separator, and pad ids removed. TF-IDF is fit on the training split only, with `min_df=5`. The vocabulary is 34,185 terms. Logistic regression uses balanced class weights, solver `saga`, and `max_iter=200`. `product_id`, `rate`, and `id` are not features.

On the test split the accuracy is 0.6319, the balanced accuracy is 0.5697, the macro F1 is 0.5492, and the ordinal distance error is 0.4866. High is 65% of the rows, so accuracy sits above the balanced accuracy and the macro F1.

| Band | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.4608 | 0.5128 | 0.4854 | 163,747 |
| Medium | 0.3436 | 0.4959 | 0.4060 | 158,765 |
| High | 0.8219 | 0.7003 | 0.7562 | 600,332 |

The test confusion matrix has rows actual low, medium, high and columns predicted in that order:

| Actual | Predicted low | Predicted medium | Predicted high |
| --- | ---: | ---: | ---: |
| Low | 83,962 | 35,455 | 44,330 |
| Medium | 33,264 | 78,733 | 46,768 |
| High | 64,974 | 114,942 | 420,416 |

Validation is nearly the same: accuracy 0.6316, balanced accuracy 0.5705, macro F1 0.5495, ordinal distance error 0.487. The last cell of the notebook writes the fitted vectorizer and classifier to `models/rating-band-tfidf-logreg.joblib`. The `models/` directory is gitignored except `.gitkeep`.

## 12. Recommendation fine-tunes on Colab

The four notebooks under `notebooks/colab/` fine-tune a classifier on `recommendation_status`. The only text field is `body`. The label has three classes: `recommended`, `not_recommended`, and `no_idea`. Each notebook was downloaded from Colab with its cell outputs. The counts below are those printed outputs.

The saved Drive folders for the three finished accounts are:

| Notebook | Drive folder |
| --- | --- |
| `notebooks/colab/01-parsbert-finetune.ipynb` | https://drive.google.com/drive/folders/1GKqsgeXT8uFFLYddJDnA9509bIiDHosd?usp=sharing |
| `notebooks/colab/02-xlmr-finetune.ipynb` | https://drive.google.com/drive/folders/1AEtn221vAuEW3yAzgQ6-JSeJd0-keHpm?usp=sharing |
| `notebooks/colab/03-tookabert-finetune.ipynb` | https://drive.google.com/drive/folders/1mcsqBfSqyEtD8iu5G0l8poeYR7oaitLl?usp=sharing |

Each folder contains `best/`, `logs/`, and `runs/`. The small files from those folders are copied under `artifacts/colab/`, with the file list in `artifacts/colab/manifest.json` and the links in `artifacts/colab/drive-links.json`. `model.safetensors` and `tokenizer.json` stayed on Drive. TookaBERT's `logs/` folder is empty. Shiraz has no shared folder here.

The split is the same in every notebook: 3,680,838 train, 788,750 validation, and 788,754 test. The class counts printed for that split are:

| Split | recommended | not_recommended | no_idea |
| --- | ---: | ---: | ---: |
| Train | 2,912,158 | 353,125 | 415,555 |
| Validation | 624,034 | 75,669 | 89,047 |
| Test | 624,035 | 75,671 | 89,048 |

Shared training settings are one epoch, maximum length 128, batch size 16, gradient accumulation 2, seed 2026, head learning rate `1e-3`, last-block learning rate `2e-5`, and head learning rate `1e-4` while the last block is unfrozen. The loss is class-weighted cross-entropy. The printed precision, recall, and F1 are macro averages over the three classes. One epoch is 115,027 optimizer steps. Stage 1 trains the new classifier with the encoder frozen. Stage 2 loads that head from Drive and trains the last encoder block. The run with the higher validation macro F1 is copied to `MyDrive/digikala-experiments/best/`.

ParsBERT, XLM-RoBERTa, and TookaBERT each completed both stages. The head cell found local checkpoint `checkpoint-115027`, wrote the head under `runs/`, and the scoring cell printed validation and test. The last-block cell then trained from that head and updated `best/`.

| Model | Stage | Split | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ParsBERT | head | validation | 0.7340 | 0.5445 | 0.6389 | 0.5715 |
| ParsBERT | head | test | 0.7343 | 0.5453 | 0.6407 | 0.5725 |
| ParsBERT | last block | validation | 0.8004 | 0.6395 | 0.7266 | 0.6687 |
| ParsBERT | last block | test | 0.8000 | 0.6390 | 0.7263 | 0.6682 |
| XLM-RoBERTa | head | validation | 0.7135 | 0.5300 | 0.6271 | 0.5540 |
| XLM-RoBERTa | head | test | 0.7129 | 0.5299 | 0.6277 | 0.5539 |
| XLM-RoBERTa | last block | validation | 0.7989 | 0.6329 | 0.7205 | 0.6630 |
| XLM-RoBERTa | last block | test | 0.7993 | 0.6334 | 0.7214 | 0.6637 |
| TookaBERT | head | validation | 0.7456 | 0.5444 | 0.6310 | 0.5718 |
| TookaBERT | head | test | 0.7450 | 0.5435 | 0.6301 | 0.5707 |
| TookaBERT | last block | validation | 0.8053 | 0.6420 | 0.7289 | 0.6722 |

TookaBERT's notebook prints only `best model updated` and validation macro F1 0.6722 for the last block. The Drive file `runs/PartAI--TookaBERT-Base--last_block/metrics.json` is that same validation pass: accuracy 0.8053, macro precision 0.6420, macro recall 0.7289, macro F1 0.6722. That folder has no `test_metrics.json`. The file `best/PartAI--TookaBERT-Base/test_metrics.json` is still the head test, with `"stage": "head"` and macro F1 0.5707, the same numbers the score cell printed.

| Model | Checkpoint | Head parameters | Last-block parameters | Best directory |
| --- | --- | ---: | ---: | --- |
| `HooshvareLab/bert-fa-base-uncased` | `HooshvareLab--bert-fa-base-uncased` | 2,307 | 7,090,179 | `best/HooshvareLab--bert-fa-base-uncased` |
| `FacebookAI/xlm-roberta-base` | `FacebookAI--xlm-roberta-base` | 592,899 | 7,680,771 | `best/FacebookAI--xlm-roberta-base` |
| `PartAI/TookaBERT-Base` | `PartAI--TookaBERT-Base` | 2,307 | 7,090,179 | `best/PartAI--TookaBERT-Base` |

Each head directory on Drive contains `config.json`, `metrics.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`. The last-block run is the saved best for all three, because its validation macro F1 is the higher of the two runs. Among those three, TookaBERT has the highest validation macro F1, 0.6722, and the highest validation accuracy, 0.8053. ParsBERT is 0.6687 and 0.8004. XLM-RoBERTa is 0.6630 and 0.7989. On the test files whose stage is `last_block`, ParsBERT macro F1 is 0.6682 and XLM-RoBERTa is 0.6637.

The per-class F1 in those Drive metrics files:

| Model | Split | Stage | recommended | not_recommended | no_idea |
| --- | --- | --- | ---: | ---: | ---: |
| ParsBERT | validation | last block | 0.8947 | 0.6902 | 0.4211 |
| ParsBERT | test | last block | 0.8944 | 0.6881 | 0.4221 |
| XLM-RoBERTa | validation | last block | 0.8943 | 0.6824 | 0.4123 |
| XLM-RoBERTa | test | last block | 0.8944 | 0.6815 | 0.4151 |
| TookaBERT | validation | last block | 0.8983 | 0.6969 | 0.4214 |

`recommended` is near 0.90 F1. `no_idea` stays near 0.42. ParsBERT's log also records the head epoch before the save crash: validation accuracy 0.7340, macro F1 0.5715, eval loss 0.6765, train loss 1.6569. The XLM-RoBERTa and TookaBERT head `metrics.json` files say `saved after the scoring crash` and contain no scores. Their head scores are the ones printed in the notebooks.

`notebooks/colab/04-shiraz-finetune.ipynb` does not contain a finished run. The model is `lifeweb-ai/shiraz`. The train cell loads it, prints 1,539 trainable head parameters, and then raises `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'`. The save cell raises `NameError: name 'DRIVE_DIR' is not defined`. The score cell and the last-block cell have no output. The MLflow cell raises `FileNotFoundError: No Trainer checkpoint under /content/checkpoints`. No validation metric, test metric, `runs/` path, or `best/` path is printed for Shiraz.

## 13. Top four blocks on a 10% slice

`notebooks/colab3/01-tookabert-top4.ipynb` and `notebooks/colab3/02-parsbert-top4.ipynb` were downloaded from Colab with their cell outputs. Each run starts from the section 12 `best/` weights, unfreezes blocks 8–11, and trains on a random 10% of the labeled reviews. The draw uses seed 2026. Both notebooks print the same slice: 525,834 kept out of 5,258,342, then 368,082 train, 78,874 validation, and 78,878 test.

| Split | recommended | not_recommended | no_idea |
| --- | ---: | ---: | ---: |
| Train | 291,060 | 35,582 | 41,440 |
| Validation | 62,370 | 7,624 | 8,880 |
| Test | 62,371 | 7,626 | 8,881 |

Shared settings match section 12 except the slice and the blocks: maximum length 128, batch size 16, gradient accumulation 2, block learning rate `2e-5`, classifier learning rate `1e-4`, class-weighted cross-entropy, and macro precision, recall, and F1. Both notebooks print 28,353,795 trainable parameters. The scores below are on this slice. They are not a new measurement of the full test set in section 12.

The saved Drive folders are:

| Notebook | Drive folder |
| --- | --- |
| `notebooks/colab3/01-tookabert-top4.ipynb` | https://drive.google.com/drive/folders/1eSvV-_vO_Pgygj34poKD8ZmJfiO5w6tI?usp=sharing |
| `notebooks/colab3/02-parsbert-top4.ipynb` | https://drive.google.com/drive/folders/1uewUGcYTCkjs0UC4JH9IB2GOZvzIjdd2?usp=sharing |

Each folder contains `best/`, `logs/`, and `runs/`. The small files are under `artifacts/colab/sample10/`, with `artifacts/colab/sample10/manifest.json` and `artifacts/colab/sample10/drive-links.json`. `model.safetensors` and `tokenizer.json` stayed on Drive. The links are also appended in `artifacts/colab/drive-links.json`.

TookaBERT keeps epoch 2. ParsBERT keeps epoch 1. Later epochs in the log have a lower validation macro F1, so `best/` was not replaced. TookaBERT's log has no epoch 3: the second extra pass is numbered 4. The epoch 5 validation line repeats the epoch 4 scores exactly. ParsBERT's epoch 4 line repeats epoch 2, and epoch 5 repeats epoch 3. The kept files are the epochs named below.

| Model | Kept epoch | Split | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| TookaBERT | 2 | validation | 0.8155 | 0.6546 | 0.7334 | 0.6825 |
| TookaBERT | 2 | test | 0.8173 | 0.6595 | 0.7365 | 0.6867 |
| ParsBERT | 1 | validation | 0.8095 | 0.6494 | 0.7321 | 0.6778 |
| ParsBERT | 1 | test | 0.8109 | 0.6530 | 0.7359 | 0.6816 |

Per-class F1 in those `best/` files:

| Model | Split | recommended | not_recommended | no_idea |
| --- | --- | ---: | ---: | ---: |
| TookaBERT | validation | 0.9058 | 0.7014 | 0.4401 |
| TookaBERT | test | 0.9062 | 0.7131 | 0.4409 |
| ParsBERT | validation | 0.9013 | 0.6972 | 0.4348 |
| ParsBERT | test | 0.9013 | 0.7064 | 0.4370 |

On this slice, TookaBERT's kept test macro F1 is 0.6867 and ParsBERT's is 0.6816. `recommended` stays near 0.90 F1. `no_idea` stays near 0.44. The saved directories are `best/PartAI--TookaBERT-Base--top4_sample10` and `best/HooshvareLab--bert-fa-base-uncased--top4_sample10`.

## 14. Published ParsBERT sentiment on every labeled review

`notebooks/colab3/05-hooshvare-digikala-sentiment.ipynb` was downloaded from Colab with its cell outputs. It does not train. It loads `HooshvareLab/bert-fa-base-uncased-sentiment-digikala` and scores every labeled review in the preprocessed CSV. The cell prints 5,258,342 labeled reviews out of 5,258,342 rows in that file: 4,160,227 `recommended`, 504,465 `not_recommended`, and 593,650 `no_idea`. The model's own index order, printed in the notebook, is `no_idea=0`, `not_recommended=1`, `recommended=2`. The metrics use this project's order. The printed precision, recall, and F1 are macro averages.

The score file is in the TookaBERT Drive folder, which is the same folder as `notebooks/colab/03-tookabert-finetune.ipynb`:

https://drive.google.com/drive/folders/1mcsqBfSqyEtD8iu5G0l8poeYR7oaitLl?usp=sharing

The file is `runs/HooshvareLab--bert-fa-base-uncased-sentiment-digikala/test_metrics.json`. The name says test, and the JSON says `"split": "labeled"` and `"rows": 5258342`. A copy of that file and the matching log are under `artifacts/colab/tookabert/`. The link is also in `artifacts/colab/drive-links.json`. No weight file was saved for this run.

| Model | Rows | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `HooshvareLab/bert-fa-base-uncased-sentiment-digikala` | 5,258,342 | 0.7403 | 0.5668 | 0.6544 | 0.5918 |

Per-class F1 in that file: `recommended` 0.8558, `not_recommended` 0.6052, `no_idea` 0.3144. The model card's accuracy 0.8268 and F1 0.8172 are on the 62,321-comment set's own test split, not on these 5,258,342 reviews.

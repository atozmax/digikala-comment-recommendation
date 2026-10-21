**(1405-07-30)** · Thursday, 22 October 2026

# Descriptive analysis: schema and missing values

This note records the first full-file audit in `notebooks/01-descriptive-analysis.ipynb`. Every count below comes from that notebook: one pandas parse of `data/raw/digikala-comments.csv`, then `DataFrame.info()` and `DataFrame.isnull().sum()`. No row was sampled.

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

A bare hyphen is therefore missing, not a category. `created_at` is still a string, and `rate` is `float64`. Neither column has been range-checked in this notebook.

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

**The recommendation task has a large labeled set and a natural unlabeled pool.** `recommendation_status` is missing on 894,426 rows (14.53%). Those rows are not supervised examples. The labeled pool is 5,261,863 rows (85.47%). This notebook has not yet counted the classes inside that pool, so imbalance is not claimed here. The missing labels should stay missing: they are not a class to impute.

**Rating bands are not blocked by null rates.** `rate` has no missing values, so every row can be tested against the band edges (`low` below 2.5, `medium` from 2.5 up to but not including 4, `high` from 4 through 5). A complete column is not the same as a valid one. Minimum, maximum, and out-of-range rates are not in this notebook.

**Pros, cons, and size-fit are sparse side text.** `advantages` is filled on 11.50% of rows and `disadvantages` on 6.70%. Shoppers who use these fields write a pro more often than a con. `true_to_size_rate` is filled on 1.47% of rows, which fits a size-fit score that applies only to some products. All three columns are outside the model inputs, so their sparsity does not change the feature set.

**Seller fields fail together in the counts.** `seller_title` and `seller_code` each have 302,181 missing values (4.91%). Equal counts are consistent with one seller block being absent on the same rows. A row-level cross-tab is still required before treating that as a fact.

**The frame is already heavy.** The CSV is 1.28 GB on disk and about 1.6 GB once loaded with these dtypes. Schema and null counts fit in one pass. Later full-file work (label vocabulary, text length, duplicate ids, reviews per product) should stay on the full 6,156,289 rows, using chunks or a columnar engine if a second in-memory copy becomes tight.

## 5. Decisions this pass supports

1. Quote **6,156,289 × 15** as the audited size, and keep the 1.28 GB file size next to it.
2. Drop the **637** rows with an empty `body` from every text model. Do not backfill them from `title`, `advantages`, or `disadvantages`.
3. Train the recommendation head only on the **5,261,863** rows with a `recommendation_status`. Hold the **894,426** unlabeled rows out of supervised loss. Do not fill that label.
4. Derive `rating_band` from `rate` on the full file, then reject non-numeric or out-of-range values in the next audit. Null rate is not the failure mode.
5. Keep `product_id` for grouped splitting. It is complete, which removes null-product rows as a split problem. Uniqueness and reviews-per-product are still unmeasured.
6. Leave `title`, seller fields, likes, dislikes, advantages, disadvantages, and `true_to_size_rate` out of the feature vector, matching the input policy. Their missingness is documented so it is not silently “fixed.”

## 6. Not measured in this notebook

The following Part A items are still open. They are absent from `01-descriptive-analysis.ipynb`, so this report does not estimate them:

- unique products and sellers, and the distribution of reviews per product
- duplicate `id`s, duplicate bodies, and identical text with conflicting labels
- the vocabulary of `recommendation_status`, and counts for `recommended`, `not_recommended`, and `no_idea`
- the distribution of `rate` and of the three rating bands
- character and word length of `body`, including very short and very long reviews
- whether missing `recommendation_status` depends on `is_buyer`, `rate`, or product frequency

Those measurements are the next descriptive pass. Class weights, `max_length`, and the grouped split should wait for them.

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

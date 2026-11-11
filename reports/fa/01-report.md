**۲۰ آبان ۱۴۰۵** (1405-08-20) · چهارشنبه، ۱۱ نوامبر ۲۰۲۶

# آموزش بعدی از کجا شروع می‌شود

اجرای آموزش بعدی از وزن‌های ذخیره‌شده روی گوگل درایو شروع می‌شود. از چک‌پوینت اصلی هاگینگ‌فیس شروع نمی‌شود، و گام بهینه‌ساز 115,027 را هم ادامه نمی‌دهد. این پوشه‌های درایو `model.safetensors` و توکنایزر را دارند. حالت بهینه‌ساز را ندارند، پس دور بعدی یک دور تازه است که از همین وزن‌ها آغاز می‌شود.

فایل‌های JSON کوچک همین پوشه‌ها زیر `artifacts/colab/` هستند. فایل وزن روی درایو مانده است. بخش ۱۲ در `reports/fa/00-report.md` سنجه‌ها را ثبت کرده است. این یادداشت فقط فایل‌هایی را ثبت می‌کند که اجرای بعدی بار می‌کند.

## وزن شروع

هر مدل تمام‌شده سه فایل `model.safetensors` دارد: `runs/...--head`، `runs/...--last_block`، و `best/...`. نسخه `best/` همان اجرای بلوک آخر است، و آموزش همان پوشه را بار می‌کند.

| مدل | پوشه درایو | پوشه شروع |
| --- | --- | --- |
| ParsBERT، `HooshvareLab/bert-fa-base-uncased` | https://drive.google.com/drive/folders/1GKqsgeXT8uFFLYddJDnA9509bIiDHosd?usp=sharing | `best/HooshvareLab--bert-fa-base-uncased` |
| XLM-RoBERTa، `FacebookAI/xlm-roberta-base` | https://drive.google.com/drive/folders/1AEtn221vAuEW3yAzgQ6-JSeJd0-keHpm?usp=sharing | `best/FacebookAI--xlm-roberta-base` |
| TookaBERT، `PartAI/TookaBERT-Base` | https://drive.google.com/drive/folders/1mcsqBfSqyEtD8iu5G0l8poeYR7oaitLl?usp=sharing | `best/PartAI--TookaBERT-Base` |

هر یک از این پوشه‌ها `config.json`، `metrics.json`، `model.safetensors`، `tokenizer.json`، و `tokenizer_config.json` دارد. پوشه `runs/<name>--last_block` همان اجراست، پیش از کپی شدن به `best/`. پوشه `runs/<name>--head` اجرای قبلی با رمزگذار منجمد است. ادامه از `best/` بار می‌شود، نه از `head/`.

`notebooks/colab/04-shiraz-finetune.ipynb` فایل وزن ندارد. `lifeweb-ai/shiraz` نقطه شروع نیست تا وقتی یک دور سر ذخیره شود.

## آنچه قبلاً آموزش دیده است

وزن بارشده از قبل یک طبقه‌بند آموزش‌دیده و بلوک بالای رمزگذار، `layers[-1]`، را دارد. بلوک‌های پایین‌تر هنوز رمزگذار پیش‌آموزش‌دیده‌اند. اجرای بعدی همین وزن‌ها را نگه می‌دارد و چهار بلوک بالا، `layers[-4:]`، را باز می‌کند، با نرخ یادگیری `2e-5` روی بلوک‌ها و `1e-4` روی طبقه‌بند. هشت بلوک پایین منجمد می‌مانند.

اعتبارسنجی وزن‌هایی که بار می‌شوند:

| مدل | دقت | Macro F1 | F1 برای recommended | F1 برای not_recommended | F1 برای no_idea |
| --- | ---: | ---: | ---: | ---: | ---: |
| ParsBERT | 0.8004 | 0.6687 | 0.8947 | 0.6902 | 0.4211 |
| XLM-RoBERTa | 0.7989 | 0.6630 | 0.8943 | 0.6824 | 0.4123 |
| TookaBERT | 0.8053 | 0.6722 | 0.8983 | 0.6969 | 0.4214 |

اولین ادامه TookaBERT است. بین سه مدل ذخیره‌شده بالاترین دقت اعتبارسنجی و بالاترین macro F1 اعتبارسنجی را دارد. ParsBERT و XLM-RoBERTa در مسیرهای بالا برای همین نوع اجرا می‌مانند. پوشه `best/` برای TookaBERT هنوز `test_metrics.json` بلوک آخر را ندارد؛ فایل آزمون آن سنجه سر قبلی است.

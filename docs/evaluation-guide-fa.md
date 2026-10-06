# راهنمای اجرای ارزیابی نهایی PPQ

این راهنما مسیر مرجع اجرای بنچمارک ۲۰۰ موردی، مقایسه با Gemma پایه، ارزیابی
کور، سنجش زمان و پروفایل اجزای سامانه را توضیح می‌دهد. فایل‌های ثبت‌شده در
`results/` ثابت هستند؛ اجرای مجدد به‌صورت پیش‌فرض در `evaluation-output/`
ذخیره می‌شود تا نتایج نهایی قبلی بازنویسی نشوند.

## اصل علمی

بنچمارک مرجع در `data/benchmark/benchmark.jsonl` شامل ۲۰۰ پرامپت است و هش
SHA-256 آن باید برابر مقدار زیر باشد:

```text
9a9af819aa020ee71f4ed1b62151972315dda67e1a63ff3ecc9e48211308ffc8
```

پس از مشاهده نتایج این مجموعه، تغییر promptها، thresholdها، بودجه verifier،
پیکره یا سیاست بازیابی و گزارش دوباره همان ۲۰۰ مورد به‌عنوان آزمون untouched
مجاز نیست. هر بهبود جدید باید با داده مستقل ارزیابی شود یا در بخش کارهای آینده
قرار گیرد.

## محیط پیشنهادی

- Python 3.12
- دو GPU مدل Tesla T4 با حافظه ۱۶ گیگابایت
- اینترنت فعال و دسترسی Hugging Face به `google/gemma-4-E4B-it`
- همان revisionهای ثبت‌شده در `README.md`

نصب کامل وابستگی‌های اجرا، آزمون و نمودار:

```bash
python -m pip install -e ".[test,evaluation]"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## ۱. بررسی اولیه

از ریشه پروژه اجرا کنید:

```bash
python -m scripts.evaluation.preflight
```

این فرمان مسیر و هش بنچمارک، وجود پیکره، بردارها، بسته runtime، رابط وب و
اطلاعات محیط را بررسی می‌کند. نبود CUDA برای کنترل فایل‌ها خطا نیست، اما اجرای
واقعی مدل باید در محیط GPU انجام شود.

## ۲. ارزیابی اصلی کیفیت

برای اجرای PPQ و Gemma پایه روی بنچمارک مهروموم‌شده:

```bash
python -m scripts.evaluation.run_benchmark \
  --embedding-device cuda:0 \
  --seed 42
```

خروجی پیش‌فرض:

```text
evaluation-output/benchmark/
├── outputs.jsonl
├── summary.json
├── blind_review.csv
└── blind_key.jsonl
```

`blind_review.csv` برای ارزیاب قابل مشاهده است. فایل `blind_key.jsonl` تا پایان
ارزیابی نباید باز شود. برای اجرای آزمایشی کوتاه می‌توان `--limit N` را به کار
برد، اما چنین اجرايی نتیجه نهایی محسوب نمی‌شود.

پس از تکمیل ستون‌های ارزیابی کور:

```bash
python -m scripts.evaluation.summarize_blind_review
```

خروجی در `evaluation-output/benchmark/blind_evaluation.json` ذخیره می‌شود.

برای بازتولید صرفاً محاسبات آماری از فایل‌های نهایی موجود، بدون اجرای مدل:

```bash
python -m scripts.evaluation.summarize_blind_review \
  --review results/benchmark/blind_review.csv \
  --key results/benchmark/blind_key.jsonl \
  --output evaluation-output/recomputed-blind-evaluation.json
```

## ۳. سنجش زوجی زمان اجرا

این اجرا برای هر پرامپت ترتیب Base و PPQ را با seed ثابت تصادفی می‌کند و هر دو
شرط از همان مدل بارگذاری‌شده Gemma استفاده می‌کنند:

```bash
python -m scripts.evaluation.measure_latency \
  --device-map balanced \
  --attention sdpa \
  --embedding-device cuda:0 \
  --embedding-batch-size 32 \
  --max-new-tokens 256 \
  --warmup 3 \
  --seed 42
```

اگر فقط BGE-M3 دچار کمبود حافظه شد، می‌توان از گزینه زیر استفاده کرد:

```bash
--embedding-device cpu
```

این گزینه Gemma را به CPU منتقل نمی‌کند. خروجی پیش‌فرض:

```text
evaluation-output/performance/
├── base_predictions_timed.jsonl
├── ppq_predictions_timed.jsonl
├── condition_order.json
└── run_metadata.json
```

اگر اجرای Kaggle قطع شد، همان فرمان را با `--resume` ادامه دهید. برای جلوگیری
از مخلوط شدن نتایج، پیکربندی resume باید با اجرای قبلی یکسان باشد.

## ۴. تحلیل زمان و سربار

پس از کامل شدن هر دو فایل زمان‌دار:

```bash
python -m scripts.evaluation.summarize_latency
```

خروجی‌های اصلی عبارت‌اند از:

```text
evaluation-output/performance/
├── final_summary.json
├── paired_cases.jsonl
├── final_200_latency.csv
├── latency_by_category.csv
├── latency_by_action.csv
└── DEFENSE_SUMMARY.md
```

تعریف شاخص‌ها:

```text
absolute_overhead = T_PPQ - T_BASE
relative_overhead_pct = (T_PPQ - T_BASE) / T_BASE * 100
slowdown_ratio = T_PPQ / T_BASE
```

برای گزارش، median و P95 هر دو شرط، median اختلاف زوجی، نسبت کندی و بازه
اطمینان bootstrap را کنار معیارهای کیفیت ارائه کنید.

## ۵. پروفایل داخلی PPQ

پروفایل داخلی جدا از latency مرجع اجرا می‌شود تا instrumentation نتیجه اصلی را
تغییر ندهد. برای زیرمجموعه متوازن ۴۰تایی:

```bash
python -m scripts.evaluation.profile_stages --n 40
```

برای پروفایل همه موارد از `--n 200` استفاده کنید. فایل خروجی
`evaluation-output/performance/ppq_stage_profile.jsonl` است. سپس analyzer زمان
را دوباره اجرا کنید تا خلاصه مرحله‌ها نیز وارد `final_summary.json` شود.

## ۶. پیوند کیفیت و زمان

پس از تولید `paired_cases.jsonl` می‌توان ارزیابی کور نهایی موجود را با latency
زوجی پیوند داد:

```bash
python -m scripts.evaluation.analyze_quality_latency
```

خروجی‌ها:

```text
evaluation-output/performance/quality_latency_tradeoff.json
evaluation-output/performance/quality_latency_tradeoff_cases.csv
```

## ۷. تولید نمودارها

```bash
python -m scripts.evaluation.generate_figures
```

نمودارها در `evaluation-output/performance/figures/` ساخته می‌شوند و شامل
scatter زمان زوجی، کندی بر اساس دسته، توزیع زمان بر اساس عملیات، مقایسه کیفیت
کور و در صورت وجود داده پروفایل، زمان مراحل داخلی هستند.

## ۸. آزمون‌های تحویل

```bash
python -m pytest -q
```

آزمون‌ها ساختار پیکره، تعداد رکوردها و variantها، سازگاری منابع، hash و توزیع
بنچمارک، بردارهای BGE-M3، بازتولید نتایج کور، مسیرهای ابزار ارزیابی، escape شدن
HTML و تنظیمات RTL رابط را کنترل می‌کنند.

## تفسیر مناسب برای دفاع

PPQ مدل زبانی پایه را با مدل دیگری جایگزین نمی‌کند؛ همان Gemma را داخل یک
معماری corpus-bound قرار می‌دهد. بنابراین نتیجه باید هم بهبود اعتمادپذیری و هم
هزینه محاسباتی را نشان دهد. ادعای اصلی کاهش fabrication، افزایش authenticity
و appropriateness و امکان ردیابی پاسخ است. برتری همیشگی زمانی ادعا نمی‌شود،
زیرا توزیع latency PPQ دنباله سنگین‌تری دارد.

# 🧠 Persian BERT (HuggingFace-based)

این پروژه نسخه‌ای بازنویسی‌شده از BERT برای زبان فارسی است که به‌جای TensorFlow 1.x از کتابخانه‌ی **HuggingFace Transformers** و **PyTorch** استفاده می‌کند.  
کدها به‌صورت کامل بازسازی شده‌اند تا بتوان با داده‌های جدید فارسی مانند **MirasText** و **Wikipedia فارسی** مدل BERT را از صفر پیش‌تمرین (pretrain) کرد.

---

## ⚙️ پیش‌نیازها

### ✅ نسخه‌ی پیشنهادی پایتون
این پروژه در محیط **Python 3.12.3** تست و توسعه داده شده است.  
نسخه‌های بالاتر (به‌ویژه **3.13** و **3.14**) ممکن است با برخی پکیج‌ها مانند `torch` یا `transformers` ناسازگار باشند (به‌خصوص در هنگام نصب با pip).

> ⚠️ **توصیه:**  
> اگر از نسخه‌های جدیدتر استفاده می‌کنید و با خطای نصب مواجه شدید، یکی از دو روش زیر را انجام دهید:

**روش ۱️⃣: استفاده از نسخه‌ی پیشنهادی پایتون (پیشنهادی‌ترین گزینه)**  
در لینوکس یا مک:
```bash
pyenv install 3.12.3
pyenv local 3.12.3
```
در ویندوز می‌توانید از python.org نسخه‌ی 3.12.3 را دانلود کنید.
روش ۲️⃣: رفع ناسازگاری در نسخه‌های بالاتر
اگر پایتون شما 3.13 یا 3.14 است، قبل از نصب بسته‌ها دستور زیر را اجرا کنید تا نسخه‌ی سازگار PyTorch نصب شود:
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.46.1
```
در صورت وجود GPU از آدرس پکیج‌های PyTorch نسخه‌ی مناسب CUDA را انتخاب کنید.

---

## 📦 نصب و اجرا

### 1️⃣ دریافت ریپازیتوری
```bash
git clone https://github.com/mohammadsaleh40/bert_hf.git
cd bert_hf
```
2️⃣ ایجاد محیط مجازی و نصب وابستگی‌ها

```bash
python3 -m venv venv
source venv/bin/activate       # در ویندوز: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
---
⚙️ آماده‌سازی داده‌ها
الف) دریافت داده‌ی نمونه MirasText
```bash
wget https://raw.githubusercontent.com/miras-tech/MirasText/refs/heads/master/MirasText/MirasText_sample.txt -O MirasText_sample.txt
```
ب) پیش‌پردازش داده‌ها
```bash
python prepare_mirastext.py
```
📄 خروجی: `mirastext_preprocessed.txt`

---
ج) افزودن Wikipedia فارسی
1. دانلود فایل فشرده ویکی‌پدیا فارسی:
```bash
wget https://dumps.wikimedia.org/fawiki/latest/fawiki-latest-pages-articles.xml.bz2
```
2. استخراج محتوای متنی با WikiExtractor:
```bash
python -m wikiextractor.WikiExtractor fawiki-latest-pages-articles.xml.bz2 -o fawiki-latest-pages-articles
```
3. اضافه کردن مقالات ویکی‌پدیا به انتهای داده‌ی MirasText:
```bash
python add_wiki_to_preprocessed.py
```
📄 خروجی نهایی: `mirastext_preprocessed.txt` شامل MirasText + Wikipedia فارسی
---
🧰 ساخت واژگان (اختیاری)

در صورت تمایل می‌توانید واژگان جدید بسازید:
```bash
python create_vocab.py
```
فایل تولیدی نامش باید به `vocab.txt` تغییر پیدا کند. با دستور زیر آن را تغییر می‌دهیم.

```bash
mv persian_bert_tokenizer/wp-vocab.txt persian_bert_tokenizer/vocab.txt
```
---
🚀 آموزش مدل BERT فارسی

فایل run_pretraining_hf_v2.py مسئول اجرای آموزش مدل بر پایه‌ی HuggingFace Trainer است.
پارامترهای اصلی درون فایل تعریف شده‌اند (مثل اندازه‌ی مدل، توکنایزر، مسیر داده‌ها و غیره).
```bash
python run_pretraining_hf_v2.py
```
📂 خروجی مدل ذخیره می‌شود در مسیر:
```bash
persian_bert_tiny_output_large_2/
```
🔍 بررسی و تست مدل

برای آزمایش مدل آموزش‌دیده، دو روش در دسترس است:

🔹 روش ۱: اجرای مستقیم اسکریپت
```bash
python check_model.py
```
این فایل چند جمله‌ی فارسی را پردازش کرده و با استفاده از t-SNE توزیع بردارهای کلمات را نمایش می‌دهد.
---

🔹 روش ۲: استفاده از نوت‌بوک

فایل chek_model.ipynb را با Jupyter باز کنید:
```bash
jupyter notebook chek_model.ipynb
```
در این نوت‌بوک:

مدل از مسیر `persian_bert_tiny_final_model_large_2` بارگذاری می‌شود.

چند جمله‌ی فارسی نمونه به مدل داده می‌شود.

و خروجی‌ها (embedding و شباهت‌ها) بررسی می‌شوند.

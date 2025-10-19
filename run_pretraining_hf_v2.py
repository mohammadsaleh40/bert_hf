# run_pretraining_hf_v2.py
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
import os

# --- 1. بارگذاری و پردازش داده با استفاده از load_dataset و streaming ---
print("Loading and processing data with streaming...")

# فرض بر این است که فایل mirastext_preprocessed.txt که شامل هم MirasText و هم Wikipedia است، در مسیر زیر است
text_file_path = "mirastext_preprocessed.txt"

# ایجاد یک Dataset از فایل متنی
# load_dataset می‌تواند فایل‌های متنی را به صورت خط به خط بخواند
# این روش کارآمدتر است، خصوصاً برای فایل‌های بزرگ
raw_dataset = load_dataset("text", data_files=text_file_path, split="train")

print(f"Raw dataset size: {len(raw_dataset)}")

# تقسیم دیتاست (اگر نیاز باشد، اما ممکن است برای streaming سخت باشد)
# یک روش ساده برای تقسیم: استفاده از select برای یک نمونه کوچک در ابتدا یا استفاده از شاخص‌ها
# برای سادگی، اینجا کل دیتاست را به عنوان train در نظر می‌گیریم و یک زیرمجموعه کوچک برای eval ایجاد می‌کنیم
# eval_dataset = raw_dataset.select(range(0, int(0.05 * len(raw_dataset)))) # 5% اول را ارزیابی فرض می‌کنیم
# train_dataset = raw_dataset.select(range(int(0.05 * len(raw_dataset)), len(raw_dataset)))

# یا استفاده از train_test_split اگر بتواند با streaming کار کند (ممکن است کل دیتاست را بار کند)
# dataset_split = raw_dataset.train_test_split(test_size=0.05)
# train_dataset = dataset_split['train']
# eval_dataset = dataset_split['test']

# روش ترجیحی برای streaming: استفاده از شاخص‌ها مستقیماً در Trainer
# یا ایجاد دو شیء جداگانه از همان load_dataset
# مثلاً اگر بدانیم اول فایل MirasText و بعد Wikipedia است، می‌توانیم تخمین بزنیم یا جدا کنیم
# اما در اینجا، فرض می‌کنیم کل فایل یک مجموعه داده است و مستقیماً از آن استفاده می‌کنیم.
# برای streaming، ممکن است نیاز باشد DatasetDict را به صورت غیر-سنتی ایجاد کنیم یا فقط از split استفاده کنیم
# اما Trainer معمولاً از DatasetDict استفاده می‌کند.

# یک روش: تقسیم شاخص‌ها در پایتون و سپس استفاده از select
total_lines = len(raw_dataset)
eval_size = int(0.05 * total_lines) # 5% برای ارزیابی
eval_indices = list(range(eval_size))
train_indices = list(range(eval_size, total_lines))

max_train_samples = 500_000
train_dataset = raw_dataset.select(range(eval_size, min(eval_size + max_train_samples, total_lines)))
eval_dataset = raw_dataset.select(eval_indices)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- 2. بارگذاری Tokenizer ---
print("Loading tokenizer...")
tokenizer = BertTokenizerFast(vocab_file="persian_bert_tokenizer/vocab.txt", do_lower_case=False)

# --- 3. تابع Tokenization ---
def tokenize_function(examples):
    # examples الان شامل یک کلید "text" است که لیستی از خطوط متن است
    # فیلتر خطوط خالی
    texts = [text for text in examples["text"] if text.strip() != ""]
    if not texts:
        return {"input_ids": []}
    # Tokenization با max_length و truncation
    # این عملیات الان روی بچ‌های کوچک از دیتاست انجام می‌شود (batched=True)
    # این کار را بسیار کارآمدتر می‌کند
    return tokenizer(texts, padding=False, truncation=True, max_length=512)

# --- 4. اعمال Tokenization (همچنان ممکن است زمان‌بر باشد، اما به صورت batch انجام می‌شود) ---
print("Tokenizing datasets...")
# حذف ردیف‌هایی که متن آنها خالی است قبل از tokenization (این قدم از قبل در tokenize_function انجام می‌شود)
# map روی کل دیتاست اعمال می‌شود
# برای داده‌های بسیار بزرگ، ممکن است بخواهید نتیجه map را cache کنید یا از num_proc برای پردازش موازی استفاده کنید
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4) # num_proc: پردازش موازی با 4 فرآیند
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=4)

# --- 5. بارگذاری مدل ---
print("Loading model...")
config = BertConfig.from_json_file("bert_config.json") # مطمئن شوید vocab_size درست است
model = BertForMaskedLM(config)

# --- 6. تعریف Data Collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# --- 7. تعریف آرگومان‌های آموزش ---
training_args = TrainingArguments(
    output_dir="./persian_bert_tiny_output_large_2",
    overwrite_output_dir=True,
    gradient_accumulation_steps = 2,
    num_train_epochs=1, # ممکن است بخواهید این را کاهش دهید یا gradient_accumulation_steps افزایش دهید
    per_device_train_batch_size=2, # ممکن است نیاز به کاهش باشد بسته به GPU
    per_device_eval_batch_size=2,
    prediction_loss_only=True,
    save_steps=50000, # کمی کمتر از قبل، چون داده بیشتر است
    save_total_limit=2,
    # evaluation_strategy="steps", # <-- حذف شد
    eval_steps=1500, # هر 25000 گام یک بار ارزیابی
    # max_steps = 100,
    logging_steps=5000, # هر 5000 گام لاگ زده شود
    learning_rate=5e-5,
    warmup_steps=5000, # افزایش داده شد
    logging_dir='./logs_large',
    fp16=True, # فعال کردن fp16 برای کاهش مصرف حافظه (اگر GPU پشتیبانی کند)
    # report_to=None, # غیرفعال کردن گزارش به wandb یا ...
    dataloader_num_workers=3, # استفاده از چند thread برای بارگذاری داده
)

# --- 8. ایجاد Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# --- 9. شروع آموزش ---
print("Starting training...")
trainer.train()

# --- 10. ذخیره مدل نهایی ---
trainer.save_model("./persian_bert_tiny_final_model_large_2")
tokenizer.save_pretrained("./persian_bert_tiny_final_model_large_2")
print("Training finished. Model saved to ./persian_bert_tiny_final_model_large_2")
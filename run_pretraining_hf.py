# run_pretraining_hf.py
from datasets import Dataset, DatasetDict
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
import os

# --- 1. بارگذاری و پردازش داده ---
print("Loading and processing data...")
text_file_path = "mirastext_preprocessed.txt"

with open(text_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    texts = [line.strip() for line in lines]

documents = []
current_doc = []
for line in texts:
    if line == "":
        if current_doc:
            documents.append(" ".join(current_doc))
            current_doc = []
    else:
        current_doc.append(line)

if current_doc:
    documents.append(" ".join(current_doc))

dataset = Dataset.from_dict({"text": documents})
dataset_split = dataset.train_test_split(test_size=0.05) # 5% تست
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test'] # استفاده از 'test' به عنوان 'eval'
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- 2. بارگذاری Tokenizer ---
print("Loading tokenizer...")
tokenizer = BertTokenizerFast(vocab_file="persian_bert_tokenizer/vocab.txt", do_lower_case=False)

# --- 3. تابع Tokenization ---
def tokenize_function(examples):
    # اطمینان از اینکه فقط رشته‌های غیر خالی پردازش شوند
    texts = [text for text in examples["text"] if text.strip() != ""]
    if not texts:
        # بسته به پیاده‌سازی trainer، ممکن است نیاز به بازگرداندن یک دیکشنری خالی یا داده با طول 0 باشد
        # برای اطمینان، می‌توانید این خطوط را بررسی کنید
        # اما معمولاً filter قبل از این مرحله این موارد را حذف می‌کند
        return {"input_ids": []} # یا نحوه دیگری برای مدیریت داده‌های خالی
    # اضافه کردن max_length و truncation
    # max_length باید کمتر یا مساوی max_position_embeddings (512) باشد
    return tokenizer(texts, padding=False, truncation=True, max_length=512) # <-- اضافه شد

# --- 4. اعمال Tokenization ---
print("Tokenizing datasets...")
# حذف ردیف‌هایی که متن آنها خالی است قبل از tokenization
train_dataset = train_dataset.filter(lambda example: example["text"].strip() != "")
eval_dataset = eval_dataset.filter(lambda example: example["text"].strip() != "")

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- 5. بارگذاری مدل ---
print("Loading model...")
# بارگذاری کانفیگ با اندازه vocab صحیح (مثلاً 30000 اگر vocab.txt شما همین است)
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
    output_dir="./persian_bert_tiny_output", # پوشه خروجی
    overwrite_output_dir=True, # بازنویسی خروجی قبلی
    num_train_epochs=3, # تعداد ایپاک‌ها
    per_device_train_batch_size=16, # اندازه دسته در هر دستگاه
    per_device_eval_batch_size=16,
    prediction_loss_only=True, # فقط زیان پیش‌بینی برای بهینه‌سازی استفاده شود (MLM)
    save_steps=10_000, # هر 10000 گام چک‌پوینت ذخیره شود
    save_total_limit=2, # حداکثر 2 چک‌پوینت نگه داشته شود
    # evaluation_strategy="steps", # ارزیابی هر چند گام یک بار انجام شود
    eval_steps=5_000, # هر 5000 گام یک بار ارزیابی انجام شود
    logging_steps=1_000, # هر 1000 گام لاگ زده شود
    learning_rate=5e-5, # نرخ یادگیری
    warmup_steps=1_000, # گام‌های گرم‌کردن
    logging_dir='./logs', # پوشه لاگ
    # fp16=True, # فعال کردن fp16 برای کاهش مصرف حافظه (اگر GPU پشتیبانی کند)
    # report_to=None, # غیرفعال کردن گزارش به wandb یا ... اگر نمی‌خواهید
)

# --- 8. ایجاد Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer, # اگرچه tokenizer در data_collator استفاده می‌شود، اما برای ذخیره در مدل نیز مفید است
)

# --- 9. شروع آموزش ---
print("Starting training...")
trainer.train()

# --- 10. ذخیره مدل نهایی ---
trainer.save_model("./persian_bert_tiny_final_model")
tokenizer.save_pretrained("./persian_bert_tiny_final_model") # ذخیره tokenizer نیز
print("Training finished. Model saved to ./persian_bert_tiny_final_model")

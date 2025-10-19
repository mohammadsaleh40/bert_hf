# inference_example.py

from transformers import AutoTokenizer, AutoModel
import torch

# 1. بارگذاری توکنایزر و مدل از دایرکتوری ذخیره شده
model_path = "./persian_bert_tiny_final_model"

# بارگذاری توکنایزر
tokenizer = AutoTokenizer.from_pretrained(model_path)

# بارگذاری مدل (AutoModel برای گرفتن خروجی پنهان استفاده می‌شود)
# اگر مدل پیش‌آموزش داده شده شما فقط برای Masked Language Modeling (MLM) است، ممکن است بخواهید
# BertForMaskedLM را بارگذاری کنید تا بتوانید خروجی احتمال برای هر توکن را بگیرید.
# اما برای گرفتن embedding کلی، AutoModel یا BertModel کافی است.
model = AutoModel.from_pretrained(model_path)

# چک کردن وجود دستگاه (CPU یا GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device) # انتقال مدل به دستگاه (اگر GPU موجود باشد)

# 2. متن ورودی نمونه
text = "این یک متن فارسی برای تست مدل است."

# 3. توکنایز کردن متن
# اطمینان از اینکه batch_size > 1 نیست یا ابعاد به درستی مدیریت می‌شوند
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) # max_length مطابق کانفیگ

# انتقال ورودی‌ها به دستگاه
inputs = {key: value.to(device) for key, value in inputs.items()}

# 4. غیر فعال کردن محاسبات گرادیان (برای inference سریع‌تر و کم‌مصرف‌تر)
with torch.no_grad():
    # اجرای مدل
    outputs = model(**inputs)

    # 5. دریافت خروجی‌ها
    # last_hidden_states: embeddingهای آخرین لایه برای هر توکن
    last_hidden_states = outputs.last_hidden_state
    # pooled_output: خروجی ترکیب شده (مثلاً برای classification)، اگر مدل از pooling پشتیبانی کند
    # pooled_output = outputs.pooler_output # ممکن است برای مدل پیش‌آموزش (نه classification) موجود نباشد یا مفید نباشد

# 6. چاپ خروجی
print("\nInput text:", text)
print("\nInput IDs (Token IDs):", inputs["input_ids"].cpu().numpy())
print("\nAttention Mask:", inputs["attention_mask"].cpu().numpy())
print("\nLast Hidden States Shape (batch_size, sequence_length, hidden_size):", last_hidden_states.shape)
print("\nLast Hidden States (first few tokens of first batch) - First 5 dimensions:\n", last_hidden_states[0, :, :5].cpu().numpy())

# اگر مدل BertForMaskedLM بود و می‌خواستید خروجی MLM را ببینید:
# from transformers import BertForMaskedLM
# model_mlm = BertForMaskedLM.from_pretrained(model_path)
# model_mlm.to(device)
# outputs_mlm = model_mlm(**inputs)
# prediction_scores = outputs_mlm.logits # احتمالات برای هر توکن در واژه‌نامه
# print("\nMLM Prediction Scores Shape:", prediction_scores.shape)

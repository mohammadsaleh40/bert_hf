# load_model.py
from transformers import BertConfig, BertForMaskedLM

# فرض بر این است که bert_config.json در مسیر فعلی ذخیره شده است
config = BertConfig.from_json_file("bert_config.json")

# ایجاد مدل جدید با کانفیگ تعیین شده (شروع به کار با وزن‌های تصادفی)
model = BertForMaskedLM(config)

print(model)
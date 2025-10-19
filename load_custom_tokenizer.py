# load_custom_tokenizer.py
from transformers import BertTokenizerFast # یا BertTokenizer اگر مدل BertTokenizer ایجاد کرده باشید

# بارگذاری Tokenizer با استفاده از فایل vocab.txt ایجاد شده در مرحله قبل
# فرض بر این است که wp-vocab.txt را به vocab.txt تغییر نام داده‌اید
tokenizer = BertTokenizerFast(vocab_file="persian_bert_tokenizer/vocab.txt", do_lower_case=False) # do_lower_case را مطابق تنظیمات ایجاد واژه تنظیم کنید

# می‌توانید یک تست سریع انجام دهید:
test_text = "این یک متن فارسی است."
tokens = tokenizer.tokenize(test_text)
print("Tokens:", tokens)
encoded = tokenizer.encode(test_text)
print("Encoded IDs:", encoded)
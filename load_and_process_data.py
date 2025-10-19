# load_and_process_data.py
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast

# فرض بر این است که فایل mirastext_preprocessed.txt در مسیر فعلی است
text_file_path = "mirastext_preprocessed.txt"

# خواندن خطوط فایل
with open(text_file_path, 'r', encoding='utf-8') as f:
    # خطوط خالی را حذف می‌کنیم، اما جدا کننده اسناد (خط خالی) باید حفظ شود.
    # اگر خطوط خالی در پیش‌پردازش شما باقی مانده بودند، باید آنها را نگه داریم.
    # روش زیر خطوط خالی را نگه می‌دارد.
    lines = f.readlines()
    # حذف \n از انتهای هر خط، اما خطوط خالی را همچنان خالی نگه می‌دارد
    texts = [line.strip() for line in lines]

# جدا کردن اسناد (هر بلوک از خطوط غیر خالی یک سند است)
documents = []
current_doc = []
for line in texts:
    if line == "": # خط خالی نشان‌دهنده پایان یک سند و شروع سند جدید است
        if current_doc: # اگر سند فعلی خالی نبود
            documents.append(" ".join(current_doc)) # ادغام خطوط سند با فاصله
            current_doc = [] # شروع سند جدید
    else:
        current_doc.append(line) # افزودن خط به سند فعلی

# اضافه کردن آخرین سند اگر موجود باشد
if current_doc:
    documents.append(" ".join(current_doc))

print(f"Number of documents loaded: {len(documents)}")
# print("First document snippet:", documents[0][:100]) # چاپ بخشی از اولین سند برای بررسی

# ایجاد دیتاست Hugging Face
dataset = Dataset.from_dict({"text": documents})

# تقسیم دیتاست (اختیاری، اما توصیه می‌شود)
dataset_split = dataset.train_test_split(test_size=0.05) # 95% آموزش، 5% ارزیابی

print(dataset_split)
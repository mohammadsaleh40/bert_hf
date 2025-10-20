# create_vocab.py (فایلی که باید ایجاد یا بازیابی کنید)
from tokenizers import BertWordPieceTokenizer

# مسیر فایل متنی پیش‌پردازش شده
files = ["mirastext_preprocessed.txt"]

# ایجاد tokenizer
tokenizer = BertWordPieceTokenizer(lowercase=False) # Persian is not cased

# آموزش tokenizer
tokenizer.train(files=files, vocab_size=32000, min_frequency=2, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

# ذخیره tokenizer و فایل vocab.txt
tokenizer.save_model(directory="persian_bert_tokenizer", prefix="wp")
print("Tokenizer saved to 'persian_bert_tokenizer/' directory.")
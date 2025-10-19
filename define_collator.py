# define_collator.py
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
# بارگذاری Tokenizer
tokenizer = BertTokenizerFast(vocab_file="persian_bert_tokenizer/vocab.txt", do_lower_case=False)

# ایجاد Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
        mlm=True, # فعال کردن Masked Language Modeling
            mlm_probability=0.15 # احتمال ماسک کردن یک توکن
            )
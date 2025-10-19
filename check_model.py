#‌ %%

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# بارگذاری مدل و توکنایزر
model_path = "./persian_bert_tiny_final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

# لیست کلمات یا جملات نمونه
sentences = [
    "این یک متن فارسی برای تست مدل است.",
    "این یک متن فارسی برای آزمایش مدل است.",
    "آسمان آبی بسیار زیبا است.",
    "دریا نیز زیبایی خاص خود را دارد.",
    "مدل زبانی بسیار پیشرفته است.",
    "سیستم هوشمند کاربرد زیادی دارد."
]
words_to_visualize = ["تست", "آزمایش", "زیبا", "آسمان", "دریا", "مدل", "سیستم"] # تعداد کلمات بیشتر از تعداد جملات

embeddings = []
labels = []

for sent in sentences:
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    for word in words_to_visualize:
        word_tokens = tokenizer.tokenize(word)
        if word_tokens and word_tokens[0] in tokens:
             start_idx = tokens.index(word_tokens[0])
             emb = last_hidden_states[0, start_idx].numpy() # شکل: (hidden_size,)
             embeddings.append(emb)
             labels.append(f"{word} (in: {sent[:20]}...)")
             # break # اگر فقط اولین ورودی هر کلمه در هر جمله مهم است، این را فعال کنید
             # اما اگر هر ورودی از هر کلمه در هر جمله مهم است، این را غیرفعال کنید
             # در این مثال، احتمالاً فقط یک بار هر کلمه در هر جمله ظاهر می‌شود،
             # بنابراین break تفاوت چندانی نمی‌کند، مگر اینکه کلمات تکراری باشند.
             # اما در هر صورت، این کد ممکن است کمتر از 6 embedding ایجاد کند اگر کلمات پیدا نشوند.

# --- اصلاح: اطمینان از وجود embedding برای نمایش ---
if len(embeddings) == 0:
    print("No embeddings found for the specified words in the sentences.")
else:
    print(f"Found {len(embeddings)} embeddings for visualization.")
    embeddings = np.array(embeddings)

    # --- اصلاح: تنظیم perplexity ---
    # تعداد نمونه‌ها (len(embeddings)) را چک کنید و perplexity مناسب انتخاب کنید
    # معمولاً perplexity باید کمتر از len(embeddings) باشد.
    # برای len(embeddings) = 6، مثلاً perplexity = min(5, len(embeddings) - 1) خوب است.
    calculated_perplexity = min(30, len(embeddings) - 1) # حداکثر 30، اما اگر کمتر از 6 بود، یکی کمتر از تعداد کل
    if calculated_perplexity <= 0:
        print("Not enough samples for t-SNE (need at least 2).")
    else:
        # اعمال t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=calculated_perplexity) # <-- perplexity اضافه شد
        embeddings_2d = tsne.fit_transform(embeddings)

        # نمایش نمودار
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        for i, label in enumerate(labels):
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9, ha='center')
        plt.title("t-SNE Visualization of BERT Embeddings for Selected Words")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid(True)
        plt.show()

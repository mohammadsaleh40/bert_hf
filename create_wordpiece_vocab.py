# create_wordpiece_vocab.py
from tokenizers import BertWordPieceTokenizer

# Initialize an empty tokenizer
# We will train it on our Persian text
tokenizer = BertWordPieceTokenizer(lowercase=False) # Persian is not cased like English, so set lowercase=False

# Path to your preprocessed text file
files = ["mirastext_preprocessed.txt"] # Use the output from step 1

# Train the tokenizer
# vocab_size: The target size of the vocabulary (e.g., 30000, 50000). Adjust based on your needs and data size.
# min_frequency: Minimum frequency a wordpiece must have to be included (e.g., 2, 5).
# limit_alphabet: Limits the number of unique characters considered initially (default is often 1000, suitable for many languages).
# initial_alphabet: Characters to always include (e.g., basic punctuation, digits).
# special_tokens: Default BERT special tokens ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] are usually added automatically,
#                 but you can specify them explicitly if needed.

# Define parameters
vocab_size = 320000  # Adjust this value based on your requirements
min_frequency = 5   # Adjust this value based on your requirements

# Train
# Removed arguments not supported by tokenizers.BertWordPieceTokenizer.train()
tokenizer.train(
    files=files,
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000, # Default, usually fine
    # Removed: clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False, wordpieces_prefix="##"
    # The BertWordPieceTokenizer from 'tokenizers' implicitly handles BERT-specific logic like '##' prefix and normalization during tokenization.
)

# Save the tokenizer files
# This will create vocab.txt (the vocabulary file) and tokenizer.json (configuration)
tokenizer.save_model(directory="persian_bert_tokenizer", prefix="wp")
# This will create 'wp-vocab.txt' and 'wp-tokenizer.json'
# Rename wp-vocab.txt to vocab.txt for BERT compatibility if needed, or use the generated name in config.

print("WordPiece vocabulary and tokenizer created successfully.")
print("Files saved in 'persian_bert_tokenizer' directory as 'wp-vocab.txt' and 'wp-tokenizer.json'.")
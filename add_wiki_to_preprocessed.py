
import json
import os
from pathlib import Path

def process_wiki_file(wiki_file_path, output_file_handle):
    """Process a single wiki JSON file and append text to the output file."""
    print(f"Processing: {wiki_file_path}")
    try:
        with open(wiki_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines in the JSON file
                try:
                    # Parse the JSON line
                    article_data = json.loads(line)
                    # Extract the 'text' field
                    text = article_data.get("text", "")
                    if text: # Only write if text exists and is not empty
                        # Write the text to the output file, followed by a newline
                        # Each Wikipedia article goes on a new line, separated by \n
                        # As per BERT pretraining format, documents should be separated by empty lines.
                        # Since each Wikipedia article is a distinct document, we write the text
                        # and then a newline. The 'add_preprocessing' script later adds empty lines
                        # between *all* lines (including MirasText and Wikipedia articles).
                        # This is acceptable for the HuggingFace Trainer's expected format.
                        output_file_handle.write(text + "\n")
                except json.JSONDecodeError as e:
                    print(f"  Warning: Could not parse JSON on line {line_num} in {wiki_file_path}: {e}")
                    continue # Skip invalid JSON lines
                except KeyError as e:
                    print(f"  Warning: Missing key {e} on line {line_num} in {wiki_file_path}, skipping article.")
                    continue # Skip if 'text' key is missing
    except FileNotFoundError:
        print(f"  Warning: File not found: {wiki_file_path}")
    except Exception as e:
        print(f"  Error processing file {wiki_file_path}: {e}")

def add_wikipedia_to_preprocessed(wikipedia_root_dir, preprocessed_file_path):
    """Add text from Wikipedia JSON files to the preprocessed MirasText file."""
    # Open the output file in append mode
    with open(preprocessed_file_path, 'a', encoding='utf-8') as output_file:
        # Walk through the wikipedia_root_dir (e.g., 'fawiki-latest-pages-articles/')
        for root, dirs, files in os.walk(wikipedia_root_dir):
            # Process only files starting with 'wiki_' (e.g., wiki_00, wiki_01, ..., wiki_99)
            wiki_files = [f for f in files if f.startswith('wiki_')]
            for wiki_file in sorted(wiki_files): # Sort for consistent processing order
                wiki_file_path = Path(root) / wiki_file
                process_wiki_file(wiki_file_path, output_file)

# --- تنظیمات ---
wikipedia_directory = "fawiki-latest-pages-articles" # نام پوشه اصلی حاوی زیرپوشه‌های AA, AB, ...
output_preprocessed_file = "mirastext_preprocessed.txt" # نام فایل خروجی MirasText پیش‌پردازش شده

# --- اجرای اصلی ---
if __name__ == "__main__":
    print(f"Starting to append Wikipedia data from '{wikipedia_directory}' to '{output_preprocessed_file}'...")
    add_wikipedia_to_preprocessed(wikipedia_directory, output_preprocessed_file)
    print(f"Finished appending Wikipedia data to '{output_preprocessed_file}'.")

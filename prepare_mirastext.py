# prepare_mirastext.py
import re

def parse_mirastext_line(line):
    """Parse a single line from MirasText dataset."""
    parts = line.strip().split(' *** ')
    if len(parts) < 6:
        # If a line doesn't have all 6 parts, handle accordingly
        # For now, we'll just return an empty content string
        return ""
    content, description, keywords, title, website, url = parts
    # You might want to process these parts further if needed
    # For pretraining, usually 'content' is the most important
    return content

def process_mirastext_file(input_file, output_file):
    """Process the MirasText file into a format suitable for BERT pretraining."""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            content = parse_mirastext_line(line)
            if content: # Only write if content exists and is not empty
                # Clean content if necessary (e.g., remove special characters added by the dataset)
                # Example: remove potential artifacts like 'nan'
                content = re.sub(r'\bnan\b', '', content)
                content = content.strip()
                if content: # Write only if content is not empty after cleaning
                    outfile.write(content + '\n')
                    # BERT pretraining script expects documents separated by empty lines.
                    # Each line here represents content potentially from one document/article.
                    # If content itself contains multiple paragraphs, they should ideally be separated by \n\n
                    # but BERT script usually handles sentence segmentation internally during data creation.
                    # A single newline separates sentences within a document for BERT.
                    # An empty line separates documents.
                    # Since each line in MirasText is one article, we add one newline for sentences
                    # within the article content and then an empty line to mark the end of the document.
                    # However, the script often treats a single newline as a sentence break within a sequence.
                    # The standard format for pretraining data fed to create_pretraining_data.py
                    # is one document per line, with sentences within the document separated by a single space or newline.
                    # Let's assume content is one logical document.
                    # Write the content and then an empty line to separate documents.
                    outfile.write('\n') # This adds the empty line between documents

if __name__ == "__main__":
    input_file_path = "MirasText_sample.txt" # Or the full path if not in the script's directory
    output_file_path = "mirastext_preprocessed.txt"
    process_mirastext_file(input_file_path, output_file_path)
    print(f"Preprocessing complete. Output saved to {output_file_path}")
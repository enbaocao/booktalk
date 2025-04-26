import re
import os
import argparse

def merge_incomplete_paragraphs(input_path, output_path):
    """
    Reads a text file, merges paragraphs that don't end with a period
    with the subsequent paragraph(s) until a period is found.

    Args:
        input_path (str): Path to the input text file.
        output_path (str): Path to save the processed text file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            text = infile.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Split into initial paragraphs based on one or more empty lines
    initial_paragraphs = re.split(r'\n\s*\n+', text)

    merged_paragraphs = []
    current_paragraph_parts = []

    for paragraph in initial_paragraphs:
        cleaned_paragraph = paragraph.strip()
        if not cleaned_paragraph:
            continue # Skip empty lines/paragraphs

        # Add the current cleaned paragraph part
        current_paragraph_parts.append(cleaned_paragraph)

        # Check if the *last* part added ends with a period
        # We check the last part specifically to handle cases like parenthetical additions
        # that might not end in periods themselves but follow a period-ending sentence.
        # A more robust check might look at the combined text, but let's stick to this logic first.
        # Revised logic: Check if the *combined* text ends with a period.
        combined_text = " ".join(current_paragraph_parts)

        # Consider other sentence-ending punctuation as well? For now, just period.
        if combined_text.endswith('.'):
            merged_paragraphs.append(combined_text)
            current_paragraph_parts = [] # Reset for the next merged paragraph

    # If there are remaining parts that didn't form a complete paragraph
    # (e.g., the file ends mid-sentence/paragraph)
    if current_paragraph_parts:
        merged_paragraphs.append(" ".join(current_paragraph_parts))

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n\n'.join(merged_paragraphs))
        print(f"Processed text saved to {output_path}")

    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge paragraphs in a text file until each ends with a period.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input text file (e.g., witching-3.txt)")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output text file.")

    args = parser.parse_args()

    merge_incomplete_paragraphs(args.input, args.output)
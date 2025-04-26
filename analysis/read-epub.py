import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os
import re

def epub_to_text(epub_path, output_combined_path):
    """
    Convert an EPUB file to a single text file containing non-quoted paragraphs
    and extracted quoted sentences. Skip headers and narrator tags.
    
    Args:
        epub_path: Path to the EPUB file
        output_combined_path: Path where the combined output text file will be saved
    """
    # Load the EPUB file
    book = epub.read_epub(epub_path)
    
    # Extract content from each document in the EPUB
    all_content = [] # List to store combined paragraphs and quotes
    start_processing = False # Flag to indicate when to start processing content
    narrator_tags = {"ore:", "eliot:", "29 barton road:"} # Set of tags to skip
    
    # Iterate through all items in the book
    for item in book.get_items():
        # Check if the item is a document (HTML content)
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Get the content of the document
            content = item.get_content()
            
            # Parse the HTML content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text from each paragraph
            paragraphs = soup.find_all('p')
            for paragraph in paragraphs:
                # Get clean text from paragraph
                text = paragraph.get_text().strip()

                # Check if we should start processing
                if not start_processing:
                    if text == "WHERE IS MIRANDA?":
                        start_processing = True
                    continue # Skip everything before the marker

                # Skip empty paragraphs and narrator tags
                if not text or text in narrator_tags:
                    continue
                
                # Regex to find text within standard double, single, and curly quotes
                quotes_found = re.findall(r'"(.*?)"|\'(.*?)\'|“(.*?)”|”(.*?)”', text)
                
                # Simple check for common list markers
                is_list_item = re.match(r'^\s*([*\-•+]|\d+\.|[a-zA-Z][.)])\s+', text)
                
                if not is_list_item: # Process only if not a list item
                    if quotes_found:
                        # If quotes are found, extract them and add each to all_content
                        for quote_tuple in quotes_found:
                            actual_quote = next((q for q in quote_tuple if q), None)
                            if actual_quote:
                                all_content.append(actual_quote.strip())
                        # Do not add the original paragraph containing quotes
                    else:
                        # No quotes found, add the paragraph to the main list
                        all_content.append(text)
    
    # Write all collected content (paragraphs and quotes) to the combined output file
    with open(output_combined_path, 'w', encoding='utf-8') as file:
        for content_item in all_content:
            file.write(content_item + '\n\n') # Double newline after each item for a blank line
    print(f"Combined content saved to {output_combined_path}")


if __name__ == "__main__":
    # Define input and output file paths
    epub_file = "witching.epub"
    combined_output_file = "witching_combined.txt" # Single output file
    
    # Check if input file exists
    if not os.path.exists(epub_file):
        print(f"Error: File '{epub_file}' not found.")
    else:
        # Convert EPUB to text and save combined content
        epub_to_text(epub_file, combined_output_file)
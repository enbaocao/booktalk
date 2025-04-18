import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os

def epub_to_text(epub_path, output_text_path):
    """
    Convert an EPUB file to a text file with paragraphs properly separated.
    
    Args:
        epub_path: Path to the EPUB file
        output_text_path: Path where the output text file will be saved
    """
    # Load the EPUB file
    book = epub.read_epub(epub_path)
    
    # Extract content from each document in the EPUB
    all_paragraphs = []
    
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
                if text:  # Only add non-empty paragraphs
                    all_paragraphs.append(text)
    
    # Write all paragraphs to the output file
    with open(output_text_path, 'w', encoding='utf-8') as file:
        for paragraph in all_paragraphs:
            file.write(paragraph + '\n\n')  # Double newline to separate paragraphs
    
    print(f"Conversion complete. Text saved to {output_text_path}")

if __name__ == "__main__":
    # Define input and output file paths
    epub_file = "witching.epub"
    output_file = "witching.txt"
    
    # Check if input file exists
    if not os.path.exists(epub_file):
        print(f"Error: File '{epub_file}' not found.")
    else:
        # Convert EPUB to text
        epub_to_text(epub_file, output_file)
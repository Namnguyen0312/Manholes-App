from PyPDF2 import PdfReader
from dotenv import load_dotenv
import json

import os

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def save_text_to_json(pdf_data, filename="extracted_data.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(pdf_data, file, ensure_ascii=False, indent=4)


def main():
    load_dotenv()

    # Define your PDF folder path here
    pdf_folder_path = 'pdf'

    # List all PDF files in the folder
    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

    # Dictionary to hold all PDF data
    pdf_data = {}

    # Extract and save each PDF's text in a dictionary
    for pdf_file in pdf_files:
        pdf_text = get_pdf_text([pdf_file])  # Extract text from each PDF
        pdf_data[os.path.basename(pdf_file)] = pdf_text  # Store in dictionary with filename as key

    # Save the dictionary to JSON file
    save_text_to_json(pdf_data)


if __name__ == '__main__':
    main()


import os
import sys
from src.exception import CustomException
from src.logger import logging
import PyPDF2

class DataExtraction:
    def __init__(self, books_dir,output_dir):
        self.books_dir = books_dir
        self.output_dir = output_dir

    def extract_text_from_pdf(self, file_path):
        logging.info("The text has been extracted from the books")
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in range(len(reader.pages)):
                    text += reader.pages[page].extract_text()
            return text
        except Exception as e:
            raise CustomException(e, sys)

    def save_extracted_text(self, text, filename):
        try:
            output_path = os.path.join(self.books_dir, filename)
            with open(output_path, 'w') as f:
                f.write(text)
            logging.info(f"Text extracted and saved to {output_path}")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_extractor = DataExtraction(books_dir="/Users/aryanmaheshwari/stepsai-test/books",output_dir="/Users/aryanmaheshwari/stepsai-test/processed_books")
    book_paths = [
        # "/Users/aryanmaheshwari/stepsai-test/books/Artificial Intelligence_ A Modern Approach.pdf",
        #"/Users/aryanmaheshwari/stepsai-test/books/Jared_Diamond-Guns_Germs_and_Steel.pdf",
        "/Users/aryanmaheshwari/stepsai-test/books/Selfish Gene.pdf"
    ]

    for book_path in book_paths:
        book_name = os.path.basename(book_path).replace('.pdf', '.txt')
        text = data_extractor.extract_text_from_pdf(book_path)
        data_extractor.save_extracted_text(text, book_name)

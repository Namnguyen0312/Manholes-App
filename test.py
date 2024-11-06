from PyPDF2 import PdfReader
reader = PdfReader('pdf/BXD_06-2021-TT-BXD_30062021.pdf')
meta = reader.metadata
print("Total Pages: ", len(reader.pages))
# All of the following could be None!
print("Author: ", meta.author)
print("Creator: ", meta.creator)
print("Producer: ", meta.producer)
print("Subject: ", meta.subject)
print("Title: ", meta.title)

page = reader.pages[5]
print(page.extract_text())
# Importing required modules
import PyPDF2
# Creating a pdf file object
file='/home/mayank_s/Documents/25862 Halsted Rd - Google Maps2.pdf'

from tika import parser

# rawText = parser.from_file(file)
#
# rawList = rawText['content'].splitlines()
1

import pdfplumber
pdf = pdfplumber.open(file)
page = pdf.pages[0]
text = page.extract_text()
print(text)
for data in text:
    print(data)
pdf.close()
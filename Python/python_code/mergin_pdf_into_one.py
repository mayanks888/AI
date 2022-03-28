import os
from PyPDF2 import PdfFileMerger
import natsort
x = [a for a in os.listdir("/home/mayank_sati/Documents/c++_notes") if a.endswith(".pdf")]
# x.sort()
x = natsort.natsorted(x, reverse=False)
merger = PdfFileMerger()
base="/home/mayank_sati/Documents/c++_notes"
base_path=os.path.join(base,)
for pdf in x:
    merger.append(open(os.path.join(base,pdf), 'rb'))

with open("result.pdf", "wb") as fout:
    merger.write(fout)

#
# from glob import glob
# from PyPDF2 import PdfFileMerger
#
#
#
# def pdf_merge():
#     ''' Merges all the pdf files in current directory '''
#     merger = PdfFileMerger()
#     allpdfs = [a for a in glob("*.pdf")]
#     [merger.append(pdf) for pdf in allpdfs]
#     with open("Merged_pdfs.pdf", "wb") as new_file:
#         merger.write(new_file)
#
#
# if __name__ == "__main__":
#     pdf_merge()
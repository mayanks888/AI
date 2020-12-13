import PyPDF2
pdfFileObject = open('/home/mayank_s/Documents/25962 Halsted Rd - Google Maps.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
count = pdfReader.numPages
# for i in range(count):
#     page = pdfReader.getPage(i)
#     print(page.extractText())
#
file='/home/mayank_s/Documents/25962 Halsted Rd - Google Maps.pdf'
#
# # import textract
# # text = textract.process(file, method='pdfminer')
# # 1



import sys
import PyPDF2
from PIL import Image

# if (len(sys.argv) != 2):
#     print("\nUsage: python {} input_file\n".format(sys.argv[0]))
#     sys.exit(1)

# pdf = sys.argv[1]

if __name__ == '__main__':
    input1 = PyPDF2.PdfFileReader(open(file, "rb"))
    # page0 = input1.getPage(1)
    count = pdfReader.numPages
    for i in range(count):
        page0 = pdfReader.getPage(i)
        if '/XObject' in page0['/Resources']:
            xObject = page0['/Resources']['/XObject'].getObject()

            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                    data = xObject[obj].getData()
                    if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                        mode = "RGB"
                    else:
                        mode = "P"

                    if '/Filter' in xObject[obj]:
                        if xObject[obj]['/Filter'] == '/FlateDecode':
                            img = Image.frombytes(mode, size, data)
                            img.save(obj[1:] + ".png")
                        elif xObject[obj]['/Filter'] == '/DCTDecode':
                            img = open(obj[1:] + ".jpg", "wb")
                            img.write(data)
                            img.close()
                        elif xObject[obj]['/Filter'] == '/JPXDecode':
                            img = open(obj[1:] + ".jp2", "wb")
                            img.write(data)
                            img.close()
                        elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                            img = open(obj[1:] + ".tiff", "wb")
                            img.write(data)
                            img.close()
                    else:
                        img = Image.frombytes(mode, size, data)
                        img.save(obj[1:] + ".png")
        else:
            print("No image found.")
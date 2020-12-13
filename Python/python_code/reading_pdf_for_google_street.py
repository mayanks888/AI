import os
import utm
import pdfplumber
import PyPDF2
from PIL import Image
input_folder="/home/mayank_s/Documents/gtreet"
save_image='/home/mayank_s/Desktop/template/googlestreet'
loop=0
for root, _, filenames in os.walk(input_folder):
    # if (len(filenames) == 0):
    #     print("Input folder is empty")
    #     # return 1
    for filename in filenames:
            loop=loop+1
            print(loop)
            print("Creating object detection for file : {fn}".format(fn=filename), '\n')
            file_path = (os.path.join(root, filename))
            pdf = pdfplumber.open(file_path)
            page = pdf.pages[0]
            text = page.extract_text()
            print(text)
            mydata=text.split("/")[6]
            mydata2=mydata.split(",")
            lat=float(mydata2[0].split("@")[-1])
            log=float(mydata2[1])
            x,y,_,_=utm.from_latlon(lat,log)
            string_name=str(x)+"_"+str(y)+".jpg"
            # for data in text:
            #     print(data)
            image_save=save_image+"/"+string_name
            pdf.close()
            ##############################################################
            # reading images
            pdfReader = PyPDF2.PdfFileReader(open(file_path, "rb"))
            # page0 = input1.getPage(1)
            count = pdfReader.numPages
            for i in range(count):
                    page0 = pdfReader.getPage(i)
                    if '/XObject' in page0['/Resources']:
                            xObject = page0['/Resources']['/XObject'].getObject()

                            for obj in xObject:
                                    if xObject[obj]['/Subtype'] == '/Image':
                                            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                                            if size[0]<1000:
                                                    continue
                                            data = xObject[obj].getData()
                                            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                                                    mode = "RGB"
                                            else:
                                                    mode = "P"

                                            if '/Filter' in xObject[obj]:
                                                    if xObject[obj]['/Filter'] == '/FlateDecode':
                                                            img = Image.frombytes(mode, size, data)
                                                            # img.save(obj[1:] + ".png")
                                                            img.save(image_save)

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
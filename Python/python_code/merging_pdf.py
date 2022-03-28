import PyPDF2

mergeFile = PyPDF2.PdfFileMerger()

mergeFile.append(PyPDF2.PdfFileReader('/media/mayank_sati/New Volume/study/CUDA/video/Udemy - CUDA programming Masterclass with C++ 2020-4/01 Introduction to CUDA programming and CUDA programming model/002 1.Introduction-to-parallel-programming.pptx', 'rb'))

mergeFile.append(PyPDF2.PdfFileReader('/media/mayank_sati/New Volume/study/CUDA/video/Udemy - CUDA programming Masterclass with C++ 2020-4/01 Introduction to CUDA programming and CUDA programming model/003 2.Introduction-to-hetrogeneous-parallel-computing-with-cuda.pptx', 'rb'))

mergeFile.write("NewMergedFile.pdf")
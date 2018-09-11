import pickle
import cv2
import _pickle as cPickle
# with open("100573400.pkl",'r' ,encoding="utf-8") as pickle_file:
with open("100573400.pkl",mode= 'rb',) as pickle_file:
    myfile=pickle.load(pickle_file,encoding='latin')
    print(myfile)

    cv2.imshow("myfilw",myfile)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 1a6ac4cb629c3b68b9c8fe1db822d396.pkl
# 100573400.pkl
# 100693900.pkl
#with open("100693900.pkl",mode= 'rb') as pickle_file:
#    myfile=cPickle.load(pickle_file, encoding='latin')
#    print(myfile)



######################################################################################
import cv2
import os
import shutil
import argparse
import time
import sys
import natsort

sys.path.append("..")

def make_video(file_path, fps=20, size=None, is_color=True, format="XVID", outvid='image_video.avi'):
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None

    images_list = os.listdir(file_path)
    images_list=natsort.natsorted(images_list, reverse=False)
    for file in images_list:
        image = (os.path.join(file_path, file))
        print("Image file : {fn} : Total file :{ct}".format(fn=file,ct=file), '\n')
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def create_video_file(input_folder,output_folder):

    if not os.path.exists(input_folder):
        print("Input folder not found")
        return 1

    if not os.path.exists(output_folder):
        print("Output folder not present. Creating New folder...")
        os.makedirs(output_folder)

    for root,_, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
            return 1

        # basepath='/home/mayank-s/Desktop/aptive_video_Diles/output_images'
        # # basepath='/home/mayank-s/Desktop/aptive_video_Diles/video_to_images_all'
        # frame_width = int(830)
        # frame_height = int(640)
        # output_path = (os.path.join(output_folder, "apt_video.mkv"))
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
        make_video(input_folder)

    ''' for file in range(1,1084):

             file_name="frame_"+str(file)+".jpg"
             file_path = (os.path.join(basepath, file_name))
             cap = cv2.VideoCapture()
             cap.open(file_path)
             frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
             print("Creating object detection on Image file : {fn}".format(fn=file_name), '\n')
             image = cv2.imread(file_path, 1)
             # image = cv2.imread(file_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
             out.write(image)

             # out.release()
             # cap.release()
             # cv2.destroyAllWindows()'''



def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Object detection')
    # path of input images
    parser.add_argument('--input_path', help="Input Folder", default='/home/mayank-s/Desktop/aptive_video_Diles/output_images')
    #path where video file will be created
    parser.add_argument('--output_path', help="Output folder", default='/home/mayank-s/Desktop/aptive_video_Diles/video')
    args = parser.parse_args()
    return args


args=parse_args()
print('\n',"Starting  conversion...","\n")

ret = create_video_file(args.input_path,args.output_path)#,maxframes=5)



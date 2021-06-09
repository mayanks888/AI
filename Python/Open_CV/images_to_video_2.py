
import os
import sys
import natsort
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
# sys.path.append("..")

def make_video(file_path, fps=20, size=None, is_color=True, format="XVID", outvid='image_video_2.avi'):
    # size=(540,960)
    # size=(1600,900)

    # VideoWriter_fourcc = cv2.CV_FOURCC('M', 'S', 'V', 'C')  # Microspoft Video 1
    fourcc = VideoWriter_fourcc(*format)
    # fourcc=CV_FOURCC('X', '2', '6', '4')
    # fourcc = VideoWriter_fourcc('M', 'S', 'V', 'C')
    vid = None

    images_list = os.listdir(file_path)
    images_list = natsort.natsorted(images_list, reverse=False)

    for file in images_list:
        image = (os.path.join(file_path, file))
        print("Image file : {fn} : Total file :{ct}".format(fn=file, ct=file), '\n')
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        ##################################33333
        if size is None:
            size = img.shape[1], img.shape[0]
            # vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        elif size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
            # img=cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            # img=cv2.resize(img, size, interpolation= cv2.INTER_CUBIC);

        if vid is None:
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        vid.write(img)
    vid.release()
    return vid


if __name__ == '__main__':
    # input_path='/home/mayank_s/Desktop/template/farm_2/farm_2_images2_scaled'
    input_path='/home/mayank_sati/Videos/fake'
    # ret = create_video_file(args.input_path, args.output_path)  # ,maxframes=5)
    make_video(input_path)
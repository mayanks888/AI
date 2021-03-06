######################################################################################
# A script to convert all mp4 videos in specified folder to jpg images with
# unique names in their respective folder.
#
# Can also optionally pass maxframes  as 3rd parameter. For example:
# Usage Example 1: Created total of 10 frames from the input video
#
# @author Mayank Sati/Ashis Samal
######################################################################################
import os
import shutil
import time
from datetime import datetime, timedelta

import cv2


def main(input_folder, output_folder, maxframes="None"):
    """Function to extract frames from input video file and save them as separate frames_folder in an output directory.

        Args:
            input_folder        : Input Video file directory.
            
            output_folder       : Output directory to save the frames.
            
            maxframes(optional) : Set the max number of frame ,(None :all the frames will be captured)

        Returns:
            None"""

    if not os.path.exists(input_folder):
        print("Input Video folder not found")
        return 1

    if not os.path.exists(output_folder):
        print("Output folder not present. Creating New folder")
        # shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    for root, directory, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
            continue
            # return 1
        for filename in filenames:
            try:
                file_path = (os.path.join(root, filename))
                Video_file_name = filename.split(".")[0]
                Gen_frame_path = (os.path.join(output_folder, Video_file_name))
                print()

                time_start = time.time()
                cap = cv2.VideoCapture()
                cap.open(file_path)
                if not cap.isOpened():
                    print("Failed to open input file : {fn}".format(fn=filename))
                    raise IOError  # ("Failed to open input file : {fn}".format(fn=filename))
                    # raise Invalid ("Invalid File")
                    # continue
                    # return 1

                print("Creating frames : {fn}".format(fn=filename))
                if os.path.exists(Gen_frame_path):
                    print("Removing existing {pt} output folder".format(pt=filename))
                    shutil.rmtree(Gen_frame_path)
                os.makedirs(Gen_frame_path)

                frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print("TotalFrame : {tf} - Frame_width : {fw} - Frame Height : {fh} - Frame Rate(FPS) : {fp} ".format(
                    tf=frameCount, fw=cap.get(cv2.CAP_PROP_FRAME_WIDTH), fh=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    fp=cap.get(cv2.CAP_PROP_FPS)))

                frameId = 0
                skipDelta = 0

                if not maxframes == "None":
                    if frameCount > maxframes:
                        skipDelta = (frameCount / maxframes)
                        # print ("Video has {fc}, but Maxframes is set to {mf}".format(fc=frameCount, mf=maxframes))
                        print("Maxframes is set to : {mf}".format(mf=maxframes))
                        print("Skip frames delta is : {d}".format(d=int(skipDelta)))
                    else:
                        print('Max frame {mf} cannot exceed total frame'.format(mf=maxframes))

                while frameId < frameCount:

                    ret, frame = cap.read()
                    # print frameId, ret, frame.shape
                    if not ret:
                        print("Failed to get the frame {f}".format(f=frameId))
                        continue

                    fname = "frame_" + str(frameId) + ".jpg"

                    # frame_folder=os.path.join(output_folder, fname)
                    ofname = os.path.join(Gen_frame_path, fname)
                    ret = cv2.imwrite(ofname, frame)

                    if not ret:
                        print("Failed to write the frame {f}".format(f=frameId))
                        continue

                    frameId += int(1 + skipDelta)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)

                time_end = time.time()
                sec = timedelta(seconds=int(time_end - time_start))
                d = datetime(1, 1, 1) + sec
                # print("DAYS:HOURS:MIN:SEC")
                # print("%d:%d:%d:%.2f" % (d.day - 1, d.hour, d.minute, d.second))
                print("Time Consumed - hours:{th} - Minutes:{mn} - Second:{sc}".format(th=d.hour, mn=d.minute,
                                                                                       sc=d.second))
                print("Output path :", Gen_frame_path)


            # except ValueError:
            #     print("")
            except IOError:
                print("Existing...")
            except:
                print('ERROR...Filename: {fn} cannot be converted into frames'.format(fn=filename))
            #     except IOError:
            #          print('Filename{fn} cannot be converted into frames, check file'.format(fn=filename))
            else:
                print("Frame generated successfully")


if __name__ == "__main__":

    Input_folder = "/home/mayank-s/PycharmProjects/Datasets/Video_to_frame/input"
    output_folder = '/home/mayank-s/PycharmProjects/Datasets/Video_to_frame/output'

    print("Start Video to Frames Converter...", "\n")

    ret = main(Input_folder, output_folder, 5)
    if ret == 1:
        print("Error in convering a file.....")
# exit(ret)

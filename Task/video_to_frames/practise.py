'''import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument("echo")
parser = argparse.ArgumentParser(description="Video2Frames converter")
parser.add_argument('input', metavar='<input_video_file>', help="Input video file")

args = parser.parse_args()
print (args.echo)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",
                    type=int)
args = parser.parse_args()
print (args.square**2)'''


'''import argparse
parser = argparse.ArgumentParser(description="Video2Frames converter")
parser.add_argument('input', metavar='<input_video_file>', help="Input video file")
parser.add_argument('output', metavar='<output_folder>', help="Output folder. If exists it will be removed")
parser.add_argument('--maxframes', type=int, help="Output max number of frames")
parser.add_argument('--rotate', type=int, choices={90, 180, 270}, help="Rotate clock-wise output frames")
parser.add_argument('--exifmodel', help="An example photo file to fill output meta-tags")
parser.add_argument('--verbose', action='store_true', help="Enable verbose")

args = parser.parse_args()


import os

input_video_file = '/home/mayank-s/PycharmProjects/Data_Science/Task/input/'

for root, directories, filenames in os.walk(input_video_file):
        for directory in directories:
            print (os.path.join(root, directory) )
        for filename in filenames:
             print (os.path.join(root,filename))'''

from datetime import datetime, timedelta

def GetTime():
    sec = timedelta(seconds=int(input('Enter the number of seconds: ')))
    d = datetime(1,1,1) + sec

    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))


GetTime()
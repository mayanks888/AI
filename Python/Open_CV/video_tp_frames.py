# Program To Read video
# and Extract Frames
import cv2


# Function to extract frames
def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        # Saves the frames with frame-count
        basepath='/home/mayank_sati/Videos/fake2'
        final_path=basepath+"/"+"frame___1%d.jpg" % count
        cv2.imwrite(final_path, image)

        count += 1
        print(count)


# Driver Code
if __name__ == '__main__':
    # Calling the function
    FrameCapture("/home/mayank_sati/Videos/minkunet_livox_tracking.mp4")
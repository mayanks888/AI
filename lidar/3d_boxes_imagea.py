import cv2
import numpy as np

img = cv2.imread('/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/testing/image_2/000000.png')
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))


# cv2.polylines(img,[pts],True,(0,255,255))
#
# #####################
# cv2.imshow('img',img)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()

###########################333

def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=3):
    qs = np.array([[689.6372, 146.1509],
                   [659.1182, 146.0869],
                   [659.1182, 177.8302],
                   [689.6372, 177.8352],
                   [679.1600, 149.2562],
                   [651.3864, 149.2032],
                   [651.3864, 178.0725],
                   [679.1600, 178.0766]])
    qs = np.array([[809.6938, 153.8009],
                   [783.4399, 153.8649],
                   [783.4399, 178.0894],
                   [809.6938, 178.0835],
                   [796.2969, 156.0498],
                   [772.2784, 156.1035],
                   [772.2784, 178.2924],
                   [796.2969, 178.2876]])
    # qs=np.array([[4.3222, -0.8966, 13.5610],
    #  [4.4295, -0.8966, 11.9792],
    #  [4.4295, 0.7155, 11.9792],
    #  [4.3222, 0.7155, 13.5610],
    #  [0.5369, -0.8966, 13.3040],
    #  [0.6442, -0.8966, 11.7222],
    #  [0.6442, 0.7155, 11.7222],
    #  [0.5369, 0.7155, 13.3040]])

    ''' Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
    1 -------- 0
    /| /|
    2 -------- 3 .
    | | | |
    . 5 -------- 4
    |/ |/
    6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (20, 160), (100, 160), (0, 0, 255), 10)
        # cv2.line(image, (689,146), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, 10)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, 10)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, 10)
        return image
    ##################################


draw_projected_box3d(img, 1)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

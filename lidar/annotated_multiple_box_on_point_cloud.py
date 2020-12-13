import mayavi.mlab as mlab
import numpy as np
import pandas as pd


def draw_lidar_simple(pc, location, color=None):
    ''' Draw lidar points. simplest set up. '''
    ##############################################33
    # location=[13,-1,0]

    # mlab.points3d(xmid, ymid, zmid, scale_factor=1.0)
    ################################################

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    # fig=draw_gt_boxes3d(1,fig)
    if color is None: color = pc[:, 2]
    # draw points
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='gnuplot', scale_factor=1,
                  figure=fig)
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    ###################################################3
    location = [24, 0, 1]
    x_loc = location[0]
    y_loc = location[1]
    z_loc = location[2]
    mlab.points3d(x_loc, y_loc, z_loc, scale_factor=1.0)
    ################################################
    ####################################################3
    mlab.show()

    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1),
                    color_list=None):
    gt_boxes3d = np.array([[4.8119, -1.9323, 39.7667],
                           [3.0897, -1.9323, 39.6927],
                           [3.0897, -0.1503, 39.6927],
                           [4.8119, -0.1503, 39.7667],
                           [4.6423, -1.9323, 43.7183],
                           [2.9200, -1.9323, 43.6443],
                           [2.9200, -0.1503, 43.6443],
                           [4.6423, -0.1503, 43.7183]])
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    gt_boxes3d = np.expand_dims(gt_boxes3d, 0)
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=None,
                        line_width=line_width, figure=fig)
    # mlab.show()
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


saving_path = "/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/pycharm_work/point_cloud_annotated/"
data = pd.read_csv("/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/pycharm_work/ppillar_bin.csv")
mydata = data.groupby('filename')
print(data.groupby('class').count())
len_group = mydata.ngroups
mygroup = mydata.groups
new = data.groupby(['filename'])['class'].count()
###############################################3333

x = data.iloc[:, 0].values
y = data.iloc[:, 8:11].values
##################################################
loop = 1
for da in mygroup.keys():
    print(da)
    index = mydata.groups[da].values
    ####################################################################################################
    # color=None
    # scan = np.fromfile(da,dtype=np.float32)
    #
    # points = scan.reshape(-1, 4)
    # pc=points
    # fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    # # fig=draw_gt_boxes3d(1,fig)
    # if color is None: color = pc[:, 2]
    # # draw points
    # mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='gnuplot', scale_factor=1,
    #               figure=fig)
    # # draw origin
    # mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    # # draw axis
    # axes = np.array([
    #     [2., 0., 0., 0.],
    #     [0., 2., 0., 0.],
    #     [0., 0., 2., 0.],
    # ], dtype=np.float64)
    # mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    ##########################################################################################################
    for read_index in index:
        # location = [24, 0, 1]
        x_loc = y[read_index][2]
        y_loc = y[read_index][1]
        z_loc = y[read_index][0]
        print(x_loc, y_loc, z_loc, )
        mlab.points3d(x_loc, y_loc, z_loc, scale_factor=1.0)
        ################################################
    mlab.show()

    ################################################################33

    #     print(index)
    #     top = (y[read_index][0], y[read_index][3])
    #     bottom = (y[read_index][2],y[read_index][1])
    #     cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    #     y[read_index][0]
    #     print(da)
    #     # break
    # cv2.imshow('streched_image', image_scale)
    # # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # # filepath=output_folder_path+my_image+".png"
    # # cv2.imwrite(filepath,my_image)
    # cv2.waitKey(100)
    # output_path = (os.path.join(saving_path, str(loop)+".jpg"))
    # loop+=1
    # cv2.imwrite(output_path, image_scale)
    # cv2.destroyAllWindows()
    # val += 1
    # print(val)

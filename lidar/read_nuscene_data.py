from math import sin, cos

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
from matplotlib import colors


def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def project_velo_to_ref(pts_3d_velo):
    pts_3d_velo = cart2hom(pts_3d_velo)  # nx4
    return np.dot(pts_3d_velo, np.transpose(V2C))


def project_ref_to_velo(pts_3d_ref):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))


def project_rect_to_ref(pts_3d_rect):
    ''' Input and Output are nx3 points '''
    # t1=np.linalg.inv(R0)
    # t2=pts_3d_rect
    # t3=np.dot(t1,t2)
    # t4 = np.transpose(t3)
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))


def project_ref_to_rect(pts_3d_ref):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))


def project_rect_to_velo(pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect)
    return project_ref_to_velo(pts_3d_ref)


def project_velo_to_rect(pts_3d_velo):
    pts_3d_ref = h.project_velo_to_ref(pts_3d_velo)
    return project_ref_to_rect(pts_3d_ref)


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def pointCloudToBirdsEyeView(ax2, velo, bb3d):
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-5, 35)
    hmax = velo[:, 2].max()
    hmin = velo[:, 2].min()
    hmean = velo[:, 2].mean()
    hmeadian = np.median(velo[:, 2])
    hstd = np.std(velo[:, 2])
    # print ('scalledh', hmax, hmean, hmeadian, hmin, hstd, scalledh.shape, scalledh[:10])
    norm = colors.Normalize(hmean - 2 * hstd, hmean + 2 * hstd, clip=True)
    sc2 = ax2.scatter(-velo[:, 1],
                      velo[:, 0],
                      s=1,
                      c=velo[:, 2],
                      cmap='viridis',
                      norm=norm,
                      marker=".",
                      )
    ax2.scatter(-bb3d[:, 1],
                bb3d[:, 0],
                c='red')
    ax2.set_facecolor('xkcd:grey')
    plt.colorbar(sc2)
    plt.waitforbuttonpress()


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
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
    # mlab.show()
    return fig


def draw_gt_boxes3d(gt_boxes3d, iIndex, fig, color=(1, 1, 1), line_width=1, draw_text=True, text_scale=(1, 1, 1),
                    color_list=None):
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%d' % iIndex, scale=text_scale, color=color, figure=fig)
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
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


# def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
#     ''' Draw lidar points
#     Args:
#         pc: numpy array (n,3) of XYZ
#         color: numpy array (n) of intensity or whatever
#         fig: mayavi figure handler, if None create new one otherwise will use it
#     Returns:
#         fig: created or used fig
#     '''
#     if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
#     if color is None: color = pc[:, 2]
#     mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
#                   scale_factor=pts_scale, figure=fig)
#
#     # draw origin
#     mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
#
#     # draw axis
#     axes = np.array([
#         [2., 0., 0., 0.],
#         [0., 2., 0., 0.],
#         [0., 0., 2., 0.],
#     ], dtype=np.float64)
#     mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
#     mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
#     mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
#
#     # draw fov (todo: update to real sensor spec.)
#     fov = np.array([  # 45 degree
#         [20., 20., 0., 0.],
#         [20., -20., 0., 0.],
#     ], dtype=np.float64)
#
#     mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
#                 figure=fig)
#     mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
#                 figure=fig)
#
#     # draw square region
#     TOP_Y_MIN = -20
#     TOP_Y_MAX = 20
#     TOP_X_MIN = 0
#     TOP_X_MAX = 40
#     TOP_Z_MIN = -2.0
#     TOP_Z_MAX = 0.4
#
#     x1 = TOP_X_MIN
#     x2 = TOP_X_MAX
#     y1 = TOP_Y_MIN
#     y2 = TOP_Y_MAX
#     mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
#     mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
#     mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
#     mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
#
#     # mlab.orientation_axes()
#     mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
#     mlab.show()
#     return fig
#
calib = '/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/testing/calib/000010.txt'
txt_path = "/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/testing/label_2/000010.txt"
bin_path = '/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/testing/velodyne_reduced/000010.bin'
# calibs = read_calib_file("000010_calib.txt")

# ++++++
# calibs = read_calib_file(calib)
# P = calibs['P2']
# P = np.reshape(P, [3,4])
# # Rigid transform from Velodyne coord to reference camera coord
# V2C = calibs['Tr_velo_to_cam']
# V2C = np.reshape(V2C, [3,4])
# C2V = inverse_rigid_trans(V2C)
# # Rotation from reference camera coord to rect camera coord
# R0 = calibs['R0_rect']
# R0 = np.reshape(R0,[3,3])
#
# # Camera intrinsics and extrinsics
# c_u = P[0,2]
# c_v = P[1,2]
# f_u = P[0,0]
# f_v = P[1,1]
# b_x = P[0,3]/(-f_u) # relative
# b_y = P[1,3]/(-f_v)
# #+++++++
# #scan = np.fromfile('n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin', dtype=np.float32)
scan = np.fromfile(bin_path, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]
pcd_data = points
fig = draw_lidar_simple(pcd_data)
# label_filename = txt_path
# lines = [line.rstrip() for line in open(label_filename)]
# iIndex=0
# for line in lines:
#     data = line.split(' ')
#     data[1:] = [float(x) for x in data[1:]]
#     type = data[0]  # 'Car', 'Pedestrian', ...
#     truncation = data[1]  # truncated pixel ratio [0..1]
#     occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
#     alpha = data[3]  # object observation angle [-pi..pi]
#     xmin = data[4]  # left
#     ymin = data[5]  # top
#     xmax = data[6]  # right
#     ymax = data[7]  # bottom
#     box2d = np.array([xmin, ymin, xmax, ymax])
h = data[8]  # box height
w = data[9]  # box width
l = data[10]  # box length (in meters)
t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
R = np.array([[+cos(ry), 0, +sin(ry)],
              [0, 1, 0],
              [-sin(ry), 0, +cos(ry)]])
x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
corners_3d[0, :] = corners_3d[0, :] + t[0]
corners_3d[1, :] = corners_3d[1, :] + t[1]
corners_3d[2, :] = corners_3d[2, :] + t[2]
corners_3d = np.transpose(corners_3d)
box3d_pts_3d_velo = project_rect_to_velo(corners_3d)
print(corners_3d)
draw_gt_boxes3d([box3d_pts_3d_velo], iIndex, fig=fig)
iIndex = iIndex + 1

mlab.show()


# f = plt.figure(figsize=plt.figaspect(0.5))
# ax2 = f.add_subplot(3, 1, 2, )  # projection="3d" )
# testp = [11.3, -2.95, -1.0]
# bb3d=[]
# bb3d.append(testp)
# bb3d = np.array(bb3d)
# pointCloudToBirdsEyeView(ax2, pcd_data, bb3d)

# fig = draw_lidar_simple(pcd_data)
def conv_loc_to_velo_pt(center, ry):
    R = np.array([[+cos(ry), 0, +sin(ry)], [0, 1, 0], [-sin(ry), 0, +cos(ry)]])
    x = center[0]
    y = center[1]
    z = center[2]

    corners_3d = np.dot(R, np.vstack([x, y, z]))

    corners_3d = np.transpose(corners_3d)
    # box3d_pts_3d_velo = project_rect_to_velo(corners_3d)
    mlab.dpoints(corners_3d[0], corners_3d[1], corners_3d[2], 0.5)
    mlab.show()

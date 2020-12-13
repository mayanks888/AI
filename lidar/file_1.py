import mayavi.mlab as mlab
import numpy as np


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    fig = draw_gt_boxes3d(1, fig)
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
#
# scan = np.fromfile('n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin', dtype=np.float32)
# scan = np.fromfile('./point_clouds/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin', dtype=np.float32)
# scan = np.fromfile('/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/testing/velodyne/000007.bin', dtype=np.float32)
scan = np.fromfile('/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/testing/velodyne_reduced/000000.bin',
                   dtype=np.float32)

# points = scan.reshape((-1, 5))[:, :4]
points = scan.reshape((-1, 4))
pcd_data = points
fig = draw_lidar_simple(pcd_data)
# draw_gt_boxes3d(1,pcd_data)


# def load_pcd_bin(file_name):
#         """
#         Loads RADAR data from a Point Cloud Data file to a list of lists (=points) and meta data.
#
#         Example of the header fields:
#         # .PCD v0.7 - Point Cloud Data file format
#         VERSION 0.7
#         FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
#         SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
#         TYPE F F F I I F F F F F I I I I I I I I
#         COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#         WIDTH 125
#         HEIGHT 1
#         VIEWPOINT 0 0 0 1 0 0 0
#         POINTS 125
#         DATA binary
#
#         :param file_name: The path of the pointcloud file.
#         :return: <np.float: 18, n>. Point cloud matrix.
#         """
#         meta = []
#         with open(file_name, 'rb') as f:
#             for line in f:
#                 line = line.strip().decode('utf-8')
#                 meta.append(line)
#                 if line.startswith('DATA'):
#                     break
#
#             data_binary = f.read()
#
#         # Get the header rows and check if they appear as expected.
#         assert meta[0].startswith('#'), 'First line must be comment'
#         assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
#         sizes = meta[3].split(' ')[1:]
#         types = meta[4].split(' ')[1:]
#         counts = meta[5].split(' ')[1:]
#         width = int(meta[6].split(' ')[1])
#         height = int(meta[7].split(' ')[1])
#         data = meta[10].split(' ')[1]
#         feature_count = len(types)
#         assert width > 0
#         assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
#         assert height == 1, 'Error: height != 0 not supported!'
#         assert data == 'binary'
#
#         # Lookup table for how to decode the binaries
#         unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
#                          'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
#                          'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
#         types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])
#
#         # Decode each point
#         offset = 0
#         point_count = width
#         points = []
#         for i in range(point_count):
#             point = []
#             for p in range(feature_count):
#                 start_p = offset
#                 end_p = start_p + int(sizes[p])
#                 assert end_p < len(data_binary)
#                 point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
#                 point.append(point_p)
#                 offset = end_p
#             points.append(point)
#
#         # A NaN in the first point indicates an empty pointcloud
#         point = np.array(points[0])
#         if np.any(np.isnan(point)):
#             return np.zeros((feature_count, 0))
#
#         # Convert to numpy matrix
#         points = np.array(points).transpose()
#         return points

# pcd_file = load_pcd_bin('n015-2018-08-01-16-41-59+0800__RADAR_FRONT__1533113398818573.pcd')

# datapath_file = 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pickle'
# dst = pickle.load(open(datapath_file, "rb"))
# t=0

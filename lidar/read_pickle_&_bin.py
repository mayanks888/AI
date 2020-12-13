import pickle

import matplotlib.pyplot as plt
import numpy as np


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


# datapath_file = f'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pickle'
# file_name ='n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin'

file_name = './point_clouds/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin'
datapath_file = './point_clouds/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pickle'
datapath_file = './point_clouds/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pickle'
boxes = pickle.load(open(datapath_file, "rb"))
# print(boxes.shape)
scan = np.fromfile(file_name, dtype=np.float32)
pcd_points = scan.reshape((-1, 5))[:, :4]
pcd_points = pcd_points.T
# axes_limit: float=40
# ax: Axes=None
# fig = draw_lidar(pcd_points)

if ax is None:
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
points = view_points(pcd_points[:3, :], np.eye(4), normalize=False)
dists = np.sqrt(np.sum(pcd_points[:2, :] ** 2, axis=0))
colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
ax.plot(0, 0, 'x', color='black')
c = np.array([0.0, 0.0, 0.9213])
for box in boxes:
    box.render(ax, view=np.eye(4), colors=(c, c, c))

# Limit visible range.
ax.set_xlim(-axes_limit, axes_limit)
ax.set_ylim(-axes_limit, axes_limit)
ax.axis('off')
# ax.set_title(sd_record['channel'])
ax.set_aspect('equal')
plt.waitforbuttonpress()

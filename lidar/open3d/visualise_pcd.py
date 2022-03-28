# examples/Python/Basic/pointcloud.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    v_path="/media/mayank_sati/New Volume/mayank_linux/roabag/livox/pcd/liv_435699658.pcd"
    v_path="/home/mayank_sati/Documents/442.499378300.pcd"
    v_path="/home/mayank_sati/Documents/datasets/nuscenes/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151608548020.pcd.bin"
    #####################################################333
    # v_path = '/home/mayank_sati/Downloads/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385098900804.pcd.bin'
    # v_path='/home/mayank_sati/Downloads/v1.0-mini/samples/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385095449675.pcd.bin'

    # points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape(-1, 5])
    ######################################################################333

    # points = np.fromfile(v_path, dtype=np.float32)
    # # lidar = np.fromstring(data, dtype=np.float32)
    # # points = lidar.reshape(-1, 6)
    # points = points.reshape(-1, 4)
    # points[:, 3] /= 255
    # # points = points[:, :4]
    # print("point.max", points.max())
    # print("point.min", points.min())
    ###########################################################
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(v_path)
    # print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10, :])
    print("")

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = o3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    o3d.visualization.draw_geometries([chair])
    print("")

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([chair])
    print("")